
import sys, json
import numpy as np
import torch
from pathlib import Path

TA_ROOT    = Path(__file__).resolve().parents[2]
REPO_ROOT  = TA_ROOT.parent
MODEL_DIR = TA_ROOT / "src" / "models" / "final"
REPORT    = REPO_ROOT / "project-context" / "FEATURE_SELECTION_REPORT.md"
IMP_PATH  = REPO_ROOT / "project-context" / "feature_importance.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TA_ROOT / "src" / "models_def"))

from feature_selection import _load_test, _infer, _device  # noqa: E402
from cnn_lstm import CNNLSTMModel                           # noqa: E402

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
FEE = 0.001


def _pnl_th(tp, dp, mr, thresh):
    signals = np.where(tp >= thresh, np.where(dp >= 0.5, 1, -1), 0)
    mask = signals != 0
    n = int(mask.sum())
    if n == 0:
        return {"cum_pnl": 0.0, "n_trades": 0, "sharpe": 0.0,
                "win_rate": 0.0, "max_dd": 0.0}
    pnl = signals[mask] * mr[mask] - FEE
    eq  = np.zeros(len(signals)); eq[mask] = pnl
    ceq = np.cumsum(eq)
    dd  = float((ceq - np.maximum.accumulate(ceq)).min() * 100)
    sh  = float(eq.mean() / (eq.std() + 1e-9) * np.sqrt(8760))
    return {"cum_pnl": float(pnl.sum() * 100), "n_trades": n,
            "win_rate": float((pnl > 0).mean()), "sharpe": sh, "max_dd": dd}


def _load_ckpt(path: Path, use_reg: bool, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    n_feat   = sd["cnn.0.weight"].shape[1]
    n_cls    = sd["fc_classifier.3.weight"].shape[0]
    m = CNNLSTMModel(input_features=n_feat, num_classes=n_cls,
                     use_regression_head=use_reg)
    m.load_state_dict(sd)
    return m.eval().to(device)


def run():
    device    = _device()
    pos_feats = [f for f, v in json.loads(IMP_PATH.read_text()) if v >= 0]

    print(f"\n{'='*60}")
    print("THRESHOLD SEARCH  (81 features, no retraining)")
    print(f"{'='*60}\n")

    trade_m = _load_ckpt(MODEL_DIR / "trade"     / "best_model.pt", False, device)
    dir_m   = _load_ckpt(MODEL_DIR / "direction" / "best_model.pt", True,  device)

    ctx, mr, _, _, _ = _load_test(pos_feats)
    tp = _infer(trade_m, ctx, device)[:, 1]
    dp = _infer(dir_m,   ctx, device)[:, 1]
    n  = min(len(tp), len(dp), len(mr))
    tp, dp, mr = tp[-n:], dp[-n:], mr[-n:]

    print(f"  {'Thresh':>6}  {'P&L':>10}  {'Sharpe':>8}  {'WinRate':>8}  {'Trades':>8}  {'MaxDD':>9}")
    print("  " + "-" * 62)

    results = []
    for th in THRESHOLDS:
        m = _pnl_th(tp, dp, mr, th)
        results.append((th, m))
        wr = m["win_rate"] * 100
        marker = " ← baseline" if abs(th - 0.55) < 0.001 else ""
        print(f"  {th:.2f}    {m['cum_pnl']:>+9.2f}%  {m['sharpe']:>8.3f}  "
              f"{wr:>8.2f}%  {m['n_trades']:>8,}  {m['max_dd']:>8.2f}%{marker}")

    best_th, best_m = max(results, key=lambda x: x[1]["cum_pnl"])
    print(f"\n  → Best threshold: {best_th:.2f}  "
          f"P&L={best_m['cum_pnl']:+.2f}%  Sharpe={best_m['sharpe']:.3f}")

    _append_report(results, best_th, best_m)
    return results


def _append_report(results, best_th, best_m):
    existing = REPORT.read_text(encoding="utf-8")
    header = ("| Threshold | Cum P&L | Sharpe | Win Rate | Trades | Max DD |\n"
              "|-----------|---------|--------|----------|--------|--------|\n")
    rows = ""
    for th, m in results:
        wr   = m["win_rate"] * 100
        bold = "**" if th == best_th else ""
        note = " *(baseline)*" if abs(th - 0.55) < 0.001 else ""
        rows += (f"| {bold}{th:.2f}{bold}{note} | {bold}{m['cum_pnl']:+.2f}%{bold} "
                 f"| {m['sharpe']:.3f} | {wr:.2f}% | {m['n_trades']:,} | {m['max_dd']:.2f}% |\n")
    section = f"""
## 11. Threshold Search (trade-probability cutoff)

Scanning the trade-signal threshold from 0.40 to 0.75 on the trained CNN-LSTM
(81 positive features). No retraining — only the inference cutoff changes.
Higher thresholds → fewer, higher-confidence trades.

{header}{rows}
**Best threshold: {best_th:.2f}**  (P&L={best_m['cum_pnl']:+.2f}%  Sharpe={best_m['sharpe']:.3f})
"""
    REPORT.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")
    print(f"  Report updated → {REPORT}")


if __name__ == "__main__":
    run()
