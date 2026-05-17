
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

from feature_selection import _load_test, _infer, _pnl, _device  # noqa: E402
from cnn_lstm import CNNLSTMModel                                   # noqa: E402

N_SLICES = 6


def _load_ckpt(path: Path, use_reg: bool, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    n_feat = sd["cnn.0.weight"].shape[1]
    n_cls  = sd["fc_classifier.3.weight"].shape[0]
    m = CNNLSTMModel(input_features=n_feat, num_classes=n_cls,
                     use_regression_head=use_reg)
    m.load_state_dict(sd)
    return m.eval().to(device)


def run():
    device    = _device()
    pos_feats = [f for f, v in json.loads(IMP_PATH.read_text()) if v >= 0]

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD EVALUATION  ({N_SLICES} equal time slices)")
    print(f"{'='*60}\n")

    trade_m = _load_ckpt(MODEL_DIR / "trade"     / "best_model.pt", False, device)
    dir_m   = _load_ckpt(MODEL_DIR / "direction" / "best_model.pt", True,  device)

    ctx, mr, _, _, _ = _load_test(pos_feats)
    tp = _infer(trade_m, ctx, device)[:, 1]
    dp = _infer(dir_m,   ctx, device)[:, 1]
    n  = min(len(tp), len(dp), len(mr))
    tp, dp, mr = tp[-n:], dp[-n:], mr[-n:]

    full_m = _pnl(tp, dp, mr)
    print(f"  Full test ({n:,} bars):  P&L={full_m['cum_pnl']:+.2f}%  "
          f"Sharpe={full_m['sharpe']:.3f}  Trades={full_m['n_trades']:,}\n")

    slice_sz = n // N_SLICES
    print(f"  {'Slice':>5}  {'Bars':>5}  {'P&L':>10}  {'Sharpe':>8}  "
          f"{'WinRate':>8}  {'Trades':>8}  {'MaxDD':>9}")
    print("  " + "-" * 62)

    results = []
    for i in range(N_SLICES):
        s = i * slice_sz
        e = (i + 1) * slice_sz if i < N_SLICES - 1 else n
        m = _pnl(tp[s:e], dp[s:e], mr[s:e])
        results.append(m)
        wr = m["win_rate"] * 100
        sign = "+" if m["cum_pnl"] > 0 else " "
        print(f"  {i+1:>5}  {e-s:>5,}  {m['cum_pnl']:>+9.2f}%  {m['sharpe']:>8.3f}  "
              f"{wr:>8.2f}%  {m['n_trades']:>8,}  {m['max_dd']:>8.2f}%")

    pos_count = sum(1 for m in results if m["cum_pnl"] > 0)
    consistency = "CONSISTENT" if pos_count >= (N_SLICES * 2) // 3 else "MIXED"
    print(f"\n  Positive slices: {pos_count}/{N_SLICES}  →  {consistency}")

    _append_report(results, full_m, slice_sz, n, pos_count)
    return results


def _append_report(results, full_m, slice_sz, n_total, pos_count):
    existing = REPORT.read_text(encoding="utf-8")
    n_slices = len(results)

    header = ("| Slice | ~Bars | Cum P&L | Sharpe | Win Rate | Trades | Max DD |\n"
              "|-------|-------|---------|--------|----------|--------|--------|\n")
    rows = ""
    for i, m in enumerate(results):
        wr   = m["win_rate"] * 100
        bold = "**" if m["cum_pnl"] > 0 else ""
        rows += (f"| {i+1} | ~{slice_sz:,} | {bold}{m['cum_pnl']:+.2f}%{bold} "
                 f"| {m['sharpe']:.3f} | {wr:.2f}% | {m['n_trades']:,} | {m['max_dd']:.2f}% |\n")

    interp = (
        "Gains are broadly distributed across time — performance is not an artifact of one lucky window."
        if pos_count >= (n_slices * 2) // 3
        else
        "Gains are concentrated in certain periods. The aggregate backtest may overstate real-world robustness."
    )

    section = f"""
## 12. Walk-Forward Evaluation ({n_slices} time slices)

The test period ({n_total:,} bars, last 15%) split into {n_slices} equal consecutive windows.
Same trained CNN-LSTM (81 features) — no retraining. Each slice evaluated independently.

**Full test period**: P&L={full_m['cum_pnl']:+.2f}%  Sharpe={full_m['sharpe']:.3f}  Trades={full_m['n_trades']:,}

{header}{rows}
**Positive slices: {pos_count}/{n_slices}**

**Interpretation**: {interp}
"""
    REPORT.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")
    print(f"  Report updated → {REPORT}")


if __name__ == "__main__":
    run()
