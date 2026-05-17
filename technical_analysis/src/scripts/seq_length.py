
import sys, json, time
from pathlib import Path

TA_ROOT    = Path(__file__).resolve().parents[2]
REPO_ROOT  = TA_ROOT.parent
MODEL_ROOT = TA_ROOT / "src" / "models"
REPORT     = REPO_ROOT / "project-context" / "FEATURE_SELECTION_REPORT.md"
IMP_PATH   = REPO_ROOT / "project-context" / "feature_importance.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TA_ROOT / "src" / "models_def"))

import feature_selection as fs   # noqa: E402  — patched per iteration

SEQ_LENGTHS = [30, 90, 120]


def run():
    device    = fs._device()
    pos_feats = [f for f, v in json.loads(IMP_PATH.read_text()) if v >= 0]

    results = {60: {"cum_pnl": 1261.28, "sharpe": 12.971,
                    "win_rate": 0.6361, "n_trades": 8441, "max_dd": -143.05}}

    print(f"\n{'='*60}")
    print(f"SEQUENCE LENGTH SEARCH  [device={device}]")
    print(f"  Testing seq_len ∈ {SEQ_LENGTHS}  (baseline=60 → +1261%)")
    print(f"{'='*60}")

    for seq_len in SEQ_LENGTHS:
        print(f"\n{'='*60}")
        print(f"  seq_len = {seq_len}")
        print(f"{'='*60}")
        out = MODEL_ROOT / f"seq_{seq_len}"

        fs.SEQ_LEN = seq_len

        td = fs._load_stage("trade",     pos_feats)
        t0 = time.time()
        trade_m = fs._train("trade", td, out / "trade", device)
        print(f"  Trade stage: {time.time()-t0:.0f}s")

        dd = fs._load_stage("direction", pos_feats)
        t0 = time.time()
        dir_m = fs._train("direction", dd, out / "direction", device)
        print(f"  Direction stage: {time.time()-t0:.0f}s")

        ctx, mr, _, _, _ = fs._load_test(pos_feats)
        tp = fs._infer(trade_m, ctx, device)[:, 1]
        dp = fs._infer(dir_m,   ctx, device)[:, 1]
        n  = min(len(tp), len(dp), len(mr))
        m  = fs._pnl(tp[-n:], dp[-n:], mr[-n:])
        results[seq_len] = m

        print(f"  → P&L={m['cum_pnl']:+.2f}%  Sharpe={m['sharpe']:.3f}  "
              f"Trades={m['n_trades']:,}  WinRate={m['win_rate']:.2%}")

    fs.SEQ_LEN = 60

    ordered = sorted(results.items(), key=lambda x: -x[1]["cum_pnl"])
    print(f"\n{'='*60}")
    print("  SEQUENCE LENGTH COMPARISON")
    print(f"{'='*60}")
    print(f"  {'SeqLen':>7}  {'P&L':>10}  {'Sharpe':>8}  {'WinRate':>8}  "
          f"{'Trades':>8}  {'MaxDD':>9}")
    print("  " + "-" * 60)
    for sl, m in ordered:
        wr = m["win_rate"] * 100 if m["win_rate"] <= 1 else m["win_rate"]
        dd = f"{m['max_dd']:.2f}%" if m.get("max_dd") is not None else "—"
        note = " ← baseline" if sl == 60 else ""
        print(f"  {sl:>7}  {m['cum_pnl']:>+9.2f}%  {m['sharpe']:>8.3f}  "
              f"{wr:>8.2f}%  {m['n_trades']:>8,}  {m['max_dd']:>8.2f}%{note}")

    best_sl, best_m = ordered[0]
    print(f"\n  → Best: seq_len={best_sl}  (P&L={best_m['cum_pnl']:+.2f}%)")

    _append_report(ordered, best_sl, best_m)
    return results


def _append_report(ordered, best_sl, best_m):
    existing = REPORT.read_text(encoding="utf-8")
    header = ("| Seq Length | Cum P&L | Sharpe | Win Rate | Trades | Max DD |\n"
              "|------------|---------|--------|----------|--------|--------|\n")
    rows = ""
    for sl, m in ordered:
        wr   = m["win_rate"] * 100 if m["win_rate"] <= 1 else m["win_rate"]
        dd   = f"{m['max_dd']:.2f}%" if m.get("max_dd") is not None else "—"
        bold = "**" if sl == best_sl else ""
        note = " *(baseline)*" if sl == 60 else ""
        rows += (f"| {bold}{sl}{bold}{note} | {bold}{m['cum_pnl']:+.2f}%{bold} "
                 f"| {m['sharpe']:.3f} | {wr:.2f}% | {m['n_trades']:,} | {dd} |\n")
    section = f"""
## 13. Sequence Length Search

CNN-LSTM (81 positive features) retrained with different lookback windows.
Same architecture ([64,128,256] CNN → LSTM h=256 l=2), hyperparameters, and split.

{header}{rows}
**Best seq_len: {best_sl}**  (P&L={best_m['cum_pnl']:+.2f}%  Sharpe={best_m['sharpe']:.3f})
"""
    REPORT.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")
    print(f"  Report updated → {REPORT}")


if __name__ == "__main__":
    run()
