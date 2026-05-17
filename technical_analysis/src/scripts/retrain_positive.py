
import sys, json, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS   = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from feature_selection import (
    _load_stage, _load_test, _train, _infer, _pnl, _device,
    MODEL_ROOT, REPORT_OUT,
)

IMP_PATH = REPO_ROOT / "project-context" / "feature_importance.json"


def _positive_features() -> list:
    data = json.loads(IMP_PATH.read_text())
    return [f for f, v in data if v >= 0]


def run():
    device = _device()
    pos_feats = _positive_features()
    n_pos = len(pos_feats)

    print(f"\n{'='*60}")
    print(f"POSITIVE-IMPORTANCE RETRAIN  [device={device}]")
    print(f"{'='*60}")
    print(f"Features with importance ≥ 0: {n_pos}")
    print(f"Dropped (negative importance): {128 - n_pos}\n")

    out_dir = MODEL_ROOT / "positive_only"

    print("[1/4] Training positive-only model (Stage 1: trade)...")
    td = _load_stage("trade", pos_feats)
    t0 = time.time()
    trade_m = _train("trade", td, out_dir / "trade", device)
    print(f"  Done in {time.time()-t0:.0f}s\n")

    print("[2/4] Training positive-only model (Stage 2: direction)...")
    dd = _load_stage("direction", pos_feats)
    t0 = time.time()
    dir_m = _train("direction", dd, out_dir / "direction", device)
    print(f"  Done in {time.time()-t0:.0f}s\n")

    print("[3/4] Evaluating positive-only model P&L on test set...")
    ctx, mr, _, _, _ = _load_test(pos_feats)
    tp = _infer(trade_m, ctx, device)[:, 1]
    dp = _infer(dir_m,   ctx, device)[:, 1]
    pos_m = _pnl(tp, dp, mr)
    print(f"  P&L={pos_m['cum_pnl']:+.2f}%  trades={pos_m['n_trades']:,}  "
          f"sharpe={pos_m['sharpe']:.3f}  win_rate={pos_m['win_rate']:.2%}")

    print(f"\n{'='*60}")
    print("  THREE-WAY COMPARISON")
    print(f"{'='*60}")
    expanded_pnl = 1003.14
    reduced_pnl  =  899.06
    print(f"  Expanded      (128 feat): P&L=+{expanded_pnl:.2f}%  Sharpe=10.471")
    print(f"  Bottom-20% pruned (102): P&L=+{reduced_pnl:.2f}%  Sharpe=9.751")
    print(f"  Positive-only  ({n_pos:3d} feat): "
          f"P&L={pos_m['cum_pnl']:+.2f}%  Sharpe={pos_m['sharpe']:.3f}")
    best = max(expanded_pnl, reduced_pnl, pos_m["cum_pnl"])
    winner = ("Expanded (128)" if best == expanded_pnl
              else "Bottom-20%-pruned (102)" if best == reduced_pnl
              else f"Positive-only ({n_pos})")
    print(f"  → Winner: {winner}")
    print()

    print("[4/4] Appending results to FEATURE_SELECTION_REPORT.md...")
    existing = REPORT_OUT.read_text(encoding="utf-8")
    neg_feats = [f for f in _all_features() if f not in set(pos_feats)]

    section = f"""
## 8. Positive-Importance Retrain (importance ≥ 0)

**Rationale**: {128 - n_pos} features have *negative* permutation importance — shuffling them
*improves* P&L, meaning the model learned spurious patterns from them.
Dropping all negatively-important features gives a cleaner, more principled feature set.

- Features dropped (negative importance): **{128 - n_pos}**
- Features kept (importance ≥ 0): **{n_pos}**

**Dropped features:**

{", ".join(f"`{f}`" for f in sorted(neg_feats))}

### Three-Way Comparison

| Metric | Expanded (128) | Bottom-20% Pruned (102) | Positive-Only ({n_pos}) |
|--------|---------------|------------------------|------------------------|
| Trades | 7,513 | 6,571 | {pos_m['n_trades']:,} |
| Win rate | 61.04% | 61.30% | {pos_m['win_rate']:.2%} |
| **Cum P&L** | **+{expanded_pnl:.2f}%** | **+{reduced_pnl:.2f}%** | **{pos_m['cum_pnl']:+.2f}%** |
| Sharpe (ann.) | 10.471 | 9.751 | {pos_m['sharpe']:.3f} |
| Max drawdown | -143.05% | -120.30% | {pos_m['max_dd']:.2f}% |

**Winner: {winner}**
"""
    REPORT_OUT.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")
    print(f"  Report updated → {REPORT_OUT}")

    return pos_m


def _all_features() -> list:
    data = json.loads(IMP_PATH.read_text())
    return [f for f, _ in data]


if __name__ == "__main__":
    run()
