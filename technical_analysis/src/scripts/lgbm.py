
import sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

TA_ROOT    = Path(__file__).resolve().parents[2]
REPO_ROOT  = TA_ROOT.parent
DATA_DIR  = TA_ROOT / "data" / "processed"
MODEL_DIR = TA_ROOT / "src" / "models" / "lgbm"
IMP_PATH  = REPO_ROOT / "project-context" / "feature_importance.json"
REPORT    = REPO_ROOT / "project-context" / "FEATURE_SELECTION_REPORT.md"

TRAIN_PCT = 0.70
VAL_PCT   = 0.15
TRADE_TH  = 0.45
FEE       = 0.001

LGB_BASE = dict(
    boosting_type   = "gbdt",
    objective       = "binary",
    metric          = "binary_logloss",
    learning_rate   = 0.05,
    num_leaves      = 127,
    max_depth       = -1,
    min_child_samples = 50,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    verbose           = -1,
    random_state      = 42,
)
NUM_BOOST_ROUND  = 2000
EARLY_STOP       = 50


def _positive_features() -> list:
    data = json.loads(IMP_PATH.read_text())
    return [f for f, v in data if v >= 0]


_EXCL = {"open","high","low","close","volume","label","hit_bars",
         "market_return","trade_return","tp_price","entry_price",
         "sl_price","atr_used","reward_risk"}

def _fcols(df: pd.DataFrame, subset: list) -> list:
    s = set(subset)
    return [c for c in df.columns if c not in _EXCL and c in s]


def _load(stage: str, feat_cols: list) -> dict:
    lbl_file = ("labels_trade_1h.parquet" if stage == "trade"
                else "labels_direction_1h.parquet")
    feats = pd.read_parquet(DATA_DIR / "features_1h.parquet")
    feats.index = pd.to_datetime(feats.index)
    lbls  = pd.read_parquet(DATA_DIR / lbl_file)
    lbls.index = pd.to_datetime(lbls.index)

    common = feats.index.intersection(lbls.index)
    feats, lbls = feats.loc[common], lbls.loc[common]

    fc = _fcols(feats, feat_cols)
    n  = len(feats)
    te = int(n * TRAIN_PCT)
    ve = int(n * (TRAIN_PCT + VAL_PCT))

    X = feats[fc].values.astype("float32")
    y = lbls["label"].values.astype(np.int32)
    mr = lbls["trade_return"].values.astype("float32")

    return {
        "X_train": X[:te],  "y_train": y[:te],
        "X_val":   X[te:ve], "y_val":   y[te:ve],
        "X_test":  X[ve:],   "y_test":  y[ve:],
        "mr_test": mr[ve:],
        "fcols":   fc,
    }


def _load_market_ret() -> np.ndarray:
    orig = pd.read_parquet(DATA_DIR / "labels_1h.parquet")
    orig.index = pd.to_datetime(orig.index)
    feats = pd.read_parquet(DATA_DIR / "features_1h.parquet")
    feats.index = pd.to_datetime(feats.index)
    common = feats.index.intersection(orig.index)
    orig = orig.loc[common]
    n  = len(orig)
    ve = int(n * (TRAIN_PCT + VAL_PCT))
    return orig["market_return"].values.astype("float32")[ve:]


def _train_lgbm(stage: str, data: dict, params: dict) -> lgb.Booster:
    scale = (data["y_train"] == 0).sum() / max((data["y_train"] == 1).sum(), 1)
    p = {**params, "scale_pos_weight": scale}

    tr_ds  = lgb.Dataset(data["X_train"], label=data["y_train"])
    val_ds = lgb.Dataset(data["X_val"],   label=data["y_val"], reference=tr_ds)

    callbacks = [
        lgb.early_stopping(EARLY_STOP, verbose=False),
        lgb.log_evaluation(100),
    ]
    model = lgb.train(
        p, tr_ds,
        num_boost_round   = NUM_BOOST_ROUND,
        valid_sets        = [val_ds],
        callbacks         = callbacks,
    )
    print(f"  [{stage}] best iteration: {model.best_iteration}  "
          f"val_loss: {model.best_score['valid_0']['binary_logloss']:.4f}")
    return model


def _pnl(tp, dp, mr) -> dict:
    signals = np.where(tp >= TRADE_TH, np.where(dp >= 0.5, 1, -1), 0)
    mask    = signals != 0
    n       = int(mask.sum())
    if n == 0:
        return {"cum_pnl": 0.0, "n_trades": 0, "sharpe": 0.0,
                "win_rate": 0.0, "max_dd": 0.0}
    pnl  = signals[mask] * mr[mask] - FEE
    eq   = np.zeros(len(signals)); eq[mask] = pnl
    ceq  = np.cumsum(eq)
    dd   = float((ceq - np.maximum.accumulate(ceq)).min() * 100)
    sh   = float(eq.mean() / (eq.std() + 1e-9) * np.sqrt(8760))
    return {"cum_pnl": float(pnl.sum() * 100), "n_trades": n,
            "win_rate": float((pnl > 0).mean()), "sharpe": sh, "max_dd": dd}


def run():
    print(f"\n{'='*60}")
    print("LIGHTGBM PIPELINE  [two-stage: trade + direction]")
    print(f"{'='*60}\n")

    pos_feats = _positive_features()
    print(f"[1/5] Loading data ({len(pos_feats)} positive-importance features)...")
    td = _load("trade",     pos_feats)
    dd = _load("direction", pos_feats)
    print(f"  Train: {len(td['X_train']):,} bars  |  "
          f"Val: {len(td['X_val']):,}  |  Test: {len(td['X_test']):,}\n")

    print("[2/5] Training Stage 1: trade detection...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trade_m = _train_lgbm("trade", td, LGB_BASE)
    trade_m.save_model(str(MODEL_DIR / "trade.lgbm"))

    print("\n[3/5] Training Stage 2: direction...")
    dir_m = _train_lgbm("direction", dd, LGB_BASE)
    dir_m.save_model(str(MODEL_DIR / "direction.lgbm"))

    print("\n[4/5] Evaluating P&L on test set...")
    mr_test = _load_market_ret()

    tp = trade_m.predict(td["X_test"])
    dp = dir_m.predict(dd["X_test"])

    n = min(len(tp), len(dp), len(mr_test))
    metrics = _pnl(tp[-n:], dp[-n:], mr_test[-n:])

    print(f"  P&L={metrics['cum_pnl']:+.2f}%  trades={metrics['n_trades']:,}  "
          f"sharpe={metrics['sharpe']:.3f}  win_rate={metrics['win_rate']:.2%}  "
          f"max_dd={metrics['max_dd']:.2f}%")

    print("\n[5/5] Top 20 features by LightGBM gain (trade stage):")
    imp = pd.Series(
        trade_m.feature_importance(importance_type="gain"),
        index=td["fcols"]
    ).sort_values(ascending=False)
    for i, (feat, val) in enumerate(imp.head(20).items(), 1):
        print(f"  {i:2d}. {feat:<35s}  {val:,.0f}")

    print(f"\n{'='*60}")
    print("  MODEL COMPARISON (held-out test set)")
    print(f"{'='*60}")
    rows = [
        ("CNN-LSTM  128 feat",  1003.14, 10.471, 61.04,  7513),
        ("CNN-LSTM  102 feat",   899.06,  9.751, 61.30,  6571),
        ("CNN-LSTM   81 feat",  1261.28, 12.971, 63.61,  8441),
        (f"LightGBM   81 feat",
         metrics["cum_pnl"], metrics["sharpe"],
         metrics["win_rate"] * 100, metrics["n_trades"]),
    ]
    print(f"  {'Model':<24} {'P&L':>10} {'Sharpe':>8} {'WinRate':>9} {'Trades':>8}")
    print("  " + "-" * 64)
    for name, pnl, sh, wr, tr in rows:
        print(f"  {name:<24} {pnl:>+9.2f}%  {sh:>7.3f}  {wr:>8.2f}%  {tr:>7,}")

    best_pnl = max(r[1] for r in rows)
    winner   = next(r[0] for r in rows if r[1] == best_pnl)
    print(f"\n  → Winner: {winner.strip()}  (P&L={best_pnl:+.2f}%)")

    existing = REPORT.read_text(encoding="utf-8")
    section  = f"""
## 9. LightGBM Comparison (81 features)

Two-stage LightGBM (GBDT, num_leaves=127, lr=0.05) trained on the same
81 positive-importance features and evaluated on the identical held-out test period.

### Results

| Model | Cum P&L | Sharpe | Win Rate | Trades |
|-------|---------|--------|----------|--------|
| CNN-LSTM 128 feat | +1003.14% | 10.471 | 61.04% | 7,513 |
| CNN-LSTM 102 feat | +899.06% | 9.751 | 61.30% | 6,571 |
| CNN-LSTM 81 feat | +1261.28% | 12.971 | 63.61% | 8,441 |
| **LightGBM 81 feat** | **{metrics['cum_pnl']:+.2f}%** | **{metrics['sharpe']:.3f}** | **{metrics['win_rate']:.2%}** | **{metrics['n_trades']:,}** |

**Winner: {winner.strip()}**

### LightGBM Top 5 Features (by gain, trade stage)

{chr(10).join(f"- `{f}`: {v:,.0f}" for f, v in imp.head(5).items())}
"""
    REPORT.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")
    print(f"\n  Report updated → {REPORT}")

    return metrics


if __name__ == "__main__":
    run()
