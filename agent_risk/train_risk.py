import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "technical_analysis" / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "risk_lgbm.pkl"
META_PATH  = MODEL_DIR / "risk_meta.json"

TRAIN_PCT, VAL_PCT = 0.70, 0.15
DRAWDOWN_HORIZON   = 48
LOW_DD_THRESH  = 0.010
HIGH_DD_THRESH = 0.030

RISK_FEATURES = [
    "atr_7_pct", "atr_14_pct", "atr_21_pct", "atr_ratio_7_21",
    "bb_width_20_2", "bb_squeeze_20_2", "bb_width_20_1", "bb_squeeze_20_1",
    "bb_position_20_2", "bb_position_20_1",
    "hist_vol_12", "hist_vol_24", "hist_vol_72", "hist_vol_168", "vol_ratio_24_168",
    "vol_ratio_12", "vol_ratio_24", "vol_ratio_48", "vol_change",
    "rsi_7", "rsi_14", "rsi_21",
    "macd_hist_12_26", "macd_hist_5_13",
    "stoch_k_14", "stoch_d_14",
    "williams_r_14",
    "log_ret_1", "log_ret_3", "log_ret_6", "log_ret_12", "log_ret_24", "log_ret_48",
    "skew_24", "kurt_24", "skew_72", "kurt_72", "skew_168", "kurt_168",
    "ret_x_rvol", "vol_x_mom",
    "hl_range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "obv_slope_12", "obv_slope_24",
    "cmf_14", "mfi_14",
    "momentum_6", "momentum_12", "momentum_24",
    "htf4h_atr_ratio", "htf4h_hist_vol", "htf4h_vol_ratio",
    "htf1d_atr_ratio", "htf1d_hist_vol", "htf1d_vol_ratio",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

LABEL_MAP = {0: "low_risk", 1: "medium_risk", 2: "high_risk"}


def _compute_forward_drawdown(close: np.ndarray, horizon: int) -> np.ndarray:
    n = len(close)
    dd = np.zeros(n)
    for i in range(n - 1):
        end = min(i + horizon + 1, n)
        future = close[i + 1: end]
        if len(future) == 0:
            continue
        dd[i] = (close[i] - future.min()) / close[i]
    return dd


def _dd_to_label(dd: np.ndarray) -> np.ndarray:
    labels = np.ones(len(dd), dtype=int)
    labels[dd < LOW_DD_THRESH] = 0
    labels[dd >= HIGH_DD_THRESH] = 2
    return labels


def train():
    print("=" * 60)
    print("RISK MODEL TRAINING  (LightGBM 3-class)")
    print("=" * 60)

    feats = pd.read_parquet(DATA_DIR / "features_1h.parquet")
    feats.index = pd.to_datetime(feats.index)
    feats = feats.sort_index()

    orig = pd.read_parquet(DATA_DIR / "labels_1h.parquet")
    orig.index = pd.to_datetime(orig.index)

    common = feats.index.intersection(orig.index)
    feats = feats.loc[common]
    orig  = orig.loc[common]

    n = len(feats)
    train_end = int(n * TRAIN_PCT)
    val_end   = int(n * (TRAIN_PCT + VAL_PCT))

    available = [f for f in RISK_FEATURES if f in feats.columns]
    missing   = [f for f in RISK_FEATURES if f not in feats.columns]
    if missing:
        print(f"  Skipping {len(missing)} missing features: {missing}")
    print(f"  Using {len(available)} features")

    close = orig["entry_price"].values.astype(float)
    fwd_dd = _compute_forward_drawdown(close, DRAWDOWN_HORIZON)
    valid_n = n - DRAWDOWN_HORIZON

    X = feats[available].values.astype("float32")[:valid_n]
    y = _dd_to_label(fwd_dd[:valid_n])

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    print(f"\n  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    dist = {LABEL_MAP[k]: int((y_train == k).sum()) for k in range(3)}
    print(f"  Label dist (train): {dist}")

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=600,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    print("\n[Val set]")
    val_pred = model.predict(X_val)
    print(classification_report(y_val, val_pred, target_names=list(LABEL_MAP.values()), digits=4))

    print("[Test set]")
    test_pred = model.predict(X_test)
    print(classification_report(y_test, test_pred, target_names=list(LABEL_MAP.values()), digits=4))
    cm = confusion_matrix(y_test, test_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':>14} {'low_risk':>10} {'medium_risk':>12} {'high_risk':>10}")
    for i, lbl in LABEL_MAP.items():
        print(f"  {lbl:>14} {cm[i][0]:>10d} {cm[i][1]:>12d} {cm[i][2]:>10d}")

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "features": available,
        "label_map": LABEL_MAP,
        "drawdown_horizon": DRAWDOWN_HORIZON,
        "low_dd_thresh": LOW_DD_THRESH,
        "high_dd_thresh": HIGH_DD_THRESH,
        "n_estimators_used": model.best_iteration_ if model.best_iteration_ else model.n_estimators,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\nModel saved  : {MODEL_PATH}")
    print(f"Metadata     : {META_PATH}")
    return model, available


if __name__ == "__main__":
    train()
