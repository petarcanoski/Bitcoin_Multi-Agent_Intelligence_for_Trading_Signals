import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_DIR   = REPO_ROOT / "technical_analysis" / "data" / "processed"
MODEL_PATH = REPO_ROOT / "agent_risk" / "models" / "risk_lgbm.pkl"
META_PATH  = REPO_ROOT / "agent_risk" / "models" / "risk_meta.json"

TRAIN_PCT, VAL_PCT     = 0.70, 0.15
DRAWDOWN_HORIZON       = 48
DRAWDOWN_THRESHOLD     = 0.03
RISK_LOW_PCT, RISK_HIGH_PCT = 30, 70

LABEL_MAP = {0: "low_risk", 1: "medium_risk", 2: "high_risk"}


def _split_indices(n: int):
    return int(n * TRAIN_PCT), int(n * (TRAIN_PCT + VAL_PCT))


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


def _heuristic_signal(hist_vol: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    signal = np.full(len(hist_vol), "medium_risk", dtype=object)
    signal[hist_vol <= low_pct] = "low_risk"
    signal[hist_vol > high_pct] = "high_risk"
    return signal


def _load_model():
    if not MODEL_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return model, meta


def evaluate(force_heuristic: bool = False) -> dict:
    print("=" * 60)
    print("RISK AGENT EVALUATION")
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
    train_end, val_end = _split_indices(n)

    close  = orig["entry_price"].values.astype(float)
    fwd_dd = _compute_forward_drawdown(close, DRAWDOWN_HORIZON)
    valid  = n - DRAWDOWN_HORIZON

    test_dd = fwd_dd[val_end:valid]
    test_ts = feats.index[val_end:valid]
    severe  = test_dd >= DRAWDOWN_THRESHOLD

    print(f"\nTest period   : {test_ts[0].date()} → {test_ts[-1].date()}")
    print(f"Test bars     : {len(test_ts):,}")
    print(f"Severe drawdowns (>{DRAWDOWN_THRESHOLD:.0%} in {DRAWDOWN_HORIZON}h): "
          f"{severe.sum():,} / {len(severe):,} ({severe.mean():.1%})")

    model, meta = _load_model()

    if model is not None and not force_heuristic:
        features = meta.get("features", [])
        available = [f for f in features if f in feats.columns]
        X_test = feats[available].iloc[val_end:valid].values.astype("float32")
        pred_int = model.predict(X_test)
        risk_pred = np.array([LABEL_MAP[int(p)] for p in pred_int])
        mode = "LightGBM model"
        print(f"\nUsing trained LightGBM model ({len(available)} features)")
    else:
        vol_col = feats["atr_14_pct"].values if "atr_14_pct" in feats.columns else feats["hist_vol_24"].values
        train_vol = vol_col[:train_end]
        low_t  = float(np.nanpercentile(train_vol, RISK_LOW_PCT))
        high_t = float(np.nanpercentile(train_vol, RISK_HIGH_PCT))
        test_vol = vol_col[val_end:valid]
        risk_pred = _heuristic_signal(test_vol, low_t, high_t)
        mode = "ATR-14 percentile heuristic"
        print(f"\nUsing heuristic: {mode}")

    dist = Counter(risk_pred)
    print("\nPredicted risk distribution:")
    for lvl in ["low_risk", "medium_risk", "high_risk"]:
        print(f"  {lvl:>14}: {dist[lvl]:5,}  ({dist[lvl]/len(risk_pred):.1%})")

    pred_binary = (risk_pred == "high_risk").astype(int)
    true_binary = severe.astype(int)

    print(f"\n[Binary] high_risk → actual severe drawdown:")
    print(classification_report(true_binary, pred_binary,
                                target_names=["no_severe_dd", "severe_dd"], digits=4))
    cm_bin = confusion_matrix(true_binary, pred_binary)

    avg_dd_by_risk   = {}
    pct_severe_by_risk = {}
    print("[Risk Gating] Forward drawdown by predicted risk level:")
    for lvl in ["low_risk", "medium_risk", "high_risk"]:
        mask = risk_pred == lvl
        if mask.sum() == 0:
            continue
        avg_dd = float(test_dd[mask].mean()) * 100
        pct_sev = float(severe[mask].mean()) * 100
        med_dd  = float(np.median(test_dd[mask])) * 100
        avg_dd_by_risk[lvl]    = avg_dd
        pct_severe_by_risk[lvl] = pct_sev
        print(f"  {lvl:>14}: avg_fwd_dd={avg_dd:.2f}%  "
              f"pct_severe={pct_sev:.1f}%  median={med_dd:.3f}%")

    report_dict = classification_report(
        true_binary, pred_binary,
        target_names=["no_severe_dd", "severe_dd"],
        digits=4, output_dict=True,
    )

    return {
        "methodology": {
            "mode": mode,
            "drawdown_horizon_bars": DRAWDOWN_HORIZON,
            "severe_drawdown_threshold_pct": DRAWDOWN_THRESHOLD * 100,
            "vol_proxy": "LightGBM on 62 features" if model else "atr_14_pct percentile",
        },
        "test_period": {
            "start": str(test_ts[0].date()),
            "end": str(test_ts[-1].date()),
            "n_bars": len(test_ts),
        },
        "severe_drawdown_rate_pct": float(severe.mean()) * 100,
        "risk_distribution": {k: int(v) for k, v in dist.items()},
        "avg_fwd_drawdown_by_risk_pct": avg_dd_by_risk,
        "pct_severe_by_risk": pct_severe_by_risk,
        "binary_classification": {
            "confusion_matrix": cm_bin.tolist(),
            "report": report_dict,
        },
        "_arrays": {
            "test_ts": [str(ts) for ts in test_ts],
            "risk_pred": risk_pred.tolist(),
        },
    }


if __name__ == "__main__":
    import sys
    force_heuristic = "--heuristic" in sys.argv
    evaluate(force_heuristic=force_heuristic)
