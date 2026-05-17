
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "technical_analysis" / "data" / "processed"

TRAIN_PCT, VAL_PCT = 0.70, 0.15
DRAWDOWN_HORIZON = 48
DRAWDOWN_THRESHOLD = 0.03
RISK_LOW_PCT = 30
RISK_HIGH_PCT = 70


def _split_indices(n: int):
    train_end = int(n * TRAIN_PCT)
    val_end = int(n * (TRAIN_PCT + VAL_PCT))
    return train_end, val_end


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


def _volatility_risk_signal(hist_vol: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    signal = np.full(len(hist_vol), "medium_risk", dtype=object)
    signal[hist_vol <= low_pct] = "low_risk"
    signal[hist_vol > high_pct] = "high_risk"
    return signal


def evaluate() -> dict:
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
    orig = orig.loc[common]

    n = len(feats)
    train_end, val_end = _split_indices(n)

    if "atr_14_pct" in feats.columns:
        vol_col = feats["atr_14_pct"].values
        vol_label = "atr_14_pct (ATR-14 as % of price)"
    else:
        vol_col = feats["hist_vol_24"].values
        vol_label = "hist_vol_24"

    train_vol = vol_col[:train_end]
    low_thresh = float(np.nanpercentile(train_vol, RISK_LOW_PCT))
    high_thresh = float(np.nanpercentile(train_vol, RISK_HIGH_PCT))

    print(f"\nVol proxy     : {vol_label}")
    print(f"Low threshold : ≤ {low_thresh:.5f} ({RISK_LOW_PCT}th pct of training)")
    print(f"High threshold: > {high_thresh:.5f} ({RISK_HIGH_PCT}th pct of training)")

    close = orig["entry_price"].values.astype(float)
    fwd_dd = _compute_forward_drawdown(close, DRAWDOWN_HORIZON)

    test_vol = vol_col[val_end:]
    test_dd = fwd_dd[val_end:]
    test_ts = feats.index[val_end:]

    print(f"\nTest period   : {test_ts[0].date()} → {test_ts[-1].date()}")
    print(f"Test bars     : {len(test_vol):,}")

    valid = len(test_vol) - DRAWDOWN_HORIZON
    test_vol = test_vol[:valid]
    test_dd = test_dd[:valid]
    test_ts = test_ts[:valid]

    risk_pred = _volatility_risk_signal(test_vol, low_thresh, high_thresh)
    severe = test_dd >= DRAWDOWN_THRESHOLD

    print(f"\nSevere drawdowns (>{DRAWDOWN_THRESHOLD:.0%} in {DRAWDOWN_HORIZON}h): "
          f"{severe.sum():,} / {len(severe):,} ({severe.mean():.1%})")

    dist = Counter(risk_pred)
    print("Predicted risk distribution:")
    for lvl in ["low_risk", "medium_risk", "high_risk"]:
        print(f"  {lvl:>14}: {dist[lvl]:5,}  ({dist[lvl]/len(risk_pred):.1%})")

    pred_binary = (risk_pred == "high_risk").astype(int)
    true_binary = severe.astype(int)

    print(f"\n[Binary] high_risk vs actual severe drawdown:")
    print(classification_report(true_binary, pred_binary,
                                target_names=["no_severe_dd", "severe_dd"], digits=4))
    cm_bin = confusion_matrix(true_binary, pred_binary)

    avg_dd_by_risk = {}
    pct_severe_by_risk = {}
    for lvl in ["low_risk", "medium_risk", "high_risk"]:
        mask = risk_pred == lvl
        if mask.sum() == 0:
            continue
        avg_dd = float(test_dd[mask].mean()) * 100
        pct_severe = float(severe[mask].mean()) * 100
        avg_dd_by_risk[lvl] = avg_dd
        pct_severe_by_risk[lvl] = pct_severe
        print(f"  {lvl:>14}: avg_fwd_dd={avg_dd:.2f}%  "
              f"pct_severe={pct_severe:.1f}%")

    print("\n[Risk Gating] Forward drawdown by risk level:")
    for lvl in ["low_risk", "medium_risk", "high_risk"]:
        mask = risk_pred == lvl
        if mask.sum() == 0:
            continue
        med_dd = float(np.median(test_dd[mask])) * 100
        print(f"  {lvl:>14}: median_fwd_dd={med_dd:.3f}%")

    one_bar_ret = np.diff(close[val_end: val_end + valid + 1]) / close[val_end: val_end + valid]
    if len(one_bar_ret) == len(risk_pred):
        print("\n[1-bar forward return] by risk level:")
        for lvl in ["low_risk", "medium_risk", "high_risk"]:
            mask = risk_pred == lvl
            if mask.sum() == 0:
                continue
            avg_ret = float(one_bar_ret[mask].mean()) * 100
            pct_pos = float((one_bar_ret[mask] > 0).mean()) * 100
            print(f"  {lvl:>14}: avg_ret={avg_ret:+.4f}%  pct_positive={pct_pos:.1f}%")

    report_dict = classification_report(
        true_binary, pred_binary,
        target_names=["no_severe_dd", "severe_dd"],
        digits=4, output_dict=True,
    )

    return {
        "methodology": {
            "vol_proxy": vol_label,
            "low_threshold": low_thresh,
            "high_threshold": high_thresh,
            "drawdown_horizon_bars": DRAWDOWN_HORIZON,
            "severe_drawdown_threshold_pct": DRAWDOWN_THRESHOLD * 100,
        },
        "test_period": {
            "start": str(test_ts[0].date()),
            "end": str(test_ts[-1].date()),
            "n_bars": len(test_vol),
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
            "test_vol": test_vol.tolist(),
            "vol_thresholds": {"low": low_thresh, "high": high_thresh},
        },
    }


if __name__ == "__main__":
    evaluate()
