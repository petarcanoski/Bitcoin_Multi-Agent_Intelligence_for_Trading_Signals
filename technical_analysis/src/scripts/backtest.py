
import json
import sys
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from pathlib import Path

TA_ROOT    = Path(__file__).resolve().parents[2]
REPO_ROOT  = TA_ROOT.parent
DATA_DIR = TA_ROOT / "data" / "processed"

TRAIN_PCT, VAL_PCT = 0.70, 0.15
TRADE_THRESHOLD = 0.45
FEE = 0.001
AGENTIC_BUY_THRESH = 0.20
AGENTIC_SELL_THRESH = -0.20

W_TA = 0.65
W_SENTIMENT = 0.25
W_RISK = 0.10


def _fetch_fear_greed(limit: int = 500) -> pd.Series:
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]
        fg = {
            datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc).date(): int(d["value"])
            for d in data
        }
        series = pd.Series(fg, name="fear_greed").sort_index()
        print(f"  Fear & Greed: {len(series)} daily values "
              f"({series.index[0]} → {series.index[-1]})")
        return series
    except Exception as e:
        print(f"  Fear & Greed API unavailable ({e}). Using fallback proxy.")
        return pd.Series(dtype=float)


def _fg_to_sentiment(fg_value: float) -> float:
    return (fg_value - 50.0) / 50.0


def _load_test_data():
    feats = pd.read_parquet(DATA_DIR / "features_1h.parquet")
    feats.index = pd.to_datetime(feats.index)
    feats = feats.sort_index()

    orig = pd.read_parquet(DATA_DIR / "labels_1h.parquet")
    orig.index = pd.to_datetime(orig.index)

    common = feats.index.intersection(orig.index)
    feats = feats.loc[common]
    orig = orig.loc[common]

    n = len(feats)
    test_start = int(n * (TRAIN_PCT + VAL_PCT))
    train_end = int(n * TRAIN_PCT)
    return feats.iloc[test_start:], orig.iloc[test_start:], feats.iloc[:train_end]


def _ta_scores(trade_p: np.ndarray, dir_p: np.ndarray) -> np.ndarray:
    direction_sign = np.where(dir_p >= 0.5, 1.0, -1.0)
    score = np.where(
        trade_p >= TRADE_THRESHOLD,
        trade_p * direction_sign,
        0.0,
    )
    return score.clip(-1.0, 1.0)


def _risk_scores(test_feats: pd.DataFrame, train_feats: pd.DataFrame) -> np.ndarray:
    vol_col = "atr_14_pct" if "atr_14_pct" in test_feats.columns else "hist_vol_24"
    train_vol = train_feats[vol_col].values
    test_vol = test_feats[vol_col].values
    low_t = float(np.nanpercentile(train_vol, 30))
    high_t = float(np.nanpercentile(train_vol, 70))
    risk = np.clip((test_vol - low_t) / (high_t - low_t + 1e-9), 0.0, 1.0)
    return 1.0 - 2.0 * risk


def _sentiment_scores_from_fg(fg_series: pd.Series, test_ts: pd.DatetimeIndex) -> np.ndarray:
    if fg_series.empty:
        return None

    fg_hourly = pd.Series(index=test_ts, dtype=float)
    for ts in test_ts:
        d = ts.date()
        if d in fg_series.index:
            fg_hourly[ts] = _fg_to_sentiment(float(fg_series[d]))

    if fg_hourly.isna().mean() > 0.5:
        print("  Insufficient Fear & Greed coverage. Using fallback.")
        return None

    fg_hourly = fg_hourly.ffill().fillna(0.0)
    return fg_hourly.values


def _sentiment_fallback(test_feats: pd.DataFrame, train_feats: pd.DataFrame) -> np.ndarray:
    col = "rsi_14"
    if col not in test_feats.columns:
        return np.zeros(len(test_feats))
    rsi = test_feats[col].fillna(0.5).values
    return (rsi - 0.5) * 2.0


def _run_strategy(signals: np.ndarray, market_ret: np.ndarray, label: str) -> dict:
    mask = signals != 0
    raw = signals[mask] * market_ret[mask]
    pnl = raw - FEE

    n = int(mask.sum())
    if n == 0:
        print(f"  {label}: 0 trades")
        return {"n_trades": 0, "label": label}

    wins = int((pnl > 0).sum())
    cum = float(pnl.sum())
    avg = float(pnl.mean())

    eq = np.zeros(len(signals))
    eq[mask] = pnl
    cum_eq = np.cumsum(eq)
    run_max = np.maximum.accumulate(cum_eq)
    max_dd = float((cum_eq - run_max).min())
    sharpe = float(eq.mean() / (eq.std() + 1e-9) * np.sqrt(8760))

    print(f"  {label}:")
    print(f"    Trades      : {n:,}  (buy={int((signals==1).sum()):,}  sell={int((signals==-1).sum()):,})")
    print(f"    Win rate    : {wins/n:.2%}")
    print(f"    Cum P&L     : {cum*100:+.2f}%")
    print(f"    Avg trade   : {avg*100:+.4f}%")
    print(f"    Sharpe      : {sharpe:.3f}")
    print(f"    Max DD      : {max_dd*100:.2f}%")

    return {
        "label": label,
        "n_trades": n,
        "n_wins": wins,
        "n_losses": int((pnl <= 0).sum()),
        "win_rate": wins / n,
        "cumulative_pnl_pct": cum * 100,
        "avg_trade_pnl_pct": avg * 100,
        "sharpe_annualized": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "signal_counts": {
            "buy": int((signals == 1).sum()),
            "sell": int((signals == -1).sum()),
            "hold": int((signals == 0).sum()),
        },
        "equity_curve": cum_eq.tolist(),
    }


def _equity_ascii(ta_curve: list, ag_curve: list, width: int = 60, height: int = 15) -> str:
    ta = np.array(ta_curve)
    ag = np.array(ag_curve)
    all_vals = np.concatenate([ta, ag])
    ymin, ymax = all_vals.min(), all_vals.max()
    yrange = ymax - ymin if ymax != ymin else 1.0

    idx = np.linspace(0, len(ta) - 1, width).astype(int)
    ta_s = ta[idx]
    ag_s = ag[idx]

    def to_row(val):
        return int((val - ymin) / yrange * (height - 1))

    lines = []
    for r in range(height - 1, -1, -1):
        row = ""
        for c in range(width):
            ta_r = to_row(ta_s[c])
            ag_r = to_row(ag_s[c])
            if ta_r == r and ag_r == r:
                row += "X"
            elif ta_r == r:
                row += "T"
            elif ag_r == r:
                row += "A"
            else:
                row += "."
        y_label = f"{(r / (height-1)) * yrange + ymin:+.2%}|"
        lines.append(f"{y_label:>10}{row}")

    lines.append(f"{'':>10}" + "-" * width)
    lines.append(f"  T=TA-only  A=Agentic  X=overlap")
    return "\n".join(lines)


def compare(tech_results: dict = None, risk_results: dict = None) -> dict:
    print("=" * 60)
    print("BACKTEST COMPARISON: TA-ONLY vs AGENTIC")
    print("=" * 60)

    test_feats, test_orig, train_feats = _load_test_data()
    market_ret = test_orig["market_return"].values
    test_ts = test_feats.index

    print(f"\nTest period  : {test_ts[0].date()} → {test_ts[-1].date()}")
    print(f"Test bars    : {len(test_ts):,}")
    bh = float(market_ret.sum()) * 100
    print(f"Buy & Hold   : {bh:+.2f}%")

    if tech_results and "_arrays" in tech_results:
        arr = tech_results["_arrays"]
        tech_ts = pd.to_datetime(arr["valid_ts"])
        overlap = test_ts.intersection(tech_ts)
        ta_p = np.array(arr["trade_probs"])
        dp = np.array(arr["dir_probs"])
        tech_mask = tech_ts.isin(overlap)
        test_mask = test_ts.isin(overlap)
        trade_p = ta_p[tech_mask]
        dir_p = dp[tech_mask]
        mr_aligned = market_ret[test_mask]
        ts_aligned = test_ts[test_mask]
    else:
        print("  Recomputing technical inference...")
        from evaluate_technical import evaluate as eval_tech
        tr = eval_tech()
        arr = tr["_arrays"]
        tech_ts = pd.to_datetime(arr["valid_ts"])
        trade_p = np.array(arr["trade_probs"])
        dir_p = np.array(arr["dir_probs"])
        mr_aligned = np.array(arr["market_returns"])
        ts_aligned = tech_ts

    n = len(trade_p)
    print(f"Aligned bars : {n:,}")

    ta_signals = np.where(trade_p < TRADE_THRESHOLD, 0,
                          np.where(dir_p >= 0.5, 1, -1)).astype(int)

    print("\nFetching Fear & Greed index...")
    fg_series = _fetch_fear_greed(limit=600)
    sentiment = _sentiment_scores_from_fg(fg_series, ts_aligned)
    if sentiment is None:
        print("  Using RSI-14 as sentiment fallback.")
        aligned_feats = test_feats.reindex(ts_aligned)
        sentiment = _sentiment_fallback(aligned_feats, train_feats)
        sentiment_source = "RSI-14 proxy"
    else:
        sentiment_source = "Bitcoin Fear & Greed Index (alternative.me)"

    aligned_feats = test_feats.reindex(ts_aligned)
    risk_action = _risk_scores(aligned_feats, train_feats)

    ta_score = _ta_scores(trade_p, dir_p)

    combined = W_TA * ta_score + W_SENTIMENT * sentiment + W_RISK * risk_action
    agentic_signals = np.where(combined >= AGENTIC_BUY_THRESH, 1,
                               np.where(combined <= AGENTIC_SELL_THRESH, -1, 0)).astype(int)

    print("\n[Results]")
    ta_res = _run_strategy(ta_signals, mr_aligned, "TA-Only")
    ag_res = _run_strategy(agentic_signals, mr_aligned, "Agentic")

    print(f"\n  Buy & Hold: {bh:+.2f}%  ({len(mr_aligned)} bars)")

    print("\n[Equity Curve]  T=TA-Only  A=Agentic")
    if ta_res.get("equity_curve") and ag_res.get("equity_curve"):
        print(_equity_ascii(ta_res["equity_curve"], ag_res["equity_curve"]))

    if ta_res.get("n_trades", 0) > 0 and ag_res.get("n_trades", 0) > 0:
        delta_pnl = ag_res["cumulative_pnl_pct"] - ta_res["cumulative_pnl_pct"]
        delta_sharpe = ag_res["sharpe_annualized"] - ta_res["sharpe_annualized"]
        print(f"\n[Agentic vs TA-Only delta]")
        print(f"  Cum P&L  : {delta_pnl:+.2f}%")
        print(f"  Sharpe   : {delta_sharpe:+.3f}")
        print(f"  Trades   : {ag_res['n_trades'] - ta_res['n_trades']:+d}")

    return {
        "sentiment_source": sentiment_source,
        "test_period": {
            "start": str(ts_aligned[0].date()),
            "end": str(ts_aligned[-1].date()),
            "n_bars": n,
        },
        "buy_and_hold_pct": bh,
        "ta_only": {k: v for k, v in ta_res.items() if k != "equity_curve"},
        "agentic": {k: v for k, v in ag_res.items() if k != "equity_curve"},
        "equity_curves": {
            "ta_only": ta_res.get("equity_curve", []),
            "agentic": ag_res.get("equity_curve", []),
        },
    }


if __name__ == "__main__":
    compare()
