from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

PROCESSED_DIR = Path("../../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@njit
def _triple_barrier_loop(
    close: np.ndarray,
    high:  np.ndarray,
    low:   np.ndarray,
    atr:   np.ndarray,
    tp_mult:  float,
    sl_mult:  float,
    max_bars: int,
) -> tuple:
    n      = len(close)
    labels   = np.zeros(n, dtype=np.int8)
    hit_bars = np.zeros(n, dtype=np.int32)
    market_returns  = np.zeros(n, dtype=np.float64)
    trade_returns  = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        entry     = close[i]
        tp_price  = entry + tp_mult * atr[i]
        sl_price  = entry - sl_mult * atr[i]
        end_bar   = min(i + max_bars, n - 1)

        label   = 0
        bar_hit = max_bars
        exit_price = close[min(i + max_bars, n - 1)]

        for j in range(i + 1, end_bar + 1):
            if high[j] >= tp_price:
                label   = 1
                bar_hit = j - i
                exit_price = tp_price
                break
            if low[j] <= sl_price:
                label   = -1
                bar_hit = j - i
                exit_price = sl_price
                break

        market_ret     = (exit_price - entry) / entry
        trade_ret      = abs(market_ret)

        labels[i]         = label
        hit_bars[i]       = bar_hit
        market_returns[i] = market_ret
        trade_returns[i]  = trade_ret

    return labels, hit_bars, market_returns, trade_returns


def build_labels(
    features_path: str | Path  = PROCESSED_DIR / "features_1h.parquet",
    atr_col: str               = "atr_14",
    tp_mult: float             = 1.0,
    sl_mult: float             = 1.0,
    max_bars: int              = 48,
    min_atr_pct: float         = 0.003,
    min_trade_ret: float       = 0.005,
) -> tuple[Any, Any]:
    print(f"Loading {features_path} ...")
    df = pd.read_parquet(features_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required = ["open", "high", "low", "close", "volume", atr_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in features file: {missing}")

    close = df["close"].to_numpy(dtype=np.float64)
    high  = df["high"].to_numpy(dtype=np.float64)
    low   = df["low"].to_numpy(dtype=np.float64)
    atr   = df[atr_col].to_numpy(dtype=np.float64)

    print(f"Running triple barrier on {len(df):,} bars ...")
    print(f"  TP mult     : {tp_mult}×  |  SL mult : {sl_mult}×  |  R:R = {tp_mult/sl_mult:.1f}")
    print(f"  Max hold    : {max_bars} bars ({max_bars}h)")
    print(f"  Min ATR pct : {min_atr_pct:.1%}")

    labels, hit_bars, market_returns, trade_returns = _triple_barrier_loop(
        close, high, low, atr,
        tp_mult, sl_mult, max_bars
    )

    out = pd.DataFrame(index=df.index)
    out["label"]       = labels
    out["hit_bars"]    = hit_bars
    out["market_return"]  = market_returns
    out["trade_return"]  = trade_returns
    out["tp_price"]    = close + tp_mult * atr
    out["entry_price"] = close
    out["sl_price"]    = close - sl_mult * atr
    out["atr_used"]    = atr
    out["reward_risk"] = tp_mult / sl_mult

    atr_pct = atr / close
    dead_market = atr_pct < min_atr_pct
    out.loc[dead_market, "label"] = 0
    print(f"  Dead market bars filtered : {dead_market.sum():,}")

    small_moves = trade_returns < min_trade_ret
    out.loc[small_moves, "label"] = 0
    print(f"  Small moves (< {min_trade_ret}) filtered: {small_moves.sum():,}")

    out = out.iloc[:-max_bars]

    _print_label_stats(out)

    out_path = PROCESSED_DIR / "labels_1h.parquet"
    out.to_parquet(out_path)
    print(f"\n✓  Saved → {out_path}")

    labels_trade = out.copy()
    labels_trade["label"] = (labels_trade["label"] != 0).astype(int)
    labels_trade_path = PROCESSED_DIR / "labels_trade_1h.parquet"
    labels_trade.to_parquet(labels_trade_path)
    print(f"\nLabels trade (should we enter?): {len(labels_trade)}")
    print(
        f"  No trade (0): {(labels_trade['label'] == 0).sum():,} ({(labels_trade['label'] == 0).sum() / len(labels_trade) * 100:.1f}%)")
    print(
        f"  Trade (1): {(labels_trade['label'] == 1).sum():,} ({(labels_trade['label'] == 1).sum() / len(labels_trade) * 100:.1f}%)")
    print(f"✓  Saved → {labels_trade_path}")

    labels_direction = out.loc[out["label"] != 0].copy()
    labels_direction["label"] = (labels_direction["label"] == 1).astype(int)
    labels_direction_path = PROCESSED_DIR / "labels_direction_1h.parquet"
    labels_direction.to_parquet(labels_direction_path)
    print(f"\nLabels direction (long vs short, tradeable bars only): {len(labels_direction)}")
    print(
        f"  Short (0): {(labels_direction['label'] == 0).sum():,} ({(labels_direction['label'] == 0).sum() / len(labels_direction) * 100:.1f}%)")
    print(
        f"  Long (1): {(labels_direction['label'] == 1).sum():,} ({(labels_direction['label'] == 1).sum() / len(labels_direction) * 100:.1f}%)")
    print(f"✓  Saved → {labels_direction_path}")

    return labels_trade, labels_direction


def _print_label_stats(labels: pd.DataFrame) -> None:
    total  = len(labels)
    counts = labels["label"].value_counts().sort_index()
    pcts   = counts / total * 100

    print(f"\n{'─'*45}")
    print(f"  LABEL DISTRIBUTION  (n={total:,})")
    print(f"{'─'*45}")
    print(f"  Short  (-1) : {counts.get(-1, 0):>7,}  ({pcts.get(-1, 0):>5.1f}%)")
    print(f"  Neutral ( 0) : {counts.get( 0, 0):>7,}  ({pcts.get( 0, 0):>5.1f}%)")
    print(f"  Long   ( 1) : {counts.get( 1, 0):>7,}  ({pcts.get( 1, 0):>5.1f}%)")
    print(f"{'─'*45}")

    print(f"\n  AVG BARS TO EXIT")
    for lbl, name in [(-1, "Short"), (0, "Neutral"), (1, "Long")]:
        mask = labels["label"] == lbl
        if mask.sum() > 0:
            avg = labels.loc[mask, "hit_bars"].mean()
            print(f"  {name:<10} : {avg:.1f} bars")

    print(f"\n  AVG TRADE RETURN AT EXIT")
    for lbl, name in [(-1, "Short"), (0, "Neutral"), (1, "Long")]:
        mask = labels["label"] == lbl
        if mask.sum() > 0:
            avg = labels.loc[mask, "trade_return"].mean() * 100
            print(f"  {name:<10} : {avg:+.3f}%")


if __name__ == "__main__":
    build_labels(
        features_path = PROCESSED_DIR / "features_1h.parquet",
        atr_col       = "atr_14",
        tp_mult       = 1.5,
        sl_mult       = 1.5,
        max_bars      = 36,
        min_atr_pct   = 0.005,
        min_trade_ret = 0.005,
    )
