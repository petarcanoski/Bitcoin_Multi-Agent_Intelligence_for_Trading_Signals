
import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

_SCRIPT_DIR   = Path(__file__).resolve().parent
RAW_DIR       = _SCRIPT_DIR.parents[1] / "data" / "raw"
PROCESSED_DIR = _SCRIPT_DIR.parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def rolling_zscore(series: pd.Series, window: int = 200) -> pd.Series:
    mu  = series.rolling(window, min_periods=window // 4).mean()
    std = series.rolling(window, min_periods=window // 4).std()
    z   = (series - mu) / std.replace(0, np.nan)
    return z.fillna(0).clip(-4, 4)


def add_price_action(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]

    for p in [1, 3, 6, 12, 24, 48]:
        df[f"log_ret_{p}"] = np.log(c / c.shift(p))

    body  = c - o
    rng   = (h - l).replace(0, np.nan)
    df["body_pct"]       = (body.abs() / rng).fillna(0)
    df["upper_wick_pct"] = ((h - c.combine(o, max)) / rng).fillna(0)
    df["lower_wick_pct"] = ((c.combine(o, min) - l) / rng).fillna(0)
    df["candle_dir"]     = np.sign(body)

    df["oc_return"] = ((c - o) / o.replace(0, np.nan)).fillna(0)

    df["hl_range_pct"] = ((h - l) / c.replace(0, np.nan)).fillna(0)

    log_ret_1bar = np.log(c / c.shift(1)).fillna(0)
    for w in [24, 168]:
        df[f"cum_ret_{w}"] = np.expm1(log_ret_1bar.rolling(w, min_periods=w // 4).sum())

    for w in [24, 48, 168]:
        hi  = h.rolling(w).max()
        lo  = l.rolling(w).min()
        rng_w = (hi - lo).replace(0, np.nan)
        df[f"dist_high_{w}"] = ((hi - c) / rng_w).fillna(0)
        df[f"dist_low_{w}"]  = ((c - lo)  / rng_w).fillna(0)

    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]

    for p in [7, 14, 21]:
        df[f"rsi_{p}"] = ta.rsi(c, length=p) / 100.0

    df["rsi_divergence"] = df["rsi_14"].diff(3) - c.pct_change(3).clip(-1, 1)

    for fast, slow, sig in [(12, 26, 9), (5, 13, 4)]:
        m = ta.macd(c, fast=fast, slow=slow, signal=sig)
        tag = f"{fast}_{slow}"
        df[f"macd_line_{tag}"]   = m[f"MACD_{fast}_{slow}_{sig}"]
        df[f"macd_signal_{tag}"] = m[f"MACDs_{fast}_{slow}_{sig}"]
        df[f"macd_hist_{tag}"]   = m[f"MACDh_{fast}_{slow}_{sig}"]
        df[f"macd_cross_{tag}"]  = np.sign(df[f"macd_hist_{tag}"])

    for k_p, d_p in [(14, 3), (21, 3)]:
        s = ta.stoch(h, l, c, k=k_p, d=d_p)
        df[f"stoch_k_{k_p}"] = s[f"STOCHk_{k_p}_{d_p}_3"] / 100.0
        df[f"stoch_d_{k_p}"] = s[f"STOCHd_{k_p}_{d_p}_3"] / 100.0
    df["stoch_cross"] = np.sign(df["stoch_k_14"] - df["stoch_d_14"])

    for p in [6, 12, 24]:
        df[f"roc_{p}"] = ta.roc(c, length=p) / 100.0

    for p in [14, 21]:
        df[f"williams_r_{p}"] = ta.willr(h, l, c, length=p) / 100.0

    for p in [9, 21, 50, 100, 200]:
        ema = ta.ema(c, length=p)
        df[f"price_vs_ema_{p}"] = ((c - ema) / ema.replace(0, np.nan)).fillna(0)

    df["ema_9_21_cross"]   = np.sign(ta.ema(c, length=9)  - ta.ema(c, length=21))
    df["ema_21_50_cross"]  = np.sign(ta.ema(c, length=21) - ta.ema(c, length=50))
    df["ema_50_200_cross"] = np.sign(ta.ema(c, length=50) - ta.ema(c, length=200))

    for p in [6, 12, 24]:
        df[f"momentum_{p}"] = c.diff(p)

    for p in [10, 20, 50, 200]:
        sma = ta.sma(c, length=p)
        df[f"price_vs_sma_{p}"] = ((c - sma) / sma.replace(0, np.nan)).fillna(0)

    df["sma_20_50_cross"]  = np.sign(ta.sma(c, length=20) - ta.sma(c, length=50))
    df["sma_50_200_cross"] = np.sign(ta.sma(c, length=50) - ta.sma(c, length=200))

    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]

    for p in [7, 14, 21]:
        atr = ta.atr(h, l, c, length=p)
        df[f"atr_{p}"]     = atr
        df[f"atr_{p}_pct"] = (atr / c.replace(0, np.nan)).fillna(0)

    df["atr_ratio_7_21"] = (df["atr_7"] / df["atr_21"].replace(0, np.nan)).fillna(1)

    for std_mult in [2.0, 1.0]:
        bb  = ta.bbands(c, length=20, std=std_mult)

        lower = bb.iloc[:, 0]
        mid   = bb.iloc[:, 1]
        upper = bb.iloc[:, 2]

        tag   = f"20_{int(std_mult)}"
        width = upper - lower
        df[f"bb_width_{tag}"]    = (width / mid.replace(0, np.nan)).fillna(0)
        df[f"bb_position_{tag}"] = ((c - lower) / width.replace(0, np.nan)).fillna(0.5)
        df[f"bb_squeeze_{tag}"]  = (
            df[f"bb_width_{tag}"] < df[f"bb_width_{tag}"].rolling(50).quantile(0.2)
        ).astype(float)

    log_ret = np.log(c / c.shift(1))
    for w in [12, 24, 72, 168]:
        df[f"hist_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(w)

    df["vol_ratio_24_168"] = (
        df["hist_vol_24"] / df["hist_vol_168"].replace(0, np.nan)
    ).fillna(1)

    return df


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["obv"]           = ta.obv(c, v)
    df["obv_slope_12"]  = df["obv"].diff(12)
    df["obv_slope_24"]  = df["obv"].diff(24)
    df["obv_divergence"] = np.sign(df["obv"].diff(12)) - np.sign(c.diff(12))

    for w in [12, 24, 48]:
        avg = v.rolling(w).mean().replace(0, np.nan)
        df[f"vol_ratio_{w}"] = (v / avg).fillna(1)

    tp = (h + l + c) / 3
    for w in [24, 48]:
        vwap = (tp * v).rolling(w).sum() / v.rolling(w).sum().replace(0, np.nan)
        df[f"price_vs_vwap_{w}"] = ((c - vwap) / vwap.replace(0, np.nan)).fillna(0)

    for p in [14, 21]:
        df[f"cmf_{p}"] = ta.cmf(h, l, c, v, length=p)

    df["mfi_14"] = ta.mfi(h, l, c, v, length=14) / 100.0

    df["vol_change"] = v.pct_change().clip(-5, 5).fillna(0)

    pvt = (v * c.pct_change().fillna(0)).cumsum()
    df["pvt"]          = pvt
    df["pvt_slope_12"] = pvt.diff(12)

    vol_avg = v.rolling(24).mean().replace(0, np.nan)
    for lag in [1, 5]:
        df[f"vol_lag_{lag}_ratio"] = (v.shift(lag) / vol_avg).fillna(1)

    return df


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)

    for w in [24, 72, 168]:
        df[f"skew_{w}"] = log_ret.rolling(w, min_periods=w // 4).skew().fillna(0)
        df[f"kurt_{w}"] = log_ret.rolling(w, min_periods=w // 4).kurt().fillna(0)

    df["ret_x_rvol"] = (df["log_ret_1"] * df["vol_ratio_24"]).fillna(0)

    df["vol_x_mom"] = (df["hist_vol_24"] * df["roc_12"]).fillna(0)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    df["hour_sin"]  = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    return df


def build_htf_context(htf_path: Path, prefix: str) -> pd.DataFrame:
    raw = pd.read_parquet(htf_path)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    raw.columns = raw.columns.str.lower()

    c, h, l, v = raw["close"], raw["high"], raw["low"], raw["volume"]
    out = pd.DataFrame(index=raw.index)

    for p in [9, 21, 50]:
        ema = ta.ema(c, length=p)
        out[f"{prefix}_price_vs_ema_{p}"] = ((c - ema) / ema.replace(0, np.nan)).fillna(0)

    out[f"{prefix}_ema_9_21_cross"]  = np.sign(ta.ema(c, 9)  - ta.ema(c, 21))
    out[f"{prefix}_ema_21_50_cross"] = np.sign(ta.ema(c, 21) - ta.ema(c, 50))

    out[f"{prefix}_rsi_14"] = ta.rsi(c, length=14) / 100.0

    atr14 = ta.atr(h, l, c, length=14)
    atr50 = ta.atr(h, l, c, length=50)
    out[f"{prefix}_atr_ratio"]    = (atr14 / atr50.replace(0, np.nan)).fillna(1)
    out[f"{prefix}_hist_vol"]     = np.log(c / c.shift(1)).rolling(24).std() * np.sqrt(24)

    out[f"{prefix}_vol_ratio"]    = (v / v.rolling(24).mean().replace(0, np.nan)).fillna(1)
    out[f"{prefix}_candle_dir"]   = np.sign(c - raw["open"])

    return out


_SKIP_PATTERNS = ["_cross", "_dir", "divergence", "rsi_", "atr_"]

_EXCLUDE = {
    "open", "high", "low", "close", "volume",
    "label", "hit_bars", "log_return",
    "tp_price", "sl_price", "atr_used", "reward_risk",
}

_SKIP_PREFIXES = ("bb_upper_", "bb_lower_", "vwap_")

_SKIP_NORM = {
    "stoch_k_14", "stoch_d_14", "stoch_k_21", "stoch_d_21",
    "williams_r_14", "williams_r_21", "mfi_14",
    "bb_position_20_2", "bb_position_20_1",
    "bb_squeeze_20_2", "bb_squeeze_20_1",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
}

def apply_rolling_normalization(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    cols = [
        col for col in df.columns
        if col not in _EXCLUDE
           and col not in _SKIP_NORM
           and not any(col.startswith(p) for p in _SKIP_PREFIXES)
           and not any(pattern in col for pattern in _SKIP_PATTERNS)
           and df[col].dtype in [np.float32, np.float64, float]
    ]

    print(f"  Normalizing {len(cols)} columns (window={window})...")
    print(f"  Normalizing: {', '.join(cols)}")
    print(f"  Skiping: {', '.join(sorted(set(df.columns) - set(cols)))}")

    for col in cols:
        df[col] = rolling_zscore(df[col], window)

    return df


def build_features(
    primary_tf: str = "1h",
    htf_list: list | None = None,
    norm_window: int = 200,
    drop_raw_ohlcv: bool = False,
) -> pd.DataFrame:
    if htf_list is None:
        htf_list = [("4h", "htf4h"), ("1D", "htf1d")]

    print(f"[1/6] Loading btc_{primary_tf}.parquet ...")
    df = pd.read_parquet(RAW_DIR / f"btc_{primary_tf}.parquet")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = df.columns.str.lower()
    print(f"      {len(df):,} rows  |  {df.index[0]}  →  {df.index[-1]}")

    print("[2/6] Price action ...")
    df = add_price_action(df)

    print("[3/6] Momentum ...")
    df = add_momentum(df)

    print("[4/6] Volatility + volume + time + stats ...")
    df = add_volatility(df)
    df = add_volume(df)
    df = add_statistical_features(df)
    df = add_time_features(df)

    print("[5/6] Multi-timeframe context ...")
    for htf, prefix in htf_list:
        path = RAW_DIR / f"btc_{htf}.parquet"
        if not path.exists():
            print(f"      SKIP: {path} not found.")
            continue
        ctx = build_htf_context(path, prefix)
        ctx = ctx.reindex(df.index, method="ffill")
        df  = df.join(ctx, how="left")
        print(f"      {htf}: +{len(ctx.columns)} columns")

    print("[6/6] Rolling normalization ...")
    df = apply_rolling_normalization(df, window=norm_window)

    if drop_raw_ohlcv:
        df.drop(columns=["open", "high", "low", "close", "volume"], errors="ignore", inplace=True)

    n_before = len(df)
    df = df.iloc[norm_window + 10:]
    print(f"\n      Warmup dropped : {n_before - len(df):,} rows")
    print(f"      Final shape    : {df.shape}")

    out_path = PROCESSED_DIR / f"features_{primary_tf}.parquet"
    df.to_parquet(out_path)
    print(f"\n✓  Saved → {out_path}")
    return df


def inspect_features(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print(f"Shape      : {df.shape}")
    print(f"Date range : {df.index[0]}  →  {df.index[-1]}")
    print(f"Total NaNs : {df.isna().sum().sum():,}")

    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols):
        print(f"\nColumns with NaNs ({len(nan_cols)}):")
        for col, n in nan_cols.items():
            print(f"  {col:<45} {n:>6}")
    else:
        print("No NaN columns ✓")

    print("\nFeature groups:")
    groups: dict[str, int] = {}
    for col in df.columns:
        k = col.split("_")[0]
        groups[k] = groups.get(k, 0) + 1
    for k, n in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {k:<22} {n:>4} columns")

    print(f"Total features: {len(df.columns)}")


if __name__ == "__main__":
    df = build_features(
        primary_tf     = "1h",
        htf_list       = [("4h", "htf4h"), ("1D", "htf1d")],
        norm_window    = 200,
        drop_raw_ohlcv = False,
    )
    inspect_features(df)