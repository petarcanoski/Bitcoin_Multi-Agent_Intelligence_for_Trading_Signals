import sys, json, io, contextlib
import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import time as _t
from pathlib import Path

PRED_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PRED_ROOT.parent
RAW_DIR   = PRED_ROOT / "data" / "raw"
PROC_DIR  = PRED_ROOT / "data" / "processed"
MODEL_DIR = PRED_ROOT / "src" / "models" / "final"
IMP_PATH  = REPO_ROOT / "project-context" / "feature_importance.json"

sys.path.insert(0, str(PRED_ROOT / "src" / "pipeline"))
sys.path.insert(0, str(PRED_ROOT / "src" / "models_def"))

from features import (                                        # noqa: E402
    add_price_action, add_momentum, add_volatility, add_volume,
    add_statistical_features, add_time_features, apply_rolling_normalization,
)
from cnn_lstm import CNNLSTMModel                             # noqa: E402

CONTEXT_BARS = 260
THRESHOLD    = 0.45
FEE          = 0.001
TP_MULT      = 1.0
SL_MULT      = 1.0
MAX_BARS     = 48

_EXCL = {"open","high","low","close","volume","label","hit_bars",
         "market_return","trade_return","tp_price","entry_price",
         "sl_price","atr_used","reward_risk"}


def _device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


_TF_FILE = {"1d": "1D", "4h": "4h", "1h": "1h", "15m": "15m"}


def fetch_and_update(tf: str) -> pd.DataFrame:
    import ccxt
    file_suffix = _TF_FILE.get(tf, tf)
    raw_path = RAW_DIR / f"btc_{file_suffix}.parquet"
    existing = pd.read_parquet(raw_path)
    existing.index = pd.to_datetime(existing.index).tz_localize(None)
    existing = existing.sort_index()
    last_ts  = existing.index[-1]

    since_ms = int((last_ts.timestamp() + 1) * 1000)
    binance  = ccxt.binance()
    rows, since = [], since_ms

    while True:
        chunk = binance.fetch_ohlcv("BTC/USDT", timeframe=tf, since=since, limit=1000)
        if not chunk: break
        rows += chunk
        since = chunk[-1][0] + 1
        _t.sleep(0.3)

    if not rows:
        print(f"  {tf}: already up to date (last: {last_ts})")
        return existing

    new_df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    new_df["ts"] = pd.to_datetime(new_df["ts"], unit="ms").dt.tz_localize(None)
    new_df = new_df.set_index("ts")

    combined = pd.concat([existing.iloc[:-1], new_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(raw_path)
    print(f"  {tf}: +{len(new_df)} new bars | total {len(combined):,} | last: {combined.index[-1]}")
    return combined


def build_new_features(df_1h, df_4h, df_1d, after: pd.Timestamp) -> pd.DataFrame:
    df = df_1h.copy(); df.columns = df.columns.str.lower(); df = df.sort_index()
    cutoff = df.index.searchsorted(after)
    df     = df.iloc[max(0, cutoff - CONTEXT_BARS):].copy()

    df = add_price_action(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_volume(df)
    df = add_statistical_features(df)
    df = add_time_features(df)

    for htf_raw, prefix in [(df_4h, "htf4h"), (df_1d, "htf1d")]:
        htf = htf_raw.copy(); htf.columns = htf.columns.str.lower(); htf = htf.sort_index()
        htf = htf[htf.index >= df.index[0] - pd.Timedelta(days=60)]
        c, h, l, v, o = htf["close"], htf["high"], htf["low"], htf["volume"], htf["open"]
        ctx = pd.DataFrame(index=htf.index)
        for p in [9, 21, 50]:
            ema = ta.ema(c, length=p)
            ctx[f"{prefix}_price_vs_ema_{p}"] = ((c - ema) / ema.replace(0, np.nan)).fillna(0)
        ctx[f"{prefix}_ema_9_21_cross"]  = np.sign(ta.ema(c, 9)  - ta.ema(c, 21))
        ctx[f"{prefix}_ema_21_50_cross"] = np.sign(ta.ema(c, 21) - ta.ema(c, 50))
        ctx[f"{prefix}_rsi_14"]          = ta.rsi(c, length=14) / 100.0
        a14 = ta.atr(h, l, c, length=14); a50 = ta.atr(h, l, c, length=50)
        ctx[f"{prefix}_atr_ratio"] = (a14 / a50.replace(0, np.nan)).fillna(1)
        ctx[f"{prefix}_hist_vol"]  = np.log(c / c.shift(1)).rolling(24).std() * np.sqrt(24)
        ctx[f"{prefix}_vol_ratio"] = (v / v.rolling(24).mean().replace(0, np.nan)).fillna(1)
        ctx[f"{prefix}_candle_dir"] = np.sign(c - o)
        df = df.join(ctx.reindex(df.index, method="ffill"), how="left")

    with contextlib.redirect_stdout(io.StringIO()):
        df = apply_rolling_normalization(df, window=200)

    return df[df.index > after].copy()


def generate_labels(feats: pd.DataFrame):
    close = feats["close"].to_numpy(np.float64)
    high  = feats["high"].to_numpy(np.float64)
    low   = feats["low"].to_numpy(np.float64)
    atr   = feats["atr_14"].to_numpy(np.float64)
    n     = len(feats)

    labels = np.zeros(n, np.int8)
    mret   = np.zeros(n, np.float64)

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        entry  = close[i]
        tp_p   = entry + TP_MULT * atr[i]
        sl_p   = entry - SL_MULT * atr[i]
        end    = min(i + MAX_BARS, n - 1)
        lbl    = 0
        exit_p = close[end]
        for j in range(i + 1, end + 1):
            if high[j] >= tp_p: lbl = 1;  exit_p = tp_p;  break
            if low[j]  <= sl_p: lbl = -1; exit_p = sl_p; break
        labels[i] = lbl
        mret[i]   = (exit_p - entry) / entry

    valid = n - MAX_BARS
    return labels[:valid], mret[:valid], feats.index[:valid]


def load_model(device):
    def _load(path, use_reg):
        ck  = torch.load(path, map_location=device, weights_only=False)
        sd  = ck["model_state_dict"]
        m   = CNNLSTMModel(
            input_features     = sd["cnn.0.weight"].shape[1],
            num_classes        = sd["fc_classifier.3.weight"].shape[0],
            use_regression_head= use_reg,
        )
        m.load_state_dict(sd); m.eval()
        return m.to(device), int(ck.get("config", {}).get("sequence_length", 30))
    trade_m, seq_len = _load(MODEL_DIR / "trade"     / "best_model.pt", False)
    dir_m,   _       = _load(MODEL_DIR / "direction" / "best_model.pt", True)
    return trade_m, dir_m, seq_len


@torch.no_grad()
def infer(model, feat: np.ndarray, seq_len: int, device, batch=512) -> np.ndarray:
    out = []
    for i in range(seq_len, len(feat), batch):
        end  = min(i + batch, len(feat))
        seqs = np.stack([feat[j - seq_len:j] for j in range(i, end)])
        x    = torch.tensor(seqs, dtype=torch.float32).to(device)
        out.append(torch.softmax(model(x)[0], dim=-1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 2))


def simulate(tp, dp, mr, labels):
    signals = np.where(tp >= THRESHOLD, np.where(dp >= 0.5, 1, -1), 0)
    mask    = signals != 0
    n       = int(mask.sum())
    if n == 0:
        return None

    pnl     = signals[mask] * mr[mask] - FEE
    eq      = np.zeros(len(signals)); eq[mask] = pnl
    ceq     = np.cumsum(eq)
    run_max = np.maximum.accumulate(ceq)

    labeled = labels[mask] != 0
    dir_acc = None
    if labeled.sum() > 0:
        dir_acc = float((signals[mask][labeled] == labels[mask][labeled]).mean())

    total = len(signals)
    return {
        "n_total":  total,
        "n_trades": n,
        "n_buy":    int((signals == 1).sum()),
        "n_sell":   int((signals == -1).sum()),
        "n_hold":   int((signals == 0).sum()),
        "win_rate": float((pnl > 0).mean()),
        "cum_pnl":  float(pnl.sum() * 100),
        "avg_pnl":  float(pnl.mean() * 100),
        "sharpe":   float(eq.mean() / (eq.std() + 1e-9) * np.sqrt(8760)),
        "max_dd":   float((ceq - run_max).min() * 100),
        "bh":       float(mr.sum() * 100),
        "dir_acc":  dir_acc,
        "equity":   ceq * 100,
    }


def _equity_ascii(eq, width=56, height=8):
    idx  = np.linspace(0, len(eq)-1, width).astype(int)
    eq_s = eq[idx]
    ymin, ymax = eq_s.min(), eq_s.max()
    yr   = ymax - ymin or 1.0
    lines = []
    for r in range(height-1, -1, -1):
        row = "".join(
            "█" if int((eq_s[c] - ymin) / yr * (height-1)) == r
            else ("▓" if int((eq_s[c] - ymin) / yr * (height-1)) > r else "·")
            for c in range(width)
        )
        lines.append(f"  {(r/(height-1))*yr+ymin:>+7.1f}%|{row}")
    lines.append(f"  {'':>8} " + "─"*width)
    return "\n".join(lines)


def print_report(m, start, end, n_bars):
    sep = "=" * 62
    print(f"\n{sep}")
    print("  LIVE EVALUATION REPORT")
    print(f"  Period     : {start}  →  {end}")
    print(f"  New bars   : {n_bars:,}  (~{n_bars//24} days at 1h resolution)")
    print(f"  Model      : final/  (CNN-LSTM, 81 feat, seq=30, th={THRESHOLD})")
    print(f"  Fee        : {FEE*100:.1f}% per side (0.2% round-trip)")
    print(sep)

    if m is None:
        print("  No trades generated.\n"); return

    total = m["n_total"]
    print(f"\n  SIGNALS")
    print(f"    Buy      : {m['n_buy']:>7,}  ({m['n_buy']/total:.1%} of bars)")
    print(f"    Sell     : {m['n_sell']:>7,}  ({m['n_sell']/total:.1%} of bars)")
    print(f"    Hold     : {m['n_hold']:>7,}  ({m['n_hold']/total:.1%} of bars)")
    print(f"    Trades   : {m['n_trades']:>7,}")

    print(f"\n  PERFORMANCE")
    print(f"    Cum P&L  : {m['cum_pnl']:>+8.2f}%")
    print(f"    Buy&Hold : {m['bh']:>+8.2f}%")
    print(f"    Alpha    : {m['cum_pnl']-m['bh']:>+8.2f}%")
    print(f"    Avg trade: {m['avg_pnl']:>+8.4f}%")

    print(f"\n  QUALITY")
    print(f"    Win rate : {m['win_rate']:>8.2%}")
    print(f"    Sharpe   : {m['sharpe']:>8.3f}  (annualized, hourly bars)")
    print(f"    Max DD   : {m['max_dd']:>8.2f}%")
    if m["dir_acc"] is not None:
        print(f"    Dir acc  : {m['dir_acc']:>8.2%}  (vs triple-barrier labels)")

    if len(m["equity"]) > 10:
        print(f"\n  EQUITY CURVE  (cumulative return %)\n")
        print(_equity_ascii(np.array(m["equity"])))

    print(f"\n  VERDICT")
    alpha = m["cum_pnl"] - m["bh"]
    if m["cum_pnl"] > 0 and alpha > 0:
        print(f"    ✓ Model outperforms buy-and-hold by {alpha:+.2f}%")
    elif m["cum_pnl"] > 0:
        print(f"    ~ Model is profitable but underperforms buy-and-hold")
    else:
        print(f"    ✗ Model lost money on this period — consider retraining")

    print(f"\n{sep}\n")


def main():
    device = _device()
    print(f"\n{'='*62}")
    print(f"LIVE EVALUATION PIPELINE  [device={device}]")
    print(f"{'='*62}")

    pos_set    = {f for f, v in json.loads(IMP_PATH.read_text()) if v >= 0}
    ref_pq     = pd.read_parquet(PROC_DIR / "features_1h.parquet")
    ref_cols   = ref_pq.columns
    orig_last  = pd.to_datetime(ref_pq.index[-1])
    pos_feats  = [c for c in ref_cols if c not in _EXCL and c in pos_set]
    del ref_pq
    print(f"\n  Positive features : {len(pos_feats)}")
    print(f"  Training cutoff   : {orig_last}  (evaluating bars after this)")

    print("\n[1/5] Fetching new BTC data from Binance...")
    df_1h = fetch_and_update("1h")
    df_4h = fetch_and_update("4h")
    df_1d = fetch_and_update("1d")

    print(f"\n[2/5] Building features for bars after {orig_last}...")
    feats = build_new_features(df_1h, df_4h, df_1d, after=orig_last)
    print(f"  New feature bars: {len(feats):,}  ({feats.index[0]}  →  {feats.index[-1]})")

    if len(feats) < MAX_BARS + 30 + 5:
        print("  Not enough new data to evaluate. Try again later."); return

    print("\n[3/5] Generating triple-barrier labels...")
    labels, mret, valid_idx = generate_labels(feats)
    n_labeled = len(labels)
    print(f"  Labeled bars: {n_labeled:,}  (last {MAX_BARS} bars dropped — no forward data)")
    print(f"  Long: {(labels==1).mean():.1%}  Short: {(labels==-1).mean():.1%}  "
          f"No-trade: {(labels==0).mean():.1%}")

    print("\n[4/5] Running model inference...")
    trade_m, dir_m, seq_len = load_model(device)
    print(f"  seq_len={seq_len}  |  features={len(pos_feats)}")

    feat_arr = feats[pos_feats].fillna(0).to_numpy(dtype="float32")
    M        = len(feats)

    n_aligned = M - MAX_BARS - seq_len
    if n_aligned <= 0:
        print("  Not enough bars for aligned inference."); return

    tp_all = infer(trade_m, feat_arr, seq_len, device)[:, 1]
    dp_all = infer(dir_m,   feat_arr, seq_len, device)[:, 1]

    tp = tp_all[:n_aligned]
    dp = dp_all[:n_aligned]
    mr = mret[seq_len: seq_len + n_aligned]
    lb = labels[seq_len: seq_len + n_aligned]
    print(f"  Aligned inference bars: {len(tp):,}")

    print("\n[5/5] Simulating trading...")
    m = simulate(tp, dp, mr, lb)
    start = str(valid_idx[seq_len])[:16]
    end   = str(valid_idx[min(seq_len + n_aligned - 1, len(valid_idx)-1)])[:16]
    print_report(m, start, end, len(tp))


if __name__ == "__main__":
    main()
