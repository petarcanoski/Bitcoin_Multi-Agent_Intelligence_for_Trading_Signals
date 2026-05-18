
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT  = Path(__file__).resolve().parents[2]
TA_ROOT    = REPO_ROOT / "technical_analysis"
DATA_DIR   = TA_ROOT / "data" / "processed"
MODEL_DIR  = TA_ROOT / "src" / "models"
IMP_PATH   = REPO_ROOT / "project-context" / "feature_importance.json"

sys.path.insert(0, str(TA_ROOT / "src" / "models_def"))
from cnn_lstm import CNNLSTMModel  # noqa: E402

TRADE_THRESHOLD = 0.45
TRAIN_PCT, VAL_PCT = 0.70, 0.15
FEE = 0.001

_EXCL = {"open","high","low","close","volume","label","hit_bars",
         "market_return","trade_return","tp_price","entry_price",
         "sl_price","atr_used","reward_risk"}
_POS_SET = {f for f, v in json.loads(IMP_PATH.read_text()) if v >= 0}


def _load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    n_feat = int(sd["cnn.0.weight"].shape[1])
    n_cls = int(sd["fc_classifier.3.weight"].shape[0])
    seq_len = int(ckpt.get("config", {}).get("sequence_length", 60))
    use_reg = any(k.startswith("fc_regression") for k in sd)
    model = CNNLSTMModel(input_features=n_feat, num_classes=n_cls,
                         use_regression_head=use_reg)
    model.load_state_dict(sd)
    model.eval()
    return model, seq_len


def _feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in _EXCL and c in _POS_SET]


def _test_split(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[int(len(df) * (TRAIN_PCT + VAL_PCT)):]


def _align_and_test(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    common = features.index.intersection(labels.index)
    merged = features.loc[common].join(labels.loc[common], how="inner")
    return _test_split(merged)


def _infer(model, feat: np.ndarray, seq_len: int, batch: int = 512) -> np.ndarray:
    results = []
    for i in range(seq_len, len(feat), batch):
        chunk = np.stack([feat[j - seq_len:j]
                          for j in range(i, min(i + batch, len(feat)))])
        x = torch.tensor(chunk, dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(model(x)[0], dim=-1)
        results.append(probs.numpy())
    return np.vstack(results)


def _backtest(signals: np.ndarray, market_ret: np.ndarray) -> dict:
    mask = signals != 0
    raw_pnl = signals[mask] * market_ret[mask]
    trade_pnl = raw_pnl - FEE

    n = int(mask.sum())
    if n == 0:
        return {"n_trades": 0, "warning": "No trades generated"}

    wins = int((trade_pnl > 0).sum())
    cum_pnl = float(trade_pnl.sum())
    avg_pnl = float(trade_pnl.mean())

    eq = np.zeros(len(signals))
    eq[mask] = trade_pnl
    cum_eq = np.cumsum(eq)
    run_max = np.maximum.accumulate(cum_eq)
    max_dd = float((cum_eq - run_max).min())

    sharpe = float(eq.mean() / (eq.std() + 1e-9) * np.sqrt(8760))

    return {
        "n_trades": n,
        "n_wins": wins,
        "n_losses": int((trade_pnl <= 0).sum()),
        "win_rate": wins / n,
        "cumulative_pnl_pct": cum_pnl * 100,
        "avg_trade_pnl_pct": avg_pnl * 100,
        "sharpe_annualized": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "buy_and_hold_pct": float(market_ret.sum()) * 100,
        "signal_counts": {
            "buy": int((signals == 1).sum()),
            "sell": int((signals == -1).sum()),
            "hold": int((signals == 0).sum()),
        },
        "equity_curve": cum_eq.tolist(),
    }


def evaluate() -> dict:
    print("=" * 60)
    print("TECHNICAL AGENT EVALUATION")
    print("=" * 60)

    feats = pd.read_parquet(DATA_DIR / "features_1h.parquet")
    feats.index = pd.to_datetime(feats.index)
    feats = feats.sort_index()

    trade_lbl = pd.read_parquet(DATA_DIR / "labels_trade_1h.parquet")
    trade_lbl.index = pd.to_datetime(trade_lbl.index)

    dir_lbl = pd.read_parquet(DATA_DIR / "labels_direction_1h.parquet")
    dir_lbl.index = pd.to_datetime(dir_lbl.index)

    orig_lbl = pd.read_parquet(DATA_DIR / "labels_1h.parquet")
    orig_lbl.index = pd.to_datetime(orig_lbl.index)

    fcols = _feature_cols(feats)

    test_trade = _align_and_test(feats, trade_lbl)
    test_dir = _align_and_test(feats, dir_lbl)
    test_orig = _align_and_test(feats, orig_lbl)

    trade_m, seq_len = _load_checkpoint(MODEL_DIR / "final" / "trade" / "best_model.pt")
    dir_m, _ = _load_checkpoint(MODEL_DIR / "final" / "direction" / "best_model.pt")

    print(f"\nTest period  : {test_trade.index[0].date()} → {test_trade.index[-1].date()}")
    print(f"Test bars    : {len(test_trade):,}")
    print(f"Seq length   : {seq_len}  |  Features: {len(fcols)}")

    feat_arr = test_trade[fcols].values.astype("float32")
    valid_ts = test_trade.index[seq_len:]

    print("\nRunning trade model inference...")
    trade_probs = _infer(trade_m, feat_arr, seq_len)
    trade_p1 = trade_probs[:, 1]
    trade_pred = (trade_p1 >= TRADE_THRESHOLD).astype(int)
    trade_true = test_trade.loc[valid_ts, "label"].values

    print("\n[Stage 1] Trade Detection:")
    print(classification_report(trade_true, trade_pred,
                                target_names=["no-trade", "trade"], digits=4))
    cm1 = confusion_matrix(trade_true, trade_pred)
    s1_acc = float((trade_pred == trade_true).mean())
    s1_report = classification_report(trade_true, trade_pred,
                                      target_names=["no-trade", "trade"],
                                      digits=4, output_dict=True)

    print("Running direction model inference...")
    dir_probs = _infer(dir_m, feat_arr, seq_len)
    dir_p1 = dir_probs[:, 1]

    dir_idx_set = set(test_dir.index)
    dir_eval = [i for i, ts in enumerate(valid_ts)
                if trade_pred[i] == 1 and ts in dir_idx_set]

    if dir_eval:
        dp_eval = (dir_p1[dir_eval] >= 0.5).astype(int)
        dt_eval = test_dir.loc[[valid_ts[i] for i in dir_eval], "label"].values
        print("\n[Stage 2] Direction (trade-predicted bars with ground truth):")
        print(classification_report(dt_eval, dp_eval,
                                    target_names=["short", "long"], digits=4))
        cm2 = confusion_matrix(dt_eval, dp_eval)
        s2_acc = float((dp_eval == dt_eval).mean())
        s2_report = classification_report(dt_eval, dp_eval,
                                          target_names=["short", "long"],
                                          digits=4, output_dict=True)
    else:
        print("\n[Stage 2] No direction ground truth available.")
        s2_acc, s2_report, cm2 = None, None, [[0, 0], [0, 0]]

    signals = np.where(trade_p1 < TRADE_THRESHOLD, 0,
               np.where(dir_p1 >= 0.5, 1, -1)).astype(int)
    market_ret = test_orig["market_return"].reindex(valid_ts).fillna(0.0).values

    print("\n[Backtest] TA-Only simulation...")
    bt = _backtest(signals, market_ret)
    print(f"  Trades       : {bt['n_trades']:,}")
    print(f"  Win rate     : {bt.get('win_rate', 0):.2%}")
    print(f"  Cum P&L      : {bt.get('cumulative_pnl_pct', 0):+.2f}%")
    print(f"  Avg trade    : {bt.get('avg_trade_pnl_pct', 0):+.4f}%")
    print(f"  Sharpe       : {bt.get('sharpe_annualized', 0):.3f}")
    print(f"  Max drawdown : {bt.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Buy & Hold   : {bt.get('buy_and_hold_pct', 0):+.2f}%")

    return {
        "test_period": {
            "start": str(valid_ts[0].date()),
            "end": str(valid_ts[-1].date()),
            "n_bars": len(valid_ts),
        },
        "stage1": {
            "accuracy": s1_acc,
            "n_predicted_trades": int(trade_pred.sum()),
            "confusion_matrix": cm1.tolist(),
            "report": s1_report,
        },
        "stage2": {
            "accuracy": s2_acc,
            "n_evaluated": len(dir_eval),
            "confusion_matrix": cm2 if isinstance(cm2, list) else cm2.tolist(),
            "report": s2_report,
        },
        "backtest_ta_only": {k: v for k, v in bt.items() if k != "equity_curve"},
        "_arrays": {
            "valid_ts": [str(ts) for ts in valid_ts],
            "signals": signals.tolist(),
            "market_returns": market_ret.tolist(),
            "trade_probs": trade_p1.tolist(),
            "dir_probs": dir_p1.tolist(),
        },
    }


if __name__ == "__main__":
    evaluate()
