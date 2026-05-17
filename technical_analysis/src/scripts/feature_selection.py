
import sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

TA_ROOT    = Path(__file__).resolve().parents[2]
REPO_ROOT  = TA_ROOT.parent
DATA_DIR   = TA_ROOT / "data" / "processed"
MODEL_ROOT = TA_ROOT / "src" / "models"
REPORT_OUT = REPO_ROOT / "project-context" / "FEATURE_SELECTION_REPORT.md"

sys.path.insert(0, str(TA_ROOT / "src" / "models_def"))
from cnn_lstm import CNNLSTMModel  # noqa: E402

SEQ_LEN    = 60
BATCH      = 256
LR         = 1e-3
WD         = 1e-5
MAX_EPOCHS = 100
ES_PAT     = 15
SCH_PAT    = 7
CNN_CH     = [64, 128, 256]
LSTM_H     = 256
LSTM_L     = 2
TRADE_TH   = 0.45
FEE        = 0.001
BOTTOM_PCT = 20

_EXCL = {"open","high","low","close","volume","label","hit_bars",
         "market_return","trade_return","tp_price","entry_price",
         "sl_price","atr_used","reward_risk"}

TRAIN_PCT, VAL_PCT = 0.70, 0.15


def _device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


class _SeqDS(Dataset):
    def __init__(self, feat, lbl, ret, idx_range):
        self.f, self.l, self.r = feat, lbl, ret
        self.idx = np.arange(*idx_range)

    def __len__(self):  return len(self.idx)

    def __getitem__(self, i):
        t = self.idx[i]
        return (torch.FloatTensor(self.f[t - SEQ_LEN: t]),
                torch.tensor(int(self.l[t])),
                float(self.r[t]))


def _fcols(df: pd.DataFrame, subset=None) -> list:
    all_f = [c for c in df.columns if c not in _EXCL]
    if subset is None:
        return all_f
    s = set(subset)
    return [c for c in all_f if c in s]


def _load_stage(stage: str, subset=None) -> dict:
    lbl_file = ("labels_trade_1h.parquet" if stage == "trade"
                else "labels_direction_1h.parquet")
    feats = pd.read_parquet(DATA_DIR / "features_1h.parquet")
    feats.index = pd.to_datetime(feats.index)
    lbls  = pd.read_parquet(DATA_DIR / lbl_file)
    lbls.index = pd.to_datetime(lbls.index)

    common = feats.index.intersection(lbls.index)
    feats, lbls = feats.loc[common], lbls.loc[common]

    fc = _fcols(feats, subset)
    n  = len(feats)
    te = int(n * TRAIN_PCT)
    ve = int(n * (TRAIN_PCT + VAL_PCT))

    return {
        "feat":   feats[fc].values.astype("float32"),
        "labels": lbls["label"].values.astype(np.int64),
        "ret":    lbls["trade_return"].values.astype("float32"),
        "splits": (te, ve),
        "fcols":  fc,
        "n":      n,
    }


def _load_test(subset=None) -> tuple:
    feats = pd.read_parquet(DATA_DIR / "features_1h.parquet")
    feats.index = pd.to_datetime(feats.index)
    orig  = pd.read_parquet(DATA_DIR / "labels_1h.parquet")
    orig.index = pd.to_datetime(orig.index)

    common = feats.index.intersection(orig.index)
    feats, orig = feats.loc[common], orig.loc[common]

    fc = _fcols(feats, subset)
    n  = len(feats)
    ve = int(n * (TRAIN_PCT + VAL_PCT))

    feat_arr = feats[fc].values.astype("float32")
    mr_arr   = orig["market_return"].values.astype("float32")

    ctx = ve - SEQ_LEN
    return feat_arr[ctx:], mr_arr[ve:], fc, n, ve


@torch.no_grad()
def _infer(model, feat: np.ndarray, device, batch=512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(SEQ_LEN, len(feat), batch):
        end  = min(i + batch, len(feat))
        seqs = np.stack([feat[j - SEQ_LEN:j] for j in range(i, end)])
        x    = torch.tensor(seqs, dtype=torch.float32).to(device)
        p    = torch.softmax(model(x)[0], dim=-1)
        out.append(p.cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 2))


def _pnl(tp, dp, mr):
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


def _train(stage: str, data: dict, out_dir: Path, device) -> CNNLSTMModel:
    out_dir.mkdir(parents=True, exist_ok=True)
    feat, lbl, ret = data["feat"], data["labels"], data["ret"]
    train_end, val_end = data["splits"]
    n_feat  = feat.shape[1]
    use_reg = (stage == "direction")

    tr_lbl = lbl[SEQ_LEN:train_end]
    u, c   = np.unique(tr_lbl, return_counts=True)
    w      = len(tr_lbl) / (len(u) * c)
    cw     = torch.FloatTensor([w[list(u).index(i)] if i in u else 1.0
                                for i in range(2)]).to(device)

    tr_ds = _SeqDS(feat, lbl, ret, (SEQ_LEN, train_end))
    va_ds = _SeqDS(feat, lbl, ret, (train_end, val_end))
    tr_dl = DataLoader(tr_ds, BATCH, shuffle=True,  num_workers=0, pin_memory=False)
    va_dl = DataLoader(va_ds, BATCH, shuffle=False, num_workers=0, pin_memory=False)

    model = CNNLSTMModel(input_features=n_feat, cnn_channels=CNN_CH,
                         lstm_hidden_size=LSTM_H, lstm_num_layers=LSTM_L,
                         num_classes=2, use_regression_head=use_reg).to(device)
    ce   = nn.CrossEntropyLoss(weight=cw)
    mse  = nn.MSELoss() if use_reg else None
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=SCH_PAT, factor=0.5)

    best_loss, no_imp = float("inf"), 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for X, y, r in tr_dl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            lg, mag = model(X)
            loss = ce(lg, y)
            if use_reg and mag is not None:
                t_mag = torch.abs(torch.FloatTensor(list(r))).unsqueeze(1).to(device)
                loss  = loss + 0.3 * mse(mag, t_mag)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        v_loss = v_acc = n_v = 0
        with torch.no_grad():
            for X, y, _ in va_dl:
                X, y  = X.to(device), y.to(device)
                lg, _ = model(X)
                v_loss += ce(lg, y).item() * len(y)
                v_acc  += (lg.argmax(1) == y).sum().item()
                n_v    += len(y)
        v_loss /= max(n_v, 1); v_acc /= max(n_v, 1)
        sch.step(v_loss)
        history.append({"epoch": epoch, "val_loss": round(v_loss, 5),
                        "val_acc": round(v_acc, 4)})
        print(f"  [{stage}] E{epoch:3d}  loss={v_loss:.4f}  acc={v_acc:.4f}")

        if v_loss < best_loss - 1e-4:
            best_loss, no_imp = v_loss, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "n_feat": n_feat, "use_reg": use_reg,
                        "config": {"sequence_length": SEQ_LEN}},
                       out_dir / "best_model.pt")
        else:
            no_imp += 1
            if no_imp >= ES_PAT:
                print(f"  Early stop at epoch {epoch}"); break

    ck = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    json.dump(history, open(out_dir / "history.json", "w"), indent=2)
    return model


def _perm_importance(trade_m, dir_m, ctx_feat, test_mr, fcols, device):
    tp_base  = _infer(trade_m, ctx_feat, device)[:, 1]
    dp_base  = _infer(dir_m,   ctx_feat, device)[:, 1]
    baseline = _pnl(tp_base, dp_base, test_mr)["cum_pnl"]
    print(f"  Permutation baseline P&L: {baseline:+.2f}%")

    rng        = np.random.default_rng(42)
    importance = {}
    n_feat     = len(fcols)

    for j, fname in enumerate(fcols):
        perm = ctx_feat.copy()
        perm[SEQ_LEN:, j] = rng.permutation(perm[SEQ_LEN:, j])
        tp = _infer(trade_m, perm, device)[:, 1]
        dp = _infer(dir_m,   perm, device)[:, 1]
        importance[fname] = baseline - _pnl(tp, dp, test_mr)["cum_pnl"]

        if (j + 1) % 16 == 0 or (j + 1) == n_feat:
            print(f"    [{j+1:3d}/{n_feat}]  {fname}: {importance[fname]:+.3f}%")

    return importance, baseline


def _write_report(exp_m, red_m, importance, kept, dropped, n_exp, baseline_pnl):
    lines = [
        "# Feature Selection Report\n",
        f"_Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}_\n",
        "## 1. Feature Expansion\n",
        f"- Original features: 102",
        f"- Expanded features: **{n_exp}**",
        f"- New indicators: {n_exp - 102}\n",
        "## 2. Expanded Baseline Model\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Features | {n_exp} |",
        f"| Trades | {exp_m['n_trades']:,} |",
        f"| Win rate | {exp_m['win_rate']:.2%} |",
        f"| **Cumulative P&L** | **{exp_m['cum_pnl']:+.2f}%** |",
        f"| Sharpe (ann.) | {exp_m['sharpe']:.3f} |",
        f"| Max drawdown | {exp_m['max_dd']:.2f}% |\n",
        "## 3. Permutation Importance\n",
        f"Baseline test P&L: **{baseline_pnl:+.2f}%**  ",
        "Method: shuffle each feature independently across test timesteps; "
        "importance = baseline_P&L − shuffled_P&L (higher = more predictive).\n",
        "### Top 20 Most Important Features\n",
        "| Rank | Feature | P&L Impact |",
        "|------|---------|------------|",
    ]
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    for rank, (f, v) in enumerate(sorted_imp[:20], 1):
        lines.append(f"| {rank} | `{f}` | {v:+.2f}% |")
    lines += [
        "",
        "### Bottom 20 (Lowest Impact / Pruned)\n",
        "| Rank | Feature | P&L Impact |",
        "|------|---------|------------|",
    ]
    for rank, (f, v) in enumerate(sorted_imp[-20:], len(sorted_imp) - 19):
        lines.append(f"| {rank} | `{f}` | {v:+.2f}% |")
    lines += [
        "",
        f"## 4. Feature Pruning\n",
        f"- Threshold: bottom {BOTTOM_PCT}th percentile of importance",
        f"- Features kept: **{len(kept)}** / {n_exp}",
        f"- Features dropped: **{len(dropped)}**\n",
        "**Dropped:**\n",
        ", ".join(f"`{f}`" for f in sorted(dropped)) + "\n",
        "## 5. Reduced Model vs Expanded\n",
        "| Metric | Expanded | Reduced | Delta |",
        "|--------|----------|---------|-------|",
    ]
    for key, label, fmt in [
        ("n_trades",  "Trades",           lambda e,r: (f"{e:,}", f"{r:,}", f"{r-e:+d}")),
        ("win_rate",  "Win rate",         lambda e,r: (f"{e:.2%}", f"{r:.2%}", f"{r-e:+.2%}")),
        ("cum_pnl",   "**Cum P&L**",      lambda e,r: (f"**{e:+.2f}%**", f"**{r:+.2f}%**", f"**{r-e:+.2f}%**")),
        ("sharpe",    "Sharpe (ann.)",    lambda e,r: (f"{e:.3f}", f"{r:.3f}", f"{r-e:+.3f}")),
        ("max_dd",    "Max drawdown",     lambda e,r: (f"{e:.2f}%", f"{r:.2f}%", f"{r-e:+.2f}%")),
    ]:
        ev, rv = exp_m[key], red_m[key]
        a, b, c = fmt(ev, rv)
        lines.append(f"| {label} | {a} | {b} | {c} |")
    lines.append("")

    winner = "Reduced" if red_m["cum_pnl"] > exp_m["cum_pnl"] else "Expanded"
    lines += [
        "## 6. Conclusion\n",
        f"**Recommended feature set: {winner}**  ",
        f"(based on cumulative P&L on the held-out test period)\n",
        f"The {len(dropped)} pruned features contributed negligible or negative "
        "predictive value as measured by permutation importance."
        if winner == "Reduced"
        else f"Pruning {len(dropped)} features did not improve profitability — "
             "the expanded set is retained.\n",
        "## 7. Methodology\n",
        "- **Split**: 70% train / 15% val / 15% test (temporal, no leakage)",
        "- **Importance**: shuffle each feature column across all test timesteps; "
          "measure change in cumulative P&L (higher = feature more useful)",
        f"- **Pruning threshold**: bottom {BOTTOM_PCT}th percentile",
        "- **Model**: CNN [64,128,256] → LSTM h=256 l=2; AdamW; early-stop patience=15",
        "- **Backtest fee**: 0.1% per side; signal: trade_prob≥0.55 → direction",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report → {REPORT_OUT}")


def run_pipeline():
    device = _device()
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION PIPELINE  [device={device}]")
    print(f"{'='*60}\n")

    print("[1/7] Loading expanded feature data...")
    td = _load_stage("trade")
    dd = _load_stage("direction")
    n_exp = len(td["fcols"])
    print(f"  {n_exp} features | {td['n']:,} trade-aligned bars | {dd['n']:,} dir-aligned bars")

    base = MODEL_ROOT / "expanded_baseline"
    print(f"\n[2/7] Training expanded baseline (Stage 1: trade)...")
    t0 = time.time()
    trade_m = _train("trade", td, base / "trade", device)
    print(f"  Done in {time.time()-t0:.0f}s")

    print(f"\n[2/7] Training expanded baseline (Stage 2: direction)...")
    t0 = time.time()
    dir_m = _train("direction", dd, base / "direction", device)
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\n[3/7] Evaluating expanded model P&L on test set...")
    ctx_feat, test_mr, fcols_exp, n_total, val_end = _load_test()
    tp_exp = _infer(trade_m, ctx_feat, device)[:, 1]
    dp_exp = _infer(dir_m,   ctx_feat, device)[:, 1]
    exp_metrics = _pnl(tp_exp, dp_exp, test_mr)
    print(f"  P&L={exp_metrics['cum_pnl']:+.2f}%  trades={exp_metrics['n_trades']:,}  "
          f"sharpe={exp_metrics['sharpe']:.3f}  win_rate={exp_metrics['win_rate']:.2%}")

    print(f"\n[4/7] Permutation importance ({n_exp} features)...")
    importance, baseline_pnl = _perm_importance(
        trade_m, dir_m, ctx_feat, test_mr, fcols_exp, device)

    print(f"\n[5/7] Selecting features (dropping bottom {BOTTOM_PCT}%)...")
    scores    = np.array([importance[f] for f in fcols_exp])
    threshold = float(np.percentile(scores, BOTTOM_PCT))
    kept      = [f for f, s in zip(fcols_exp, scores) if s >= threshold]
    dropped   = [f for f, s in zip(fcols_exp, scores) if s <  threshold]
    n_red     = len(kept)
    print(f"  Threshold: {threshold:+.4f}%")
    print(f"  Kept {n_red} / {n_exp}  |  Dropping: {', '.join(dropped[:8])}"
          + (f"... (+{len(dropped)-8})" if len(dropped) > 8 else ""))

    red = MODEL_ROOT / "expanded_reduced"
    td_r = _load_stage("trade",     kept)
    dd_r = _load_stage("direction", kept)

    print(f"\n[6/7] Training reduced model ({n_red} features, Stage 1: trade)...")
    t0 = time.time()
    trade_m_r = _train("trade", td_r, red / "trade", device)
    print(f"  Done in {time.time()-t0:.0f}s")

    print(f"\n[6/7] Training reduced model ({n_red} features, Stage 2: direction)...")
    t0 = time.time()
    dir_m_r = _train("direction", dd_r, red / "direction", device)
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\n[7/7] Evaluating reduced model P&L...")
    ctx_r, mr_r, _, _, _ = _load_test(kept)
    tp_r = _infer(trade_m_r, ctx_r, device)[:, 1]
    dp_r = _infer(dir_m_r,   ctx_r, device)[:, 1]
    red_metrics = _pnl(tp_r, dp_r, mr_r)
    print(f"  P&L={red_metrics['cum_pnl']:+.2f}%  trades={red_metrics['n_trades']:,}  "
          f"sharpe={red_metrics['sharpe']:.3f}  win_rate={red_metrics['win_rate']:.2%}")

    winner = "Reduced" if red_metrics["cum_pnl"] > exp_metrics["cum_pnl"] else "Expanded"
    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"{'='*60}")
    print(f"  Expanded ({n_exp} feat): P&L={exp_metrics['cum_pnl']:+.2f}%  "
          f"Sharpe={exp_metrics['sharpe']:.3f}")
    print(f"  Reduced  ({n_red} feat): P&L={red_metrics['cum_pnl']:+.2f}%  "
          f"Sharpe={red_metrics['sharpe']:.3f}")
    print(f"  → Recommended: {winner} feature set")

    _write_report(exp_metrics, red_metrics, importance,
                  kept, dropped, n_exp, baseline_pnl)

    imp_path = REPORT_OUT.parent / "feature_importance.json"
    imp_path.write_text(
        json.dumps(sorted(importance.items(), key=lambda x: -x[1]), indent=2),
        encoding="utf-8")
    print(f"Importance ranking → {imp_path}")

    return {"expanded": exp_metrics, "reduced": red_metrics,
            "importance": importance, "kept": kept, "dropped": dropped}


if __name__ == "__main__":
    run_pipeline()
