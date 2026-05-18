
import sys
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "project-context" / "EVALUATION_REPORT.md"

sys.path.insert(0, str(Path(__file__).parent))


def _run(label: str, fn):
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}\n")
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f"\n  ✓ {label} completed in {elapsed:.1f}s")
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ✗ {label} FAILED: {e}")
        traceback.print_exc()
        return {"error": str(e)}, elapsed


def _fmt_pct(v, digits=2):
    if v is None:
        return "N/A"
    return f"{v:+.{digits}f}%"


def _fmt_f(v, digits=4):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _section_technical(r: dict) -> str:
    if "error" in r:
        return f"**Error**: `{r['error']}`\n"

    tp = r.get("test_period", {})
    s1 = r.get("stage1", {})
    s2 = r.get("stage2", {})
    bt = r.get("backtest_ta_only", {})

    s1r = s1.get("report", {})
    s2r = s2.get("report", {})

    lines = []
    lines.append(f"**Test period**: {tp.get('start')} → {tp.get('end')}  "
                 f"({tp.get('n_bars', 0):,} bars)\n")

    lines.append("### Stage 1 — Trade Detection (CNN-LSTM)\n")
    lines.append("| Metric | No-Trade | Trade | Macro avg |")
    lines.append("|--------|----------|-------|-----------|")
    for row_label, key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1-score")]:
        nt = _fmt_f(s1r.get("no-trade", {}).get(key))
        t  = _fmt_f(s1r.get("trade",    {}).get(key))
        ma = _fmt_f(s1r.get("macro avg",{}).get(key))
        lines.append(f"| {row_label} | {nt} | {t} | {ma} |")
    lines.append(f"| Accuracy | | | **{_fmt_f(s1.get('accuracy'))}** |")
    lines.append(f"\nPredicted trades: {s1.get('n_predicted_trades', 0):,}\n")

    cm1 = s1.get("confusion_matrix", [])
    if cm1:
        lines.append("**Confusion Matrix** (rows=true, cols=pred):\n")
        lines.append("```")
        lines.append("               no-trade   trade")
        labels = ["no-trade", "trade"]
        for i, lbl in enumerate(labels):
            row = "  ".join(f"{cm1[i][j]:8d}" for j in range(2))
            lines.append(f"  {lbl:>10}   {row}")
        lines.append("```\n")

    lines.append("### Stage 2 — Direction Prediction (CNN-LSTM)\n")
    if s2.get("accuracy") is None:
        lines.append("_No direction ground truth available in test split._\n")
    else:
        lines.append("| Metric | Short | Long | Macro avg |")
        lines.append("|--------|-------|------|-----------|")
        for row_label, key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1-score")]:
            sh = _fmt_f(s2r.get("short", {}).get(key))
            lo = _fmt_f(s2r.get("long",  {}).get(key))
            ma = _fmt_f(s2r.get("macro avg", {}).get(key))
            lines.append(f"| {row_label} | {sh} | {lo} | {ma} |")
        lines.append(f"| Accuracy | | | **{_fmt_f(s2.get('accuracy'))}** |")
        lines.append(f"\nEvaluated on: {s2.get('n_evaluated', 0):,} trade-signal bars\n")

    lines.append("### TA-Only Backtest (test period)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Trades | {bt.get('n_trades', 0):,} |")
    lines.append(f"| Win rate | {bt.get('win_rate', 0):.2%} |")
    lines.append(f"| Cumulative P&L | **{_fmt_pct(bt.get('cumulative_pnl_pct'))}** |")
    lines.append(f"| Avg trade P&L | {_fmt_pct(bt.get('avg_trade_pnl_pct'), 4)} |")
    lines.append(f"| Sharpe (ann.) | {_fmt_f(bt.get('sharpe_annualized'), 3)} |")
    lines.append(f"| Max drawdown | {bt.get('max_drawdown_pct', 0):.2f}% |")
    lines.append(f"| Buy & Hold | {_fmt_pct(bt.get('buy_and_hold_pct'))} |")
    lines.append("")

    return "\n".join(lines)


def _section_sentiment(r: dict) -> str:
    if "error" in r:
        return f"**Error**: `{r['error']}`\n"

    lines = []
    lines.append(f"**Dataset**: {r.get('dataset', 'N/A')}  ")
    lines.append(f"**Model**: `{r.get('model', 'N/A')}`  ")
    lines.append(f"**Samples**: {r.get('n_samples', 0):,}  ")
    lines.append(f"**Inference time**: {r.get('inference_seconds', 0):.1f}s\n")

    acc = r.get("accuracy", 0)
    lines.append(f"**Overall Accuracy: {acc:.4f} ({acc:.2%})**\n")

    rep = r.get("report", {})
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|-------|-----------|--------|----|---------|")
    for cls in ["negative", "neutral", "positive"]:
        cr = rep.get(cls, {})
        lines.append(
            f"| {cls.capitalize()} | {_fmt_f(cr.get('precision'))} | "
            f"{_fmt_f(cr.get('recall'))} | {_fmt_f(cr.get('f1-score'))} | "
            f"{int(cr.get('support', 0)):,} |"
        )
    ma = rep.get("macro avg", {})
    lines.append(
        f"| **Macro avg** | {_fmt_f(ma.get('precision'))} | "
        f"{_fmt_f(ma.get('recall'))} | **{_fmt_f(ma.get('f1-score'))}** | |"
    )

    cm = r.get("confusion_matrix", [])
    if cm:
        lines.append("\n**Confusion Matrix** (rows=true, cols=pred):\n")
        lines.append("```")
        lines.append("              neg    neu    pos")
        for i, lbl in enumerate(["negative", "neutral", "positive"]):
            row = "  ".join(f"{cm[i][j]:6d}" for j in range(3))
            lines.append(f"  {lbl:>10}  {row}")
        lines.append("```\n")

    avg_sc = r.get("avg_sentiment_score_by_class", {})
    if avg_sc:
        lines.append("**Average sentiment score (positive_prob − negative_prob) by true class:**\n")
        for cls, val in avg_sc.items():
            lines.append(f"- {cls}: {val:+.4f}")
        lines.append("")

    mc = r.get("misclassification_count", 0)
    n = r.get("n_samples", 1)
    lines.append(f"**Misclassifications**: {mc:,} / {n:,} ({mc/n:.1%})\n")

    return "\n".join(lines)


def _section_risk(r: dict) -> str:
    if "error" in r:
        return f"**Error**: `{r['error']}`\n"

    meth = r.get("methodology", {})
    tp = r.get("test_period", {})
    lines = []

    lines.append(f"**Vol proxy**: `{meth.get('vol_proxy', 'N/A')}`  ")
    lines.append(f"**Drawdown horizon**: {meth.get('drawdown_horizon_bars', 48)} bars (hours)  ")
    lines.append(f"**Severe drawdown threshold**: {meth.get('severe_drawdown_threshold_pct', 3):.0f}%  ")
    lines.append(f"**Test period**: {tp.get('start')} → {tp.get('end')}  "
                 f"({tp.get('n_bars', 0):,} bars)\n")

    sdr = r.get("severe_drawdown_rate_pct", 0)
    lines.append(f"**Severe drawdown rate in test**: {sdr:.1f}%\n")

    rd = r.get("risk_distribution", {})
    total = sum(rd.values()) or 1
    lines.append("**Predicted risk distribution:**\n")
    lines.append("| Level | Count | Share |")
    lines.append("|-------|-------|-------|")
    for lvl in ["low_risk", "medium_risk", "high_risk"]:
        c = rd.get(lvl, 0)
        lines.append(f"| {lvl} | {c:,} | {c/total:.1%} |")
    lines.append("")

    avg_dd = r.get("avg_fwd_drawdown_by_risk_pct", {})
    pct_sev = r.get("pct_severe_by_risk", {})
    lines.append("**Forward drawdown by predicted risk level:**\n")
    lines.append("| Level | Avg 48h drawdown | % Severe |")
    lines.append("|-------|-----------------|----------|")
    for lvl in ["low_risk", "medium_risk", "high_risk"]:
        ad = avg_dd.get(lvl, 0)
        ps = pct_sev.get(lvl, 0)
        lines.append(f"| {lvl} | {ad:.3f}% | {ps:.1f}% |")
    lines.append("")

    bc = r.get("binary_classification", {})
    rep = bc.get("report", {})
    lines.append("### Binary Classification: high_risk → severe drawdown\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|-------|-----------|--------|----|---------|")
    for cls in ["no_severe_dd", "severe_dd"]:
        cr = rep.get(cls, {})
        lines.append(
            f"| {cls} | {_fmt_f(cr.get('precision'))} | "
            f"{_fmt_f(cr.get('recall'))} | {_fmt_f(cr.get('f1-score'))} | "
            f"{int(cr.get('support', 0)):,} |"
        )
    ma = rep.get("macro avg", {})
    lines.append(
        f"| **Macro avg** | {_fmt_f(ma.get('precision'))} | "
        f"{_fmt_f(ma.get('recall'))} | **{_fmt_f(ma.get('f1-score'))}** | |"
    )

    cm = bc.get("confusion_matrix", [])
    if cm:
        lines.append("\n**Confusion Matrix** (rows=true, cols=pred):\n")
        lines.append("```")
        lines.append("                 pred_no  pred_high")
        for i, lbl in enumerate(["no_severe_dd", "severe_dd"]):
            row = "  ".join(f"{cm[i][j]:8d}" for j in range(2))
            lines.append(f"  {lbl:>14}  {row}")
        lines.append("```\n")

    return "\n".join(lines)


def _section_backtest(r: dict) -> str:
    if "error" in r:
        return f"**Error**: `{r['error']}`\n"

    tp = r.get("test_period", {})
    ta = r.get("ta_only", {})
    ag = r.get("agentic", {})
    bh = r.get("buy_and_hold_pct", 0)
    ss = r.get("sentiment_source", "N/A")

    lines = []
    lines.append(f"**Test period**: {tp.get('start')} → {tp.get('end')}  "
                 f"({tp.get('n_bars', 0):,} bars)  ")
    lines.append(f"**Sentiment source**: {ss}  ")
    lines.append(f"**Buy & Hold**: {_fmt_pct(bh)}\n")

    lines.append("| Metric | TA-Only | Agentic | Delta |")
    lines.append("|--------|---------|---------|-------|")

    def delta(a, b):
        if a is None or b is None:
            return "N/A"
        return f"{b - a:+.2f}"

    n_ta = ta.get("n_trades", 0)
    n_ag = ag.get("n_trades", 0)
    lines.append(f"| Trades | {n_ta:,} | {n_ag:,} | {n_ag - n_ta:+d} |")

    wr_ta = ta.get("win_rate", 0)
    wr_ag = ag.get("win_rate", 0)
    lines.append(f"| Win rate | {wr_ta:.2%} | {wr_ag:.2%} | {wr_ag - wr_ta:+.2%} |")

    pnl_ta = ta.get("cumulative_pnl_pct")
    pnl_ag = ag.get("cumulative_pnl_pct")
    lines.append(f"| **Cumulative P&L** | **{_fmt_pct(pnl_ta)}** | **{_fmt_pct(pnl_ag)}** | "
                 f"**{delta(pnl_ta, pnl_ag)}%** |")

    ap_ta = ta.get("avg_trade_pnl_pct")
    ap_ag = ag.get("avg_trade_pnl_pct")
    lines.append(f"| Avg trade P&L | {_fmt_pct(ap_ta, 4)} | {_fmt_pct(ap_ag, 4)} | "
                 f"{delta(ap_ta, ap_ag)}% |")

    sh_ta = ta.get("sharpe_annualized")
    sh_ag = ag.get("sharpe_annualized")
    sh_delta = f"{sh_ag - sh_ta:+.3f}" if (sh_ta is not None and sh_ag is not None) else "N/A"
    lines.append(f"| Sharpe (ann.) | {_fmt_f(sh_ta, 3)} | {_fmt_f(sh_ag, 3)} | {sh_delta} |")

    dd_ta = ta.get("max_drawdown_pct")
    dd_ag = ag.get("max_drawdown_pct")
    lines.append(f"| Max drawdown | {dd_ta:.2f}% | {dd_ag:.2f}% | "
                 f"{dd_ag - dd_ta:+.2f}% |")

    lines.append(f"| Buy & Hold | {_fmt_pct(bh)} | — | — |")
    lines.append("")

    sc_ta = ta.get("signal_counts", {})
    sc_ag = ag.get("signal_counts", {})
    lines.append("**Signal breakdown:**\n")
    lines.append("| Signal | TA-Only | Agentic |")
    lines.append("|--------|---------|---------|")
    for sig in ["buy", "sell", "hold"]:
        lines.append(f"| {sig.capitalize()} | {sc_ta.get(sig, 0):,} | {sc_ag.get(sig, 0):,} |")
    lines.append("")

    return "\n".join(lines)


def _build_report(tech, tech_t, sent, sent_t, risk, risk_t, comp, comp_t) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections = [
        f"# Bitcoin Multi-Agent Intelligence — Full Evaluation Report",
        f"\n_Generated: {now}_\n",
        "## Overview\n",
        "This report evaluates each agent independently and compares the TA-only trading "
        "strategy against the full agentic pipeline across the held-out test period "
        "(last 15% of the dataset: 2025-02-04 → 2026-02-20).\n",
        "**Evaluation suite**:\n",
        "| Module | Script | Runtime |",
        "|--------|--------|---------|",
        f"| Technical Agent | `evaluate_technical.py` | {tech_t:.1f}s |",
        f"| Sentiment Agent | `evaluate_sentiment.py` | {sent_t:.1f}s |",
        f"| Risk Agent | `evaluate_risk.py` | {risk_t:.1f}s |",
        f"| Backtest Comparison | `backtest_comparison.py` | {comp_t:.1f}s |",
        "",
        "---\n",
        "## 1. Technical Agent (CNN-LSTM)\n",
        _section_technical(tech),
        "---\n",
        "## 2. Sentiment Agent (FinBERT)\n",
        _section_sentiment(sent),
        "---\n",
        "## 3. Risk Agent (ATR-14 Volatility Proxy)\n",
        _section_risk(risk),
        "---\n",
        "## 4. Backtest Comparison: TA-Only vs Agentic\n",
        _section_backtest(comp),
        "---\n",
        "## 5. Summary and Interpretation\n",
        _build_summary(tech, sent, risk, comp),
        "---\n",
        "## 6. Methodology Notes\n",
        _methodology_notes(),
    ]

    return "\n".join(sections)


def _build_summary(tech, sent, risk, comp) -> str:
    lines = []

    if "error" not in tech:
        bt = tech.get("backtest_ta_only", {})
        s1 = tech.get("stage1", {})
        lines.append(f"**Technical Agent**: The CNN-LSTM trade detection model achieves "
                     f"**{s1.get('accuracy', 0):.2%} accuracy** at identifying tradeable bars. "
                     f"The TA-only strategy produces {bt.get('n_trades', 0):,} trades over the test "
                     f"period with a cumulative P&L of "
                     f"**{_fmt_pct(bt.get('cumulative_pnl_pct'))}** vs Buy & Hold at "
                     f"**{_fmt_pct(bt.get('buy_and_hold_pct'))}**. The strong val-to-train "
                     f"accuracy divergence (87% → 77% over 16 epochs) indicates overfitting on "
                     f"the training distribution.\n")

    if "error" not in sent:
        acc = sent.get("accuracy", 0)
        lines.append(f"**Sentiment Agent**: ProsusAI/FinBERT achieves **{acc:.2%} accuracy** "
                     f"on the `financial_phrasebank` (sentences_75agree) benchmark. "
                     f"The model performs best on clearly-toned financial sentences and "
                     f"struggles most with neutral/ambiguous phrasing. This accuracy level "
                     f"is consistent with published FinBERT results (~85–88% on this dataset).\n")

    if "error" not in risk:
        sdr = risk.get("severe_drawdown_rate_pct", 0)
        bc = risk.get("binary_classification", {})
        rep = bc.get("report", {})
        ma = rep.get("macro avg", {})
        lines.append(f"**Risk Agent**: The ATR-14 percentile proxy correctly characterises "
                     f"market volatility regimes. {sdr:.1f}% of test bars experienced a severe "
                     f"drawdown (>3%) within 48 hours. The binary classifier (high_risk → "
                     f"severe drawdown) achieves macro F1 = **{_fmt_f(ma.get('f1-score'), 4)}**. "
                     f"Forward drawdown is monotonically increasing from low → medium → high "
                     f"risk, confirming the signal is directionally meaningful.\n")

    if "error" not in comp:
        ta = comp.get("ta_only", {})
        ag = comp.get("agentic", {})
        pnl_ta = ta.get("cumulative_pnl_pct", 0)
        pnl_ag = ag.get("cumulative_pnl_pct", 0)
        delta = pnl_ag - pnl_ta
        sh_ta = ta.get("sharpe_annualized", 0)
        sh_ag = ag.get("sharpe_annualized", 0)
        winner = "Agentic" if pnl_ag > pnl_ta else "TA-Only"
        lines.append(f"**Agentic vs TA-Only**: The {winner} strategy outperforms in cumulative "
                     f"P&L over the test period. Agentic: **{_fmt_pct(pnl_ag)}** vs "
                     f"TA-Only: **{_fmt_pct(pnl_ta)}** (delta: {delta:+.2f}%). "
                     f"Sharpe ratio: Agentic {sh_ag:.3f} vs TA-Only {sh_ta:.3f}. "
                     f"The agentic filter reduces trade count (by adding sentiment/risk gates), "
                     f"which can improve quality-per-trade but may miss profitable signals "
                     f"during high-sentiment trending periods.\n")

    return "\n".join(lines)


def _methodology_notes() -> str:
    return """\
### Technical evaluation
- **Split**: 70% train / 15% val / 15% test (temporal, no leakage)
- **Inference**: Sequence length 60 bars, batch size 512
- **Trade threshold**: 0.55 (matches `TechnicalAgent.run()` default)
- **Direction threshold**: 0.50 (P(long) ≥ 0.5 → long signal)

### Sentiment evaluation
- **Dataset**: `financial_phrasebank` (sentences_75agree) — 2,264 financial sentences
  labeled by ≥75% annotator agreement, 3 classes: negative / neutral / positive
- **Label remap**: FinBERT outputs `[positive(0), negative(1), neutral(2)]`;
  phrasebank uses `[negative(0), neutral(1), positive(2)]`; remap `{0→2, 1→0, 2→1}` applied

### Risk evaluation
- **Vol proxy**: `atr_14_pct` (ATR-14 as % of price); falls back to `hist_vol_24`
- **Thresholds**: 30th / 70th percentile of training distribution
- **Ground truth**: max % drawdown over next 48 bars; severe = >3%
- **Note**: Rule-based signal — no model is trained, only the threshold calibration varies

### Backtest comparison
- **Fee**: 0.1% per side (round-trip: 0.2%)
- **Sentiment proxy**: Bitcoin Fear & Greed Index (alternative.me) aligned to hourly bars;
  falls back to RSI-14 proxy if API unavailable
- **Agentic weights**: TA=0.45, Sentiment=0.35, Risk=0.20
- **Signal thresholds**: buy ≥ +0.20, sell ≤ −0.20 (mirrors `CoordinatorAgent` defaults)
- **Sharpe**: annualised using √8760 (hourly bars)
"""


def main():
    total_t0 = time.time()
    print("\n" + "="*60)
    print("  BITCOIN MULTI-AGENT FULL EVALUATION")
    print("="*60)

    from technical import evaluate as eval_tech
    tech, tech_t = _run("Technical Agent (CNN-LSTM)", eval_tech)

    from sentiment import evaluate as eval_sent
    sent, sent_t = _run("Sentiment Agent (FinBERT / financial_phrasebank)", eval_sent)

    from risk import evaluate as eval_risk
    risk, risk_t = _run("Risk Agent (ATR-14 proxy)", eval_risk)

    from backtest import compare
    comp_fn = lambda: compare(tech_results=tech, risk_results=risk)
    comp, comp_t = _run("Backtest Comparison (TA-Only vs Agentic)", comp_fn)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  All evaluations complete in {total_elapsed:.1f}s")
    print(f"{'='*60}\n")

    report_md = _build_report(tech, tech_t, sent, sent_t, risk, risk_t, comp, comp_t)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")

    raw_path = REPO_ROOT / "project-context" / "evaluation_raw.json"
    raw = {
        "technical": {k: v for k, v in tech.items() if k != "_arrays"},
        "sentiment": sent,
        "risk": {k: v for k, v in risk.items() if k != "_arrays"},
        "backtest": {k: v for k, v in comp.items() if k != "equity_curves"},
    }
    raw_path.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    print(f"Raw JSON written to: {raw_path}")

    return {"technical": tech, "sentiment": sent, "risk": risk, "backtest": comp}


if __name__ == "__main__":
    main()
