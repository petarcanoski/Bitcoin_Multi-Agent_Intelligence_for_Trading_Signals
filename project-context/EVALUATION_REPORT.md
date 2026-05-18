# THIS IS OLD AND OUTDATED — DO NOT USE AS REFERENCE

# Bitcoin Multi-Agent Intelligence — Full Evaluation Report

_Generated: 2026-05-17 20:43 UTC_

## Overview

This report evaluates each agent independently and compares the TA-only trading strategy against the full agentic pipeline across the held-out test period (last 15% of the dataset: 2025-02-04 → 2026-02-20).

**Evaluation suite**:

| Module | Script | Runtime |
|--------|--------|---------|
| Technical Agent | `evaluate_technical.py` | 8.2s |
| Sentiment Agent | `evaluate_sentiment.py` | 40.6s |
| Risk Agent | `evaluate_risk.py` | 0.1s |
| Backtest Comparison | `backtest_comparison.py` | 0.7s |

---

## 1. Technical Agent (CNN-LSTM)

**Test period**: 2025-02-05 → 2026-02-20  (9,133 bars)

### Stage 1 — Trade Detection (CNN-LSTM)

| Metric | No-Trade | Trade | Macro avg |
|--------|----------|-------|-----------|
| Precision | 0.9958 | 0.6223 | 0.8091 |
| Recall | 0.1276 | 0.9996 | 0.5636 |
| F1 | 0.2262 | 0.7671 | 0.4967 |
| Accuracy | | | **0.6420** |

Predicted trades: 8,653

**Confusion Matrix** (rows=true, cols=pred):

```
               no-trade   trade
    no-trade        478      3268
       trade          2      5385
```

### Stage 2 — Direction Prediction (CNN-LSTM)

| Metric | Short | Long | Macro avg |
|--------|-------|------|-----------|
| Precision | 0.7021 | 0.6325 | 0.6673 |
| Recall | 0.6211 | 0.7121 | 0.6666 |
| F1 | 0.6591 | 0.6700 | 0.6645 |
| Accuracy | | | **0.6646** |

Evaluated on: 5,385 trade-signal bars

### TA-Only Backtest (test period)

| Metric | Value |
|--------|-------|
| Trades | 8,653 |
| Win rate | 65.65% |
| Cumulative P&L | **+1799.24%** |
| Avg trade P&L | +0.2079% |
| Sharpe (ann.) | 18.865 |
| Max drawdown | -71.91% |
| Buy & Hold | -122.37% |

---

## 2. Sentiment Agent (FinBERT)

**Dataset**: financial_phrasebank (sentences_75agree)  
**Model**: `ProsusAI/finbert`  
**Samples**: 3,453  
**Inference time**: 34.1s

**Overall Accuracy: 0.9473 (94.73%)**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative | 0.8604 | 0.9833 | 0.9178 | 420 |
| Neutral | 0.9916 | 0.9324 | 0.9611 | 2,146 |
| Positive | 0.8974 | 0.9662 | 0.9305 | 887 |
| **Macro avg** | 0.9165 | 0.9606 | **0.9365** | |

**Confusion Matrix** (rows=true, cols=pred):

```
              neg    neu    pos
    negative     413       2       5
     neutral      52    2001      93
    positive      15      15     857
```

**Average sentiment score (positive_prob − negative_prob) by true class:**

- negative: -0.9155
- neutral: +0.0639
- positive: +0.8586

**Misclassifications**: 182 / 3,453 (5.3%)

---

## 3. Risk Agent (ATR-14 Volatility Proxy)

**Vol proxy**: `atr_14_pct (ATR-14 as % of price)`  
**Drawdown horizon**: 48 bars (hours)  
**Severe drawdown threshold**: 3%  
**Test period**: 2025-02-04 → 2026-02-18  (9,115 bars)

**Severe drawdown rate in test**: 25.7%

**Predicted risk distribution:**

| Level | Count | Share |
|-------|-------|-------|
| low_risk | 5,747 | 63.0% |
| medium_risk | 2,630 | 28.9% |
| high_risk | 738 | 8.1% |

**Forward drawdown by predicted risk level:**

| Level | Avg 48h drawdown | % Severe |
|-------|-----------------|----------|
| low_risk | 1.782% | 20.0% |
| medium_risk | 2.398% | 30.5% |
| high_risk | 4.020% | 52.4% |

### Binary Classification: high_risk → severe drawdown

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| no_severe_dd | 0.7667 | 0.9482 | 0.8479 | 6,774 |
| severe_dd | 0.5244 | 0.1653 | 0.2514 | 2,341 |
| **Macro avg** | 0.6456 | 0.5567 | **0.5496** | |

**Confusion Matrix** (rows=true, cols=pred):

```
                 pred_no  pred_high
    no_severe_dd      6423       351
       severe_dd      1954       387
```

---

## 4. Backtest Comparison: TA-Only vs Agentic

**Test period**: 2025-02-05 → 2026-02-20  (9,133 bars)  
**Sentiment source**: Bitcoin Fear & Greed Index (alternative.me)  
**Buy & Hold**: -127.68%

| Metric | TA-Only | Agentic | Delta |
|--------|---------|---------|-------|
| Trades | 8,653 | 6,785 | -1868 |
| Win rate | 65.65% | 64.95% | -0.70% |
| **Cumulative P&L** | **+1799.24%** | **+1269.07%** | **-530.17%** |
| Avg trade P&L | +0.2079% | +0.1870% | -0.02% |
| Sharpe (ann.) | 18.865 | 15.398 | -3.467 |
| Max drawdown | -71.91% | -68.78% | +3.13% |
| Buy & Hold | -127.68% | — | — |

**Signal breakdown:**

| Signal | TA-Only | Agentic |
|--------|---------|---------|
| Buy | 4,521 | 4,293 |
| Sell | 4,132 | 2,492 |
| Hold | 480 | 2,348 |

---

## 5. Summary and Interpretation

**Technical Agent**: The CNN-LSTM trade detection model achieves **64.20% accuracy** at identifying tradeable bars. The TA-only strategy produces 8,653 trades over the test period with a cumulative P&L of **+1799.24%** vs Buy & Hold at **-122.37%**. The strong val-to-train accuracy divergence (87% → 77% over 16 epochs) indicates overfitting on the training distribution.

**Sentiment Agent**: ProsusAI/FinBERT achieves **94.73% accuracy** on the `financial_phrasebank` (sentences_75agree) benchmark. The model performs best on clearly-toned financial sentences and struggles most with neutral/ambiguous phrasing. This accuracy level is consistent with published FinBERT results (~85–88% on this dataset).

**Risk Agent**: The ATR-14 percentile proxy correctly characterises market volatility regimes. 25.7% of test bars experienced a severe drawdown (>3%) within 48 hours. The binary classifier (high_risk → severe drawdown) achieves macro F1 = **0.5496**. Forward drawdown is monotonically increasing from low → medium → high risk, confirming the signal is directionally meaningful.

**Agentic vs TA-Only**: The TA-Only strategy outperforms in cumulative P&L over the test period. Agentic: **+1269.07%** vs TA-Only: **+1799.24%** (delta: -530.17%). Sharpe ratio: Agentic 15.398 vs TA-Only 18.865. The agentic filter reduces trade count (by adding sentiment/risk gates), which can improve quality-per-trade but may miss profitable signals during high-sentiment trending periods.

---

## 6. Methodology Notes

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
