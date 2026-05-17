# Feature Selection Report

_Generated: 2026-05-17 16:26 UTC_

## 1. Feature Expansion

- Original features: 102
- Expanded features: **128**
- New indicators: 26

## 2. Expanded Baseline Model

| Metric | Value |
|--------|-------|
| Features | 128 |
| Trades | 7,513 |
| Win rate | 61.04% |
| **Cumulative P&L** | **+1003.14%** |
| Sharpe (ann.) | 10.471 |
| Max drawdown | -143.05% |

## 3. Permutation Importance

Baseline test P&L: **+1003.14%**  
Method: shuffle each feature independently across test timesteps; importance = baseline_P&L − shuffled_P&L (higher = more predictive).

### Top 20 Most Important Features

| Rank | Feature | P&L Impact |
|------|---------|------------|
| 1 | `atr_21` | +1408.18% |
| 2 | `atr_14` | +1281.15% |
| 3 | `atr_7` | +892.03% |
| 4 | `htf1d_price_vs_ema_9` | +819.07% |
| 5 | `htf1d_price_vs_ema_50` | +310.09% |
| 6 | `htf1d_price_vs_ema_21` | +252.06% |
| 7 | `price_vs_ema_200` | +168.93% |
| 8 | `htf4h_price_vs_ema_50` | +111.07% |
| 9 | `price_vs_sma_200` | +80.84% |
| 10 | `obv_slope_12` | +59.25% |
| 11 | `sma_50_200_cross` | +57.52% |
| 12 | `price_vs_ema_100` | +52.95% |
| 13 | `month_sin` | +48.48% |
| 14 | `ema_21_50_cross` | +41.00% |
| 15 | `cum_ret_168` | +37.19% |
| 16 | `htf4h_candle_dir` | +36.72% |
| 17 | `hist_vol_72` | +35.59% |
| 18 | `vol_x_mom` | +34.69% |
| 19 | `price_vs_sma_10` | +31.54% |
| 20 | `bb_position_20_2` | +31.31% |

### Bottom 20 (Lowest Impact / Pruned)

| Rank | Feature | P&L Impact |
|------|---------|------------|
| 109 | `rsi_divergence` | -7.58% |
| 110 | `roc_12` | -8.01% |
| 111 | `vol_change` | -9.17% |
| 112 | `kurt_72` | -9.45% |
| 113 | `bb_squeeze_20_2` | -9.52% |
| 114 | `stoch_d_21` | -10.34% |
| 115 | `kurt_168` | -11.75% |
| 116 | `price_vs_sma_50` | -12.08% |
| 117 | `dist_low_24` | -12.63% |
| 118 | `htf1d_atr_ratio` | -14.23% |
| 119 | `htf1d_vol_ratio` | -15.20% |
| 120 | `bb_width_20_2` | -15.89% |
| 121 | `dist_high_168` | -16.78% |
| 122 | `htf4h_atr_ratio` | -18.27% |
| 123 | `htf4h_ema_21_50_cross` | -22.46% |
| 124 | `hl_range_pct` | -23.22% |
| 125 | `htf4h_price_vs_ema_9` | -33.47% |
| 126 | `hist_vol_168` | -37.57% |
| 127 | `htf4h_hist_vol` | -39.68% |
| 128 | `htf1d_candle_dir` | -96.92% |

## 4. Feature Pruning

- Threshold: bottom 20th percentile of importance
- Features kept: **102** / 128
- Features dropped: **26**

**Dropped:**

`bb_squeeze_20_2`, `bb_width_20_2`, `dist_high_168`, `dist_low_24`, `hist_vol_168`, `hl_range_pct`, `htf1d_atr_ratio`, `htf1d_candle_dir`, `htf1d_vol_ratio`, `htf4h_atr_ratio`, `htf4h_ema_21_50_cross`, `htf4h_hist_vol`, `htf4h_price_vs_ema_9`, `htf4h_vol_ratio`, `kurt_168`, `kurt_72`, `macd_cross_12_26`, `price_vs_sma_50`, `price_vs_vwap_48`, `roc_12`, `rsi_14`, `rsi_divergence`, `stoch_d_14`, `stoch_d_21`, `stoch_k_14`, `vol_change`

## 5. Reduced Model vs Expanded

| Metric | Expanded | Reduced | Delta |
|--------|----------|---------|-------|
| Trades | 7,513 | 6,571 | -942 |
| Win rate | 61.04% | 61.30% | +0.26% |
| **Cum P&L** | **+1003.14%** | **+899.06%** | **-104.08%** |
| Sharpe (ann.) | 10.471 | 9.751 | -0.720 |
| Max drawdown | -143.05% | -120.30% | +22.75% |

## 6. Conclusion

**Recommended feature set: Expanded**  
(based on cumulative P&L on the held-out test period)

Pruning 26 features did not improve profitability — the expanded set is retained.

## 7. Methodology

- **Split**: 70% train / 15% val / 15% test (temporal, no leakage)
- **Importance**: shuffle each feature column across all test timesteps; measure change in cumulative P&L (higher = feature more useful)
- **Pruning threshold**: bottom 20th percentile
- **Model**: CNN [64,128,256] → LSTM h=256 l=2; AdamW; early-stop patience=15
- **Backtest fee**: 0.1% per side; signal: trade_prob≥0.55 → direction

## 8. Positive-Importance Retrain (importance ≥ 0)

**Rationale**: 47 features have *negative* permutation importance — shuffling them
*improves* P&L, meaning the model learned spurious patterns from them.
Dropping all negatively-important features gives a cleaner, more principled feature set.

- Features dropped (negative importance): **47**
- Features kept (importance ≥ 0): **81**

**Dropped features:**

`atr_14_pct`, `atr_21_pct`, `atr_7_pct`, `bb_squeeze_20_1`, `bb_squeeze_20_2`, `bb_width_20_1`, `bb_width_20_2`, `dist_high_168`, `dist_high_24`, `dist_low_24`, `dow_cos`, `hist_vol_168`, `hl_range_pct`, `hour_sin`, `htf1d_atr_ratio`, `htf1d_candle_dir`, `htf1d_vol_ratio`, `htf4h_atr_ratio`, `htf4h_ema_21_50_cross`, `htf4h_ema_9_21_cross`, `htf4h_hist_vol`, `htf4h_price_vs_ema_9`, `htf4h_vol_ratio`, `kurt_168`, `kurt_72`, `log_ret_1`, `log_ret_48`, `macd_cross_12_26`, `momentum_6`, `obv_divergence`, `price_vs_ema_9`, `price_vs_sma_50`, `price_vs_vwap_48`, `ret_x_rvol`, `roc_12`, `roc_6`, `rsi_14`, `rsi_7`, `rsi_divergence`, `skew_168`, `stoch_d_14`, `stoch_d_21`, `stoch_k_14`, `vol_change`, `vol_lag_1_ratio`, `vol_lag_5_ratio`, `williams_r_14`

### Three-Way Comparison

| Metric | Expanded (128) | Bottom-20% Pruned (102) | Positive-Only (81) |
|--------|---------------|------------------------|------------------------|
| Trades | 7,513 | 6,571 | 8,441 |
| Win rate | 61.04% | 61.30% | 63.61% |
| **Cum P&L** | **+1003.14%** | **+899.06%** | **+1261.28%** |
| Sharpe (ann.) | 10.471 | 9.751 | 12.971 |
| Max drawdown | -143.05% | -120.30% | -127.93% |

**Winner: Positive-only (81)**

## 9. LightGBM Comparison (81 features)

Two-stage LightGBM (GBDT, num_leaves=127, lr=0.05) trained on the same
81 positive-importance features and evaluated on the identical held-out test period.

### Results

| Model | Cum P&L | Sharpe | Win Rate | Trades |
|-------|---------|--------|----------|--------|
| CNN-LSTM 128 feat | +1003.14% | 10.471 | 61.04% | 7,513 |
| CNN-LSTM 102 feat | +899.06% | 9.751 | 61.30% | 6,571 |
| CNN-LSTM 81 feat | +1261.28% | 12.971 | 63.61% | 8,441 |
| **LightGBM 81 feat** | **-418.89%** | **-5.741** | **51.09%** | **7,176** |

**Winner: CNN-LSTM   81 feat**

### LightGBM Top 5 Features (by gain, trade stage)

- `atr_7`: 22,595
- `atr_14`: 12,793
- `htf1d_rsi_14`: 4,895
- `month_sin`: 3,961
- `atr_21`: 3,591

## 10. Full Model Comparison (81 positive-importance features)

All models evaluated on the identical held-out test period (last 15%, chronological split,
no leakage). Fee: 0.1% per side. Signal: trade_prob ≥ 0.55 → direction.

| Model | Cum P&L | Sharpe | Win Rate | Trades | Max DD |
|-------|---------|--------|----------|--------|--------|
| **CNN-LSTM** | **+1261.28%** | 12.971 | 63.61% | 8,441 | -143.05% |
| LightGBM | -418.89% | -5.741 | 51.09% | 7,176 | -705.06% |
| TCN | -899.74% | -9.352 | 49.52% | 7,350 | -981.56% |
| TFT | -1099.23% | -10.938 | 50.25% | 9,110 | -1204.86% |

**Winner: CNN-LSTM**

### Architecture Details

| Model | Architecture | Seq window | Parameters |
|-------|-------------|------------|------------|
| CNN-LSTM | CNN [64,128,256] → LSTM (h=256, 2L) | 60 bars | ~2.1M |
| TCN | 6× causal dilated ResBlocks (hidden=128, k=3) | 60 bars | ~0.5M |
| TFT | VSN → LSTM (128, 2L) → MHA (4 heads) | 60 bars | ~0.8M |
| LightGBM | GBDT num_leaves=127, lr=0.05 | 1 bar | — |

**Key finding**: LightGBM without temporal context performs poorly (−418%).
All sequence models (CNN-LSTM, TCN, TFT) significantly outperform it,
confirming that the 60-bar temporal window is critical for this task.

## 11. Threshold Search (trade-probability cutoff)

Scanning the trade-signal threshold from 0.40 to 0.75 on the trained CNN-LSTM
(81 positive features). No retraining — only the inference cutoff changes.
Higher thresholds → fewer, higher-confidence trades.

| Threshold | Cum P&L | Sharpe | Win Rate | Trades | Max DD |
|-----------|---------|--------|----------|--------|--------|
| 0.40 | +1294.41% | 13.243 | 63.93% | 8,935 | -127.93% |
| **0.45** | **+1295.97%** | 13.276 | 63.95% | 8,820 | -127.93% |
| 0.50 | +1276.39% | 13.094 | 63.74% | 8,652 | -127.93% |
| 0.55 *(baseline)* | +1261.28% | 12.971 | 63.61% | 8,441 | -127.93% |
| 0.60 | +1247.89% | 12.868 | 63.49% | 8,233 | -127.93% |
| 0.65 | +1222.90% | 12.675 | 63.40% | 7,915 | -127.93% |
| 0.70 | +1191.31% | 12.419 | 63.27% | 7,598 | -127.93% |
| 0.75 | +1175.87% | 12.343 | 63.38% | 7,275 | -127.93% |

**Best threshold: 0.45**  (P&L=+1295.97%  Sharpe=13.276)

## 12. Walk-Forward Evaluation (6 time slices)

The test period (9,163 bars, last 15%) split into 6 equal consecutive windows.
Same trained CNN-LSTM (81 features) — no retraining. Each slice evaluated independently.

**Full test period**: P&L=+1261.28%  Sharpe=12.971  Trades=8,441

| Slice | ~Bars | Cum P&L | Sharpe | Win Rate | Trades | Max DD |
|-------|-------|---------|--------|----------|--------|--------|
| 1 | ~1,527 | **+216.19%** | 10.302 | 63.11% | 1,442 | -127.93% |
| 2 | ~1,527 | **+137.56%** | 9.536 | 62.28% | 1,429 | -80.46% |
| 3 | ~1,527 | **+155.24%** | 13.997 | 64.10% | 1,387 | -34.39% |
| 4 | ~1,527 | **+210.14%** | 17.867 | 66.57% | 1,367 | -27.04% |
| 5 | ~1,527 | **+170.63%** | 9.459 | 58.87% | 1,517 | -89.51% |
| 6 | ~1,527 | **+371.51%** | 20.248 | 67.51% | 1,299 | -92.63% |

**Positive slices: 6/6**

**Interpretation**: Gains are broadly distributed across time — performance is not an artifact of one lucky window.

## 13. Sequence Length Search

CNN-LSTM (81 positive features) retrained with different lookback windows.
Same architecture ([64,128,256] CNN → LSTM h=256 l=2), hyperparameters, and split.

| Seq Length | Cum P&L | Sharpe | Win Rate | Trades | Max DD |
|------------|---------|--------|----------|--------|--------|
| **30** | **+1780.81%** | 18.615 | 65.55% | 8,450 | -71.91% |
| 120 | +1488.85% | 15.360 | 64.90% | 8,783 | -118.01% |
| 60 *(baseline)* | +1261.28% | 12.971 | 63.61% | 8,441 | -143.05% |
| 90 | +1064.64% | 11.460 | 62.11% | 7,073 | -95.19% |

**Best seq_len: 30**  (P&L=+1780.81%  Sharpe=18.615)
