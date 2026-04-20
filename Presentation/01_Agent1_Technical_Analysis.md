# Agent 1 - Technical Analysis (OHLCV + Indicators)

## 1) What this agent does
Agent 1 is the market-structure specialist. It looks only at price/volume-derived data (OHLCV and engineered technical features) and predicts a trading direction.

In your project, this agent is already trained by your teammate and reused in inference mode (no retraining needed).

## 2) Inputs and artifacts used
From `bitcoin-predictor-dev`:

- `src/models/trade/best_model.pt`
- `src/models/direction/best_model.pt`
- `data/processed/features_1h.parquet`

The coordinator-side implementation that consumes these files is in:

- `coordinator_agent/technical_agent.py`

## 3) Model design used at inference time
The architecture loaded from checkpoints is `CNNLSTMModel`:

1. `Conv1d + BatchNorm + ReLU + Dropout` blocks extract local temporal patterns.
2. `LSTM` captures sequential dependencies across the time window.
3. `Classifier head` outputs class probabilities.

There are two separate models:

- **Trade model**: decides if conditions are strong enough to trade.
- **Direction model**: if trading is allowed, decides long (BUY) vs short (SELL).

## 4) Inference pipeline (step by step)
1. Load both checkpoint files.
2. Infer expected `input_features` directly from model weights.
3. Read `features_1h.parquet`.
4. Select usable features (exclude raw OHLCV and label/target columns).
5. Take latest `sequence_length` window (default from checkpoint config, often 60).
6. Run forward pass on trade model and direction model.
7. Convert logits to probabilities via softmax.
8. Apply decision logic:
   - If `trade_prob < 0.55` -> `HOLD`
   - Else:
     - `long_prob >= 0.5` -> `BUY`
     - otherwise -> `SELL`

## 5) Decision logic and confidence
From `coordinator_agent/technical_agent.py`:

- `trade_prob = P(trade class = 1)`
- `long_prob = P(direction class = long)`

Signal:

- `HOLD` if `trade_prob < 0.55`
- `BUY` if `trade_prob >= 0.55` and `long_prob >= 0.5`
- `SELL` if `trade_prob >= 0.55` and `long_prob < 0.5`

Confidence:

- For `HOLD`: `confidence = 1 - trade_prob`
- For `BUY/SELL`: `confidence = trade_prob * max(long_prob, 1 - long_prob)`

## 6) Output schema
The technical agent returns a `TechnicalSignal` object with:

- `signal` (`buy/sell/hold`)
- `confidence` (0..1)
- `trade_probability`
- `long_probability`
- `sequence_length`
- `feature_count`
- `reasoning`
- timestamp metadata

## 7) Strengths and limitations
### Strengths
- Uses pretrained deep sequence models on engineered market features.
- Fast inference (no training in presentation/demo flow).
- Clear two-stage decision process (trade filter + direction).

### Limitations
- Purely technical: no macro/news/social context by itself.
- Sensitive to feature schema mismatch (column count/order assumptions).
- Thresholds (`0.55`, `0.5`) are heuristic and may need calibration.

## 8) Short version:

"Agent 1 transforms recent 1-hour market features into a signal with two neural models: first it checks whether trading conditions are strong enough, then it predicts direction. This gives us a robust technical baseline that the coordinator later combines with sentiment and risk intelligence."

## 9) Quick run reference
```powershell
python coordinator_agent/main.py
```
(Agent 1 is executed internally by the coordinator.)

