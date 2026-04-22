# Multi-Agent Bitcoin Trading Intelligence System - Project Report

## 1. Executive Overview
This project implements a modular multi-agent system for Bitcoin trading support. Instead of relying on one monolithic model, the system separates intelligence into specialized agents and then fuses them through a coordinator.

Core idea:
- Agent 1 understands **market structure** from technical features.
- Agent 2 understands **narrative pressure** from news, social sentiment, and macro indicators.
- Agent 3 understands **risk regime** from on-chain and geopolitical stress.
- Agent 4 (Coordinator) combines all outputs into one final `BUY/SELL/HOLD` decision with explainable reasoning.

This architecture improves interpretability, resilience, and team parallelization.

## 2. Project Goals
The system is designed to:
1. Produce an explainable trading signal (`buy/sell/hold`).
2. Combine heterogeneous information sources (price, text, macro, blockchain, events).
3. Keep each module independent for team development.
4. Support both real-data mode and fallback/mock mode for reliable demos.

## 3. System Architecture
### 3.1 High-level flow
1. **Agent 1 (Technical)** runs inference over pretrained models using `features_1h.parquet`.
2. **Agent 2 (Sentiment+Macro)** collects text + macro context and outputs a directional sentiment signal.
3. **Agent 3 (Risk+Volatility)** produces a risk state (`low_risk/medium_risk/high_risk`).
4. **Agent 4 (Coordinator)** fuses Agent 1 and Agent 2 directional conviction, then adjusts by Agent 3 risk.
5. Final result is returned as an explainable coordinator signal.

### 3.2 Modularity principle
Each agent is intentionally isolated:
- Agent folders are separate: `coordinator_agent/`, `sentiment_analysis/`, `agent_risk/`, and technical inference wrapper in `coordinator_agent/technical_agent.py`.
- Coordinator invokes sentiment and risk through JSON subprocess bridges (`run_agent_json.py`) to avoid module-name collisions and keep dependency boundaries clean.

## 4. Agent 1 - Technical Analysis
Reference: `Presentation/01_Agent1_Technical_Analysis.md`

### 4.1 Role
Agent 1 is the directional market-structure module. It predicts whether to trade and in which direction based only on engineered technical features derived from OHLCV.

### 4.2 Inputs
From `bitcoin-predictor-dev`:
- `src/models/trade/best_model.pt`
- `src/models/direction/best_model.pt`
- `data/processed/features_1h.parquet`

### 4.3 Method
- Loads two pretrained `CNNLSTMModel` checkpoints.
- Reconstructs expected input dimension from checkpoint weights.
- Builds latest sequence window (typically 60 rows).
- Runs:
  - Trade model: is a trade justified?
  - Direction model: if yes, long or short?

### 4.4 Decision logic
- If `trade_prob < 0.55` -> `hold`
- Else if `long_prob >= 0.5` -> `buy`
- Else -> `sell`

Confidence:
- Hold case: `1 - trade_prob`
- Buy/sell case: `trade_prob * max(long_prob, 1 - long_prob)`

### 4.5 Why it matters
This agent contributes high-frequency market behavior intelligence and is weighted more heavily in final fusion because it is directly trained on market feature sequences.

## 5. Agent 2 - Sentiment + Macro Analysis
Reference: `Presentation/02_Agent2_Sentiment_Macro.md`

### 5.1 Role
Agent 2 is the NLP + macro intelligence module. It captures fundamental and narrative pressure that technical indicators alone cannot capture.

### 5.2 Inputs
- News data (NewsAPI; fallback mock when unavailable)
- Twitter/X monitored account data (currently configurable; fallback mock)
- Macro indicators (Alpha Vantage CPI attempt + simulated macro events when needed)
- Market context (CoinGecko)

### 5.3 NLP engine
- Model: `ProsusAI/finbert`
- For each text item:
  - infer class probabilities
  - score = `positive_prob - negative_prob`
  - confidence = max class probability
- If model fails, fallback keyword sentiment is used.

### 5.4 Time relevance and aggregation
Each item is time-decayed:
- `relevance = decay_factor * confidence`
- more recent items have higher impact

Weighted sentiment average:
- `avg_sentiment = sum(sentiment * relevance) / sum(relevance)`

### 5.5 Macro scoring
Macro impacts map to numeric values:
- positive -> `+1`
- neutral -> `0`
- negative -> `-1`

Aggregate macro score is weighted by indicator importance.

### 5.6 Final Agent 2 signal
`combined_score = 0.40*news + 0.30*twitter + 0.30*macro`

- `> 0.3` -> `buy`
- `< -0.3` -> `sell`
- otherwise -> `hold`

### 5.7 Why it matters
Agent 2 injects policy/news/social context (CPI, Fed tone, trade-war/tariff sentiment, market narrative), improving robustness against non-technical shocks.

## 6. Agent 3 - Risk + Volatility Analysis
Reference: `Presentation/03_Agent3_Risk_Volatility.md`

### 6.1 Role
Agent 3 does not predict direction directly. It evaluates regime risk and market instability so the system can avoid overconfident trades in dangerous environments.

### 6.2 Inputs
- On-chain metrics (volume, active addresses, exchange flows, whale activity, hash rate, mempool, etc.)
- Geopolitical events (wars, sanctions, political/economic instability)

### 6.3 Scoring
`combined_risk = 0.60*onchain_risk + 0.40*geopolitical_risk`

Thresholds:
- `< 0.3` -> `low_risk`
- `> 0.7` -> `high_risk`
- else -> `medium_risk`

A volatility sub-score is also computed (supporting diagnostic signal).

### 6.4 Confidence
Confidence increases with data availability:
- on-chain data coverage
- geopolitical event coverage
- final confidence is averaged across both components.

### 6.5 Why it matters
Agent 3 acts as a safety layer for final decision control. It can suppress aggressive bullish/bearish actions when the environment is unstable.

## 7. Agent 4 - Coordinator and Final Decision
Reference: `Presentation/04_Agent4_Coordinator.md`

### 7.1 Role
Coordinator merges all agent outputs into one actionable, explainable decision.

### 7.2 Integration approach
- Agent 1: direct class call (`TechnicalAgent.run()`)
- Agent 2 and Agent 3: subprocess JSON scripts (`run_agent_json.py`)

### 7.3 Fusion pipeline
1. Convert directional signals to scores:
   - `buy=+1`, `hold=0`, `sell=-1`
2. Multiply by each agent confidence.
3. Directional fusion:
   - `combined_score = 0.60*tech_score + 0.40*sentiment_score`
4. Risk adjustment multiplier:
   - `low_risk=1.10`, `medium_risk=1.00`, `high_risk=0.65`
5. Clamp adjusted score to `[-1, +1]`.
6. Decision thresholds:
   - high risk and strong buy -> can be forced to `hold` (conservative gate)
   - `>= 0.25` -> `buy`
   - `<= -0.25` -> `sell`
   - otherwise `hold`
7. Final confidence:
   - `abs(adjusted_score) + 0.15*risk_confidence` (capped at 1.0)

### 7.4 Explainability
Coordinator output includes:
- final signal
- confidence
- risk level
- key factors from all agents
- combined reasoning paragraph with each agent's rationale

## 8. How the Final Result Is Obtained (End-to-End)
1. Raw inputs are collected in each specialized domain.
2. Each agent transforms raw data into normalized scores and confidence values.
3. Coordinator performs weighted directional fusion (technical + sentiment).
4. Risk regime modifies directional conviction.
5. Threshold logic maps numeric conviction to discrete signal (`buy/sell/hold`).
6. Human-readable reasoning is generated from component explanations.

In short:
- Agent 1 and Agent 2 propose direction.
- Agent 3 controls aggression level.
- Agent 4 decides final action.

## 9. What Affects the Result Most
### 9.1 Primary drivers
- Technical model probabilities (`trade_prob`, `long_prob`)
- Sentiment channel balance (news/social/macro)
- Risk multiplier from Agent 3

### 9.2 Sensitivity points
- Thresholds (`0.55`, `0.5`, `+-0.3`, `+-0.25`, risk thresholds)
- API data quality and recency
- Feature schema compatibility for Agent 1
- Event labeling quality for geopolitical scoring

### 9.3 Inter-agent influence examples
- If Agent 1 says `buy` with high confidence and Agent 2 says `hold`, final score may still be buy-leaning due to technical 60% weight.
- If both Agent 1 and Agent 2 are bullish but Agent 3 outputs `high_risk`, coordinator dampens confidence and may downgrade to `hold`.
- If sentiment turns strongly negative while technical is weak-positive, final output can move to neutral/hold or sell depending on confidence values.

## 10. Real Data vs Mock Data Behavior
The system is designed for graceful degradation:
- Missing API keys do not crash the pipeline.
- Agents return fallback/mock-based outputs so coordinator can still run.
- This is useful for demos, development, and presentation stability.

Practical interpretation:
- Real API mode gives better realism.
- Mock mode gives reproducibility and reliability when external services fail.

## 11. Reliability, Error Handling, and Safety Defaults
- Agent-level try/except blocks return safe defaults (`hold` or `medium_risk`) on failure.
- Coordinator validates subprocess return code and JSON payload extraction.
- Fallback behavior prevents total system failure when one source is down.

This is important for production-style robustness and for classroom demonstrations.

## 12. Current Limitations
1. Fusion weights are heuristic and not yet fully calibrated by large-scale backtesting.
2. External APIs can be rate-limited and sometimes paid on advanced tiers.
3. Social/geopolitical text can include noise and interpretation bias.
4. Agent 1 checkpoint expectations require strict feature compatibility.

## 13. Validation and Evaluation Strategy
Current validation style:
- Unit-level/agent-level execution checks
- Smoke test flow through coordinator
- Reasoning inspection for explainability consistency

Recommended next evaluation layer:
1. Historical backtest with synchronized snapshots from all agents.
2. Ablation studies (remove one agent at a time).
3. Threshold and weight calibration on validation windows.
4. Regime-specific analysis (high-volatility vs calm periods).

## 14. Why Multi-Agent Was a Good Design Choice
- Better separation of concerns than one giant model.
- Easier team collaboration (parallel development per agent).
- Easier debugging (identify which agent caused drift).
- Better explainability for academic evaluation.
- Better extensibility for future data sources or model upgrades.

## 15. Conclusion
This project demonstrates a practical, explainable, and modular multi-agent architecture for Bitcoin trading decision support.

Final decision quality emerges from **complementarity**:
- Agent 1 contributes quantitative market structure,
- Agent 2 contributes NLP and macro narrative context,
- Agent 3 contributes regime-risk control,
- Agent 4 integrates all three into a final actionable signal.

The result is a system that is more robust and interpretable than any single-source approach, while remaining flexible for future upgrades and team integration.

---

## Appendix A - Key Project Paths
- `bitcoin-predictor-dev/src/models/direction/best_model.pt`
- `bitcoin-predictor-dev/src/models/trade/best_model.pt`
- `bitcoin-predictor-dev/data/processed/features_1h.parquet`
- `sentiment_analysis/sentiment_agent.py`
- `sentiment_analysis/sentiment_analyzer.py`
- `agent_risk/risk_agent.py`
- `coordinator_agent/technical_agent.py`
- `coordinator_agent/coordinator_agent.py`

## Appendix B - Related Presentation Files
- `Presentation/01_Agent1_Technical_Analysis.md`
- `Presentation/02_Agent2_Sentiment_Macro.md`
- `Presentation/03_Agent3_Risk_Volatility.md`
- `Presentation/04_Agent4_Coordinator.md`

