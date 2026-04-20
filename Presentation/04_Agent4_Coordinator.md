# Agent 4 - Coordinator (Fusion of Agents 1, 2, and 3)

## 1) What this agent does
Agent 4 is the decision fusion layer. It does not fetch raw market data directly. Instead, it receives outputs from:

- Agent 1: technical signal (`BUY/SELL/HOLD`)
- Agent 2: sentiment+macro signal (`BUY/SELL/HOLD`)
- Agent 3: risk state (`LOW/MEDIUM/HIGH_RISK`)

Then it produces one final trading decision with confidence and full textual reasoning.

Implementation folder:

- `coordinator_agent/`

Main file:

- `coordinator_agent/coordinator_agent.py`

## 2) Why subprocess JSON bridges are used
Agent 2 and Agent 3 both have generic module names (`config.py`, `models.py`).
To avoid import collisions, the coordinator calls them as subprocess scripts:

- `sentiment_analysis/run_agent_json.py`
- `agent_risk/run_agent_json.py`

This keeps all agents independent and easy to develop in parallel.

## 3) Technical agent integration
Agent 1 is integrated directly through Python class call:

- `TechnicalAgent(repo_root).run()`

It loads pretrained models and latest features from:

- `bitcoin-predictor-dev/src/models/trade/best_model.pt`
- `bitcoin-predictor-dev/src/models/direction/best_model.pt`
- `bitcoin-predictor-dev/data/processed/features_1h.parquet`

## 4) Fusion math
### Step A: Convert class signal to numeric score
Mapping:

- `buy -> +1`
- `hold -> 0`
- `sell -> -1`

Then multiply by each agent confidence:

- `tech_score = signal_to_score(tech_signal) * tech_confidence`
- `sent_score = signal_to_score(sent_signal) * sent_confidence`

### Step B: Weighted directional fusion
`combined_score = 0.60 * tech_score + 0.40 * sent_score`

Technical is weighted higher because it comes from pretrained OHLCV-based models.

### Step C: Risk adjustment
Risk multiplier:

- `low_risk -> 1.10`
- `medium_risk -> 1.00`
- `high_risk -> 0.65`

`adjusted_score = clamp(combined_score * risk_multiplier, -1, +1)`

### Step D: Final decision thresholds
- if `high_risk` and `adjusted_score > 0.55` -> force `HOLD` (conservative gate)
- else if `adjusted_score >= 0.25` -> `BUY`
- else if `adjusted_score <= -0.25` -> `SELL`
- else -> `HOLD`

### Step E: Final confidence
`confidence = min(1.0, abs(adjusted_score) + 0.15 * risk_confidence)`

## 5) Output schema
`FinalCoordinatorSignal` contains:

- `signal`
- `confidence`
- `score`
- `risk_level`
- `key_factors`
- `reasoning`
- timestamp and `data_sources`

The key explainability benefit: reasoning includes snippets from all three agents.

## 6) End-to-end execution flow
1. Run technical inference in-process.
2. Run sentiment agent JSON script and parse output.
3. Run risk agent JSON script and parse output.
4. Combine scores with weights and risk multipliers.
5. Return final explainable decision.

## 7) Why this architecture is good for team projects
- Each teammate can ship one agent independently.
- Agents remain modular and testable.
- Coordinator provides one single interface to downstream users.
- Easy to swap a single agent without rewriting the rest.

## 8) Known limitations
- Fusion weights are currently heuristic and may need validation.
- Subprocess calls add small runtime overhead.
- If one upstream agent fails, coordinator needs graceful fallback policy.

## 9) Short version:

"Agent 4 is our orchestration brain. It takes directional intelligence from technical and sentiment agents, then applies a risk gate from Agent 3. This gives one final trading signal that is both quantitative and explainable."

## 10) Quick run reference
```powershell
python coordinator_agent/main.py
```

Smoke test:

```powershell
python coordinator_agent/smoke_test.py
```

