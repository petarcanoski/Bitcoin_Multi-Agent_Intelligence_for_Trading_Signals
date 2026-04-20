# Agent 3 - Risk and Volatility (On-Chain + Geopolitics)

## 1) What this agent does
Agent 3 evaluates market danger level, not direct price direction.

Output is a risk state:

- `LOW_RISK`
- `MEDIUM_RISK`
- `HIGH_RISK`

This risk layer is later used by the coordinator to amplify or suppress final trading conviction.

Implementation folder:

- `agent_risk/`

Core files:

- `agent_risk/risk_agent.py`
- `agent_risk/onchain_clients.py`
- `agent_risk/geopolitical_clients.py`
- `agent_risk/config.py`
- `agent_risk/models.py`

## 2) Inputs and analysis domains
### On-chain domain (60%)
Tracked metrics include:

- transaction volume
- active addresses
- exchange inflows/outflows
- whale movements
- hash rate
- mining difficulty
- mempool size

### Geopolitical domain (40%)
Monitors event classes such as:

- war / conflict
- sanctions
- trade war / tariffs
- economic or financial instability
- political instability

## 3) Scoring framework
From `agent_risk/config.py`:

- `ONCHAIN_WEIGHT = 0.60`
- `GEOPOLITICAL_WEIGHT = 0.40`

Combined risk score:

- `combined_risk = 0.60 * onchain_risk + 0.40 * geopolitical_risk`

Thresholds:

- `< 0.3` -> `low_risk`
- `> 0.7` -> `high_risk`
- otherwise -> `medium_risk`

## 4) Volatility sub-score
Agent 3 also computes a volatility score (supporting signal) from on-chain behavior:

- high/extreme transaction-volume risk increases volatility estimate
- high mempool congestion increases volatility estimate

The volatility score is reported in output and used in reasoning, even though final classification is based on combined risk thresholds.

## 5) Confidence logic
Confidence is data-availability-based:

- on-chain confidence increases with number of on-chain datapoints
- geopolitical confidence increases with number of geopolitical events
- final confidence is the average of the two

Config defaults:

- `MIN_ONCHAIN_DATAPOINTS = 7`
- `MIN_GEOPOLITICAL_EVENTS = 2`

## 6) Explainability layer
Agent 3 outputs:

- `key_risks` (top risk drivers)
- `key_opportunities` (favorable factors)
- `reasoning` paragraph with on-chain/geopolitical/volatility interpretation

This is important for presentation because it shows the model is not a black box.

## 7) Output schema
`RiskSignalOutput` contains:

- `signal`, `confidence`, `risk_score`
- `onchain_risk`, `geopolitical_risk`, `volatility_score`
- `onchain_metrics`, `geopolitical_events`
- `key_risks`, `key_opportunities`
- `reasoning`, timestamp, `data_sources`

Coordinator bridge script:

- `agent_risk/run_agent_json.py`

## 8) Why this agent is useful
- Prevents overconfident trading during unstable environments.
- Adds context that technical indicators alone cannot capture.
- Creates a robust risk gate for final decision-making.

## 9) Limitations and practical notes
- Geopolitical labeling can be noisy (event interpretation bias).
- Some on-chain APIs can be rate-limited or paid on higher tiers.
- Risk score is not direction by itself; it must be fused with technical/sentiment outputs.

## 10) Short version:

"Agent 3 is our safety module. It combines blockchain risk signals and geopolitical stress signals into a single risk score. Instead of predicting direction, it tells us how dangerous the environment is, which helps the coordinator avoid aggressive trades in unstable periods."

## 11) Quick run reference
```powershell
python agent_risk/main.py
```

For coordinator JSON bridge:

```powershell
python agent_risk/run_agent_json.py
```

