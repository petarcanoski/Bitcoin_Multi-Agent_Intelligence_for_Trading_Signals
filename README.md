# Bitcoin Multi-Agent Intelligence for Trading Signals

> Four specialized agents. One coordinated decision. A transparent **BUY / SELL / HOLD** output with reasoning.

![Status](https://img.shields.io/badge/status-in%20development-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Architecture](https://img.shields.io/badge/architecture-multi--agent-purple)

## Long Story, Short

Bitcoin does not move because of one thing.
It moves because **charts**, **news**, **macro policy**, **on-chain behavior**, and **global risk** collide in real time.

So instead of betting everything on one model, this project splits intelligence into specialized agents:

- **Agent 1 - Technical Analysis** reads OHLCV + indicators and outputs direction.
- **Agent 2 - Sentiment & Macro** reads financial news, social sentiment, and macro signals.
- **Agent 3 - Risk & Volatility** tracks on-chain stress and geopolitical pressure.
- **Agent 4 - Coordinator** combines the three views into one final trading signal with explanation.

The goal is simple: **fewer blind spots, more robust decisions**.

## Why This Is Interesting

- Combines quantitative and qualitative signals in one pipeline.
- Produces explainable outputs, not just raw probabilities.
- Keeps agents modular, so each teammate can improve one part independently.
- Supports real APIs (when keys are available) with mock fallbacks for development.

## System Architecture

```text
[Agent 1: Technical]  ----\
                          \
[Agent 2: Sentiment]  ----->  [Agent 4: Coordinator]  --->  Final BUY/SELL/HOLD + reasoning
                          /
[Agent 3: Risk]       ----/
```

## Agent Overview

### Agent 1 - Technical Analysis (teammate)

- Inputs: OHLCV, indicators, patterns
- Existing assets used for integration:
  - `bitcoin-predictor-dev/src/models/direction/best_model.pt`
  - `bitcoin-predictor-dev/src/models/trade/best_model.pt`
  - `bitcoin-predictor-dev/data/processed/features_1h.parquet`
- Output: `BUY | SELL | HOLD`

### Agent 2 - Sentiment & Macro (implemented)

- Folder: `sentiment_analysis`
- Inputs: crypto/business news, social sentiment stream, macro indicators (CPI/Fed/policy)
- Default weighting:
  - News sentiment: `40%`
  - Social sentiment: `30%`
  - Macro indicators: `30%`
- Output: `BUY | SELL | HOLD` + confidence + explanation

### Agent 3 - Risk & Volatility (implemented)

- Folder: `agent_risk`
- Inputs: on-chain activity + geopolitical event pressure
- Default weighting:
  - On-chain metrics: `60%`
  - Geopolitical events: `40%`
- Output: `LOW_RISK | MEDIUM_RISK | HIGH_RISK`

### Agent 4 - Coordinator (integration layer)

- Folder: `coordinator_agent`
- Responsibility: fuse signals from Agents 1-3 and produce final decision text
- Output: final `BUY | SELL | HOLD` + rationale that can be consumed by UI/API

## Repository Structure

```text
Code/
├── sentiment_analysis/      # Agent 2
├── agent_risk/              # Agent 3
├── coordinator_agent/       # Agent 4
└── bitcoin-predictor-dev/   # Agent 1 assets and training artifacts
```

## Data Sources

Depending on configuration, agents can use:

- News APIs for headlines and market narrative
- Macro APIs for CPI/rates/policy context
- On-chain/market APIs for blockchain and volatility signals
- Mock providers for local development and testing

## Design Principles

- **Modular first**: each agent can be built and tested on its own.
- **Coordinator last**: merge independent outputs at the orchestration layer.
- **Explainability**: every signal should include human-readable reasoning.
- **Resilience**: degrade gracefully when one source is missing.

## Current Project Phase

- Agent 2: ready and running
- Agent 3: ready and running
- Agent 1: model artifacts available for integration
- Agent 4: active integration target

## Roadmap

- Unify output schema across all agents
- Add confidence calibration and signal conflict resolution in coordinator
- Add backtesting report for end-to-end multi-agent decisions
- Expose full system via API/dashboard

## Team Note

This repository is intentionally organized for collaborative development: each teammate can ship improvements in a dedicated agent folder without blocking others.

---

If you are here for the one-liner:

**This project turns fragmented Bitcoin signals into one coordinated, explainable trading decision.**
