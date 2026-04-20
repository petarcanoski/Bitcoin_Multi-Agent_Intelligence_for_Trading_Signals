# Agent 2 - Sentiment Analysis + Macro News

## 1) What this agent does
Agent 2 measures market mood and macro pressure, then converts that into a trading signal (`BUY/SELL/HOLD`).

It combines three information channels:

- **News sentiment** (40%)
- **Twitter/X sentiment** (30%)
- **Macro indicators** (30%)

Implementation folder:

- `sentiment_analysis/`

Main logic lives in:

- `sentiment_analysis/sentiment_agent.py`
- `sentiment_analysis/sentiment_analyzer.py`
- `sentiment_analysis/api_clients.py`
- `sentiment_analysis/config.py`

## 2) Data sources and fallback strategy
### News
- Source: NewsAPI (`/v2/everything`), query terms like `bitcoin`, `BTC`, `cryptocurrency`.
- If API key is missing or request fails -> use mock news.

### Twitter/X
- Source: X API v2 via bearer token.
- Monitored accounts configured in `Config.TWITTER_ACCOUNTS` (currently includes `WatcherGuru`).
- If bearer token is missing/fails -> use mock tweets.

### Macro
- Real CPI attempt from Alpha Vantage (`function=CPI`).
- Additional macro indicators (Fed, tariffs) are currently simulated in mock mode.
- If Alpha Vantage key missing/fails, macro analysis still runs with mock indicators.

### Market context
- CoinGecko is used for BTC context (`price_change_24h`, community sentiment, etc.).

## 3) NLP engine and sentiment scoring
The NLP model is FinBERT:

- Model: `ProsusAI/finbert`
- Loaded via Hugging Face transformers.

For each text:

1. Tokenize text (`title + description`).
2. Run model inference.
3. Convert probabilities to score using:
   - `sentiment_score = positive_prob - negative_prob` (range about -1 to +1)
4. Confidence = highest class probability.

If model loading/inference fails, fallback uses keyword-based sentiment.

## 4) Time decay and relevance weighting
Recent articles/tweets are weighted more heavily.

For each item:

- `hours_old = now - published_at`
- `decay_factor = max(0, 1 - hours_old / NEWS_RELEVANCE_DECAY)`
- `relevance_score = decay_factor * item_confidence`

Then aggregate with weighted average:

- `avg_sentiment = sum(sentiment * relevance) / sum(relevance)`

This avoids old news dominating the signal.

## 5) Macro analysis scoring
Each macro indicator has:

- `impact` in `{positive, neutral, negative}`
- `weight` from config (`MACRO_WEIGHTS`)

Mapping:

- positive -> `+1`
- neutral -> `0`
- negative -> `-1`

Weighted macro score:

- `macro_score = sum(impact_score * weight) / sum(weights)`

## 6) Final signal generation
From `sentiment_analysis/sentiment_agent.py`:

- `combined_score = 0.40*news + 0.30*twitter + 0.30*macro`

Decision thresholds:

- `combined_score > 0.3` -> `BUY`
- `combined_score < -0.3` -> `SELL`
- otherwise -> `HOLD`

Confidence:

- Average of channel confidences (news, twitter, macro).

## 7) Output schema
Returns `SignalOutput` with:

- `signal`, `confidence`, `sentiment_score`
- `news_sentiment`, `twitter_sentiment`, `macro_score`
- `key_factors`
- `reasoning` (human-readable explanation)
- timestamp and `data_sources`

This output is exported to coordinator through:

- `sentiment_analysis/run_agent_json.py`

## 8) Why this agent is useful
- Captures information that technical charts miss (policy decisions, inflation, narrative shocks).
- Produces explainable text reasons, not only a class label.
- Works in degraded mode with mock data, so demos do not break.

## 9) Common pitfalls to mention in presentation
- External API quality/rate limits affect real-time quality.
- Social data can be noisy and biased.
- Keyword filtering on X may miss relevant posts or include false positives.
- Real macro calendar updates are lower frequency than market price moves.

## 10) Short version:

"Agent 2 is our NLP and macro intelligence module. It reads financial news and social text with FinBERT, combines that with macro indicators like CPI/Fed/tariff signals, and outputs an explainable BUY/SELL/HOLD recommendation. It supports real APIs and graceful fallback mocks for robust operation."

## 11) Quick run reference
```powershell
python sentiment_analysis/main.py
```

For coordinator JSON bridge:

```powershell
python sentiment_analysis/run_agent_json.py
```
