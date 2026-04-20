import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration for Sentiment Analysis Agent"""

    # API Keys
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

    # Twitter/X API Keys (for real-time social sentiment)
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
    TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET', '')

    # News Sources Configuration
    NEWS_FETCH_LIMIT = int(os.getenv('NEWS_FETCH_LIMIT', 50))
    NEWS_SOURCES = ['coindesk', 'crypto-coins-news', 'the-wall-street-journal']
    NEWS_QUERY_TERMS = ['bitcoin', 'BTC', 'cryptocurrency', 'crypto market']

    # Twitter/X Configuration
    TWITTER_ACCOUNTS = ['WatcherGuru']  # Accounts to monitor
    TWITTER_FETCH_LIMIT = int(os.getenv('TWITTER_FETCH_LIMIT', 30))
    TWITTER_KEYWORDS = ['bitcoin', 'btc', 'crypto', 'fed', 'inflation', 'cpi', 'tariff']

    # Cache Configuration
    CACHE_EXPIRY_MINUTES = int(os.getenv('CACHE_EXPIRY_MINUTES', 15))

    # Sentiment Thresholds
    SENTIMENT_THRESHOLD_POSITIVE = float(os.getenv('SENTIMENT_THRESHOLD_POSITIVE', 0.6))
    SENTIMENT_THRESHOLD_NEGATIVE = float(os.getenv('SENTIMENT_THRESHOLD_NEGATIVE', -0.6))

    # Macro Event Weights (must sum to 1.0)
    MACRO_WEIGHTS = {
        'fed_decisions': 0.40,
        'cpi_inflation': 0.30,
        'tariffs': 0.20,
        'other': 0.10
    }

    # Signal Generation Weights
    NEWS_SENTIMENT_WEIGHT = 0.40  # 40% weight for news sentiment
    TWITTER_SENTIMENT_WEIGHT = 0.30  # 30% weight for Twitter/X sentiment
    MACRO_ANALYSIS_WEIGHT = 0.30  # 30% weight for macro analysis

    # Model Configuration
    SENTIMENT_MODEL = 'ProsusAI/finbert'  # FinBERT for financial sentiment

    # Time decay for news relevance (hours)
    NEWS_RELEVANCE_DECAY = 24  # News older than 24 hours gets less weight

