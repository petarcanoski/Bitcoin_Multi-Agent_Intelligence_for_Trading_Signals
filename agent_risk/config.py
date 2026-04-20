# Risk & Volatility Agent Configuration

import os
from dotenv import load_dotenv

load_dotenv()


class RiskConfig:
    """Configuration for Risk & Volatility Analysis Agent"""

    # API Keys

    # Blockchain/On-chain data APIs
    GLASSNODE_API_KEY = os.getenv('GLASSNODE_API_KEY', '')
    BLOCKCHAIN_INFO_API = os.getenv('BLOCKCHAIN_INFO_API', '')

    # Geopolitical data APIs
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')  # For geopolitical events
    GDELT_API_KEY = os.getenv('GDELT_API_KEY', '')  # Global events database

    # On-Chain Analysis Configuration

    # Metrics to track
    ONCHAIN_METRICS = [
        'transaction_volume',      # Daily BTC transaction volume
        'active_addresses',        # Number of active addresses
        'exchange_inflows',        # BTC flowing into exchanges
        'exchange_outflows',       # BTC flowing out of exchanges
        'whale_movements',         # Large transactions (>100 BTC)
        'hash_rate',              # Network hash rate
        'mining_difficulty',      # Mining difficulty
        'mempool_size',           # Unconfirmed transactions
    ]

    # Whale transaction threshold (BTC)
    WHALE_THRESHOLD = 100  # Transactions > 100 BTC

    # Volume analysis (USD)
    HIGH_VOLUME_THRESHOLD = 10_000_000_000  # $10B
    LOW_VOLUME_THRESHOLD = 1_000_000_000    # $1B

    # Geopolitical Events Configuration

    # Event categories to monitor
    GEOPOLITICAL_CATEGORIES = [
        'war',
        'conflict',
        'sanctions',
        'trade_war',
        'military_action',
        'economic_crisis',
        'financial_instability',
        'currency_crisis',
        'political_instability'
    ]

    # Keywords for event detection
    GEOPOLITICAL_KEYWORDS = [
        'war', 'conflict', 'sanctions', 'military',
        'crisis', 'instability', 'tension', 'dispute',
        'attack', 'invasion', 'embargo', 'tariff',
        'recession', 'default', 'collapse'
    ]

    # Impact regions (higher weight for major economies)
    REGION_WEIGHTS = {
        'usa': 1.0,
        'china': 0.9,
        'europe': 0.8,
        'russia': 0.7,
        'middle_east': 0.6,
        'asia': 0.5,
        'other': 0.3
    }

    # Risk Scoring Configuration

    # Component weights (must sum to 1.0)
    ONCHAIN_WEIGHT = 0.60      # 60% on-chain metrics
    GEOPOLITICAL_WEIGHT = 0.40 # 40% geopolitical events

    # Risk thresholds
    HIGH_RISK_THRESHOLD = 0.7   # >0.7 = HIGH RISK
    LOW_RISK_THRESHOLD = 0.3    # <0.3 = LOW RISK

    # Volatility periods (days)
    VOLATILITY_SHORT_TERM = 7   # 7 days
    VOLATILITY_MEDIUM_TERM = 30 # 30 days
    VOLATILITY_LONG_TERM = 90   # 90 days

    # Cache & Performance

    CACHE_EXPIRY_MINUTES = 15
    REQUEST_TIMEOUT = 10  # seconds

    # Data fetch limits
    ONCHAIN_DATA_DAYS = 30  # Last 30 days of on-chain data
    GEOPOLITICAL_EVENTS_DAYS = 14  # Last 14 days of events

    # Signal Generation

    # Trading signal thresholds
    RISK_SIGNAL_THRESHOLDS = {
        'high_risk': 0.7,   # Sell or reduce position
        'medium_risk': 0.5, # Hold or cautious
        'low_risk': 0.3     # Buy or increase position
    }

    # Confidence calculation
    MIN_ONCHAIN_DATAPOINTS = 7   # Minimum data points for confidence
    MIN_GEOPOLITICAL_EVENTS = 2  # Minimum events for analysis

