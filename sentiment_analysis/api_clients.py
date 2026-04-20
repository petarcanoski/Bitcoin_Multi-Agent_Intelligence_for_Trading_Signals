import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests_cache
from config import Config
from models import NewsItem, MacroIndicator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

requests_cache.install_cache(
    'sentiment_agent_cache',
    expire_after=Config.CACHE_EXPIRY_MINUTES * 60
)


class TwitterClient:
    """Client for fetching tweets from Twitter/X using API v2"""

    BASE_URL = "https://api.twitter.com/2"

    def __init__(self, bearer_token: str = None):
        self.bearer_token = bearer_token or Config.TWITTER_BEARER_TOKEN
        if not self.bearer_token:
            logger.warning("Twitter Bearer Token not found. Using mock Twitter data.")

    def fetch_tweets_from_account(self, username: str, days_back: int = 3) -> List[NewsItem]:
        """Fetch recent tweets from a specific account (e.g., @WatcherGuru)"""
        if not self.bearer_token:
            return self._get_mock_tweets(username)

        try:
            # First, get user ID from username
            user_id = self._get_user_id(username)
            if not user_id:
                return self._get_mock_tweets(username)

            # Calculate time range
            start_time = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"

            # Fetch tweets
            url = f"{self.BASE_URL}/users/{user_id}/tweets"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            params = {
                'max_results': min(Config.TWITTER_FETCH_LIMIT, 100),
                'tweet.fields': 'created_at,public_metrics,entities',
                'start_time': start_time,
                'exclude': 'retweets,replies'
            }

            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            tweets = []
            for tweet in data.get('data', []):
                try:
                    # Filter by keywords if needed
                    text = tweet.get('text', '')
                    if any(keyword.lower() in text.lower() for keyword in Config.TWITTER_KEYWORDS):
                        tweets.append(NewsItem(
                            title=text[:100] + ('...' if len(text) > 100 else ''),
                            description=text,
                            source=f"Twitter/@{username}",
                            published_at=datetime.fromisoformat(
                                tweet.get('created_at', '').replace('Z', '+00:00')
                            ),
                            url=f"https://twitter.com/{username}/status/{tweet.get('id')}"
                        ))
                except Exception as e:
                    logger.debug(f"Error parsing tweet: {e}")
                    continue

            logger.info(f"Fetched {len(tweets)} tweets from @{username}")
            return tweets

        except Exception as e:
            logger.error(f"Error fetching tweets from @{username}: {e}")
            return self._get_mock_tweets(username)

    def _get_user_id(self, username: str) -> Optional[str]:
        """Get user ID from username"""
        try:
            url = f"{self.BASE_URL}/users/by/username/{username}"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('data', {}).get('id')
        except Exception as e:
            logger.error(f"Error getting user ID for @{username}: {e}")
            return None

    def fetch_all_monitored_accounts(self, days_back: int = 3) -> List[NewsItem]:
        """Fetch tweets from all monitored accounts"""
        all_tweets = []
        for account in Config.TWITTER_ACCOUNTS:
            tweets = self.fetch_tweets_from_account(account, days_back)
            all_tweets.extend(tweets)
        return all_tweets

    def _get_mock_tweets(self, username: str = "WatcherGuru") -> List[NewsItem]:
        """Return mock tweets for testing when API is unavailable"""
        logger.info(f"Using mock Twitter data for @{username}")
        return [
            NewsItem(
                title="BREAKING: Bitcoin surges past $70,000 as institutional demand reaches new highs",
                description="BREAKING: Bitcoin surges past $70,000 as institutional demand reaches new highs. Major ETF inflows continue.",
                source=f"Twitter/@{username}",
                published_at=datetime.now() - timedelta(hours=1),
                url=f"https://twitter.com/{username}/status/mock1"
            ),
            NewsItem(
                title="Federal Reserve signals potential rate cuts in Q3 2026, crypto markets rally",
                description="Federal Reserve signals potential rate cuts in Q3 2026. Crypto markets rally on dovish sentiment. #Bitcoin #Fed",
                source=f"Twitter/@{username}",
                published_at=datetime.now() - timedelta(hours=4),
                url=f"https://twitter.com/{username}/status/mock2"
            ),
            NewsItem(
                title="⚠️ US CPI data shows inflation cooling to 2.8%, positive for risk assets including Bitcoin",
                description="US CPI data shows inflation cooling to 2.8%, lowest in 3 years. Positive signal for risk assets including Bitcoin.",
                source=f"Twitter/@{username}",
                published_at=datetime.now() - timedelta(hours=8),
                url=f"https://twitter.com/{username}/status/mock3"
            ),
            NewsItem(
                title="Bitcoin whale accumulation continues, on-chain data shows strong HODLer conviction",
                description="Bitcoin whale accumulation continues. On-chain data shows strong HODLer conviction despite short-term volatility.",
                source=f"Twitter/@{username}",
                published_at=datetime.now() - timedelta(hours=12),
                url=f"https://twitter.com/{username}/status/mock4"
            )
        ]


class NewsAPIClient:
    """Client for fetching news from NewsAPI"""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.NEWSAPI_KEY
        if not self.api_key:
            logger.warning("NewsAPI key not found. News fetching will be limited.")

    def fetch_crypto_news(self, days_back: int = 3) -> List[NewsItem]:
        """Fetch cryptocurrency and Bitcoin related news"""
        if not self.api_key:
            return self._get_mock_news()

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        news_items = []

        try:
            for query in Config.NEWS_QUERY_TERMS:
                url = f"{self.BASE_URL}/everything"
                params = {
                    'q': query,
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': Config.NEWS_FETCH_LIMIT,
                    'apiKey': self.api_key
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                for article in data.get('articles', []):
                    try:
                        news_items.append(NewsItem(
                            title=article.get('title', ''),
                            description=article.get('description', ''),
                            source=article.get('source', {}).get('name', 'Unknown'),
                            published_at=datetime.fromisoformat(
                                article.get('publishedAt', '').replace('Z', '+00:00')
                            ),
                            url=article.get('url')
                        ))
                    except Exception as e:
                        logger.debug(f"Error parsing article: {e}")
                        continue

            logger.info(f"Fetched {len(news_items)} news articles")
            return news_items[:Config.NEWS_FETCH_LIMIT]

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return self._get_mock_news()

    def _get_mock_news(self) -> List[NewsItem]:
        """Return mock news for testing when API is unavailable"""
        logger.info("Using mock news data")
        return [
            NewsItem(
                title="Bitcoin Shows Strong Momentum Amid Institutional Adoption",
                description="Major institutions continue to increase Bitcoin holdings",
                source="Mock Source",
                published_at=datetime.now() - timedelta(hours=2)
            ),
            NewsItem(
                title="Federal Reserve Maintains Interest Rates, Crypto Markets React",
                description="Fed decision impacts cryptocurrency market sentiment",
                source="Mock Source",
                published_at=datetime.now() - timedelta(hours=5)
            ),
            NewsItem(
                title="Bitcoin Volatility Increases Amid Geopolitical Tensions",
                description="Market uncertainty drives crypto price fluctuations",
                source="Mock Source",
                published_at=datetime.now() - timedelta(hours=12)
            )
        ]


class MacroDataClient:
    """Client for fetching macroeconomic data"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.ALPHA_VANTAGE_KEY
        if not self.api_key:
            logger.warning("Alpha Vantage key not found. Using mock macro data.")

    def fetch_macro_indicators(self) -> List[MacroIndicator]:
        """Fetch relevant macroeconomic indicators"""
        indicators = []

        # Try to fetch real CPI data
        if self.api_key:
            try:
                cpi_data = self._fetch_cpi_data()
                if cpi_data:
                    indicators.append(cpi_data)
            except Exception as e:
                logger.error(f"Error fetching CPI data: {e}")

        # Add mock/simulated macro indicators
        indicators.extend(self._get_mock_macro_indicators())

        logger.info(f"Fetched {len(indicators)} macro indicators")
        return indicators

    def _fetch_cpi_data(self) -> Optional[MacroIndicator]:
        """Fetch real CPI data from Alpha Vantage or FRED"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'CPI',
                'interval': 'monthly',
                'apikey': self.api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'data' in data and len(data['data']) > 0:
                latest = data['data'][0]
                cpi_value = float(latest.get('value', 0))

                # Determine impact based on CPI trend
                impact = 'negative' if cpi_value > 3.0 else 'neutral' if cpi_value > 2.0 else 'positive'

                return MacroIndicator(
                    indicator_type='cpi_inflation',
                    value=cpi_value,
                    description=f"US CPI at {cpi_value}% - inflation indicator",
                    impact=impact,
                    weight=Config.MACRO_WEIGHTS['cpi_inflation'],
                    published_at=datetime.fromisoformat(latest.get('date', datetime.now().isoformat()))
                )
        except Exception as e:
            logger.debug(f"Could not fetch real CPI data: {e}")

        return None

    def _get_mock_macro_indicators(self) -> List[MacroIndicator]:
        """Return mock macro indicators for testing"""
        logger.info("Using mock macro indicators")

        return [
            MacroIndicator(
                indicator_type='fed_decisions',
                value=5.25,
                description="Federal Reserve maintains interest rate at 5.25%",
                impact='negative',  # High rates = negative for risk assets
                weight=Config.MACRO_WEIGHTS['fed_decisions'],
                published_at=datetime.now() - timedelta(days=2)
            ),
            MacroIndicator(
                indicator_type='cpi_inflation',
                value=3.2,
                description="US CPI inflation at 3.2%, showing cooling trend",
                impact='neutral',
                weight=Config.MACRO_WEIGHTS['cpi_inflation'],
                published_at=datetime.now() - timedelta(days=7)
            ),
            MacroIndicator(
                indicator_type='tariffs',
                value=None,
                description="New tariff announcements create market uncertainty",
                impact='negative',
                weight=Config.MACRO_WEIGHTS['tariffs'],
                published_at=datetime.now() - timedelta(days=1)
            )
        ]


class CryptoMarketClient:
    """Client for cryptocurrency market context"""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def fetch_btc_market_context(self) -> Dict:
        """Fetch Bitcoin market context from CoinGecko (free API)"""
        try:
            url = f"{self.BASE_URL}/coins/bitcoin"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'community_data': 'true',
                'developer_data': 'false'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            market_data = data.get('market_data', {})
            sentiment_data = data.get('sentiment_votes_up_percentage', 50)

            context = {
                'current_price': market_data.get('current_price', {}).get('usd', 0),
                'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                'market_cap_rank': data.get('market_cap_rank', 1),
                'sentiment_up_percentage': sentiment_data
            }

            logger.info(f"Fetched BTC market context: ${context['current_price']}")
            return context

        except Exception as e:
            logger.error(f"Error fetching BTC market context: {e}")
            return {
                'current_price': 0,
                'price_change_24h': 0,
                'price_change_7d': 0,
                'market_cap_rank': 1,
                'sentiment_up_percentage': 50
            }

