from typing import Dict, List
from datetime import datetime
import logging

from config import Config
from models import SignalOutput, NewsItem, MacroIndicator
from api_clients import NewsAPIClient, MacroDataClient, CryptoMarketClient, TwitterClient
from sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAgent:
    """
    Sentiment Analysis Agent for Bitcoin Trading

    This agent:
    1. Fetches news from various sources
    2. Analyzes sentiment using FinBERT
    3. Gathers macroeconomic indicators (CPI, Fed decisions, tariffs)
    4. Generates buy/sell/hold signals with confidence scores
    5. Provides reasoning for the coordinator agent
    """

    def __init__(self):
        self.news_client = NewsAPIClient()
        self.twitter_client = TwitterClient()
        self.macro_client = MacroDataClient()
        self.market_client = CryptoMarketClient()
        self.sentiment_analyzer = SentimentAnalyzer()

        logger.info("Sentiment Agent initialized")

    def fetch_news_data(self, days_back: int = 3) -> List[NewsItem]:
        """Fetch recent Bitcoin and crypto news"""
        logger.info(f"Fetching news from last {days_back} days")
        return self.news_client.fetch_crypto_news(days_back=days_back)

    def fetch_twitter_data(self, days_back: int = 3) -> List[NewsItem]:
        """Fetch recent tweets from monitored accounts (e.g., @WatcherGuru)"""
        logger.info(f"Fetching Twitter data from last {days_back} days")
        return self.twitter_client.fetch_all_monitored_accounts(days_back=days_back)

    def fetch_macro_data(self) -> List[MacroIndicator]:
        """Fetch macroeconomic indicators"""
        logger.info("Fetching macro indicators")
        return self.macro_client.fetch_macro_indicators()

    def analyze_sentiment(self, news_items: List[NewsItem]) -> Dict:
        """Analyze sentiment of news articles"""
        logger.info("Analyzing news sentiment")
        news_with_sentiment = self.sentiment_analyzer.analyze_news_batch(news_items)
        return self.sentiment_analyzer.aggregate_news_sentiment(news_with_sentiment)

    def analyze_macro(self, indicators: List[MacroIndicator]) -> Dict:
        """Analyze macroeconomic indicators"""
        logger.info("Analyzing macro indicators")
        return self.sentiment_analyzer.analyze_macro_indicators(indicators)

    def generate_signal(self,
                       news_sentiment: Dict,
                       twitter_sentiment: Dict,
                       macro_analysis: Dict,
                       market_context: Dict = None) -> SignalOutput:
        """
        Generate final trading signal based on sentiment and macro analysis

        Signal logic:
        - Combined score > 0.3: BUY
        - Combined score < -0.3: SELL
        - Otherwise: HOLD
        """
        # Calculate weighted combined score
        news_score = news_sentiment.get('average_sentiment', 0.0)
        twitter_score = twitter_sentiment.get('average_sentiment', 0.0)
        macro_score = macro_analysis.get('macro_score', 0.0)

        combined_score = (
            news_score * Config.NEWS_SENTIMENT_WEIGHT +
            twitter_score * Config.TWITTER_SENTIMENT_WEIGHT +
            macro_score * Config.MACRO_ANALYSIS_WEIGHT
        )

        # Calculate confidence (average of component confidences)
        news_conf = news_sentiment.get('confidence', 0.0)
        twitter_conf = twitter_sentiment.get('confidence', 0.0)
        macro_conf = macro_analysis.get('confidence', 0.0)
        overall_confidence = (news_conf + twitter_conf + macro_conf) / 3

        # Determine signal
        if combined_score > 0.3:
            signal = 'buy'
        elif combined_score < -0.3:
            signal = 'sell'
        else:
            signal = 'hold'

        # Build reasoning
        reasoning = self._generate_reasoning(
            signal, news_score, twitter_score, macro_score, combined_score,
            news_sentiment.get('count', 0),
            twitter_sentiment.get('count', 0),
            macro_analysis.get('count', 0),
            market_context
        )

        # Identify key factors
        key_factors = self._identify_key_factors(news_score, twitter_score, macro_score, market_context)

        # Create signal output
        signal_output = SignalOutput(
            agent_type='sentiment_analysis',
            signal=signal,
            confidence=overall_confidence,
            sentiment_score=combined_score,
            news_sentiment=news_score,
            twitter_sentiment=twitter_score,
            macro_score=macro_score,
            key_factors=key_factors,
            reasoning=reasoning,
            timestamp=datetime.now(),
            data_sources=['NewsAPI', 'Twitter/X', 'AlphaVantage', 'CoinGecko']
        )

        logger.info(f"Generated signal: {signal.upper()} (confidence: {overall_confidence:.2f})")
        return signal_output

    def _generate_reasoning(self,
                          signal: str,
                          news_score: float,
                          twitter_score: float,
                          macro_score: float,
                          combined_score: float,
                          news_count: int,
                          twitter_count: int,
                          macro_count: int,
                          market_context: Dict = None) -> str:
        """Generate human-readable reasoning for the signal"""

        # Signal interpretation
        signal_text = {
            'buy': 'BUY signal',
            'sell': 'SELL signal',
            'hold': 'HOLD signal'
        }

        reasoning_parts = [
            f"{signal_text[signal]} generated with combined sentiment score of {combined_score:.2f}."
        ]

        # News sentiment component
        if news_count > 0:
            news_sentiment_text = "positive" if news_score > 0.1 else "negative" if news_score < -0.1 else "neutral"
            reasoning_parts.append(
                f"News sentiment ({news_count} articles analyzed) is {news_sentiment_text} "
                f"with score {news_score:.2f}."
            )
        else:
            reasoning_parts.append("Insufficient news data available.")

        # Twitter sentiment component
        if twitter_count > 0:
            twitter_sentiment_text = "positive" if twitter_score > 0.1 else "negative" if twitter_score < -0.1 else "neutral"
            reasoning_parts.append(
                f"Twitter/X sentiment ({twitter_count} tweets analyzed) is {twitter_sentiment_text} "
                f"with score {twitter_score:.2f}."
            )
        else:
            reasoning_parts.append("Limited Twitter data available.")

        # Macro component
        if macro_count > 0:
            macro_sentiment_text = "favorable" if macro_score > 0.1 else "unfavorable" if macro_score < -0.1 else "neutral"
            reasoning_parts.append(
                f"Macroeconomic indicators ({macro_count} factors) are {macro_sentiment_text} "
                f"with score {macro_score:.2f}."
            )
        else:
            reasoning_parts.append("Limited macro data available.")

        # Market context
        if market_context:
            price_change = market_context.get('price_change_24h', 0)
            if abs(price_change) > 3:
                direction = "up" if price_change > 0 else "down"
                reasoning_parts.append(
                    f"Bitcoin price is {direction} {abs(price_change):.1f}% in the last 24 hours."
                )

        # Final recommendation context
        if signal == 'buy':
            reasoning_parts.append(
                "Overall sentiment and macro factors suggest positive momentum for Bitcoin."
            )
        elif signal == 'sell':
            reasoning_parts.append(
                "Overall sentiment and macro factors indicate potential downside risk."
            )
        else:
            reasoning_parts.append(
                "Mixed signals suggest maintaining current positions until clearer trends emerge."
            )

        return " ".join(reasoning_parts)

    def _identify_key_factors(self,
                            news_score: float,
                            twitter_score: float,
                            macro_score: float,
                            market_context: Dict = None) -> List[str]:
        """Identify the most influential factors"""
        factors = []

        if abs(news_score) > 0.2:
            sentiment = "Positive" if news_score > 0 else "Negative"
            factors.append(f"{sentiment} news sentiment")

        if abs(twitter_score) > 0.2:
            sentiment = "Positive" if twitter_score > 0 else "Negative"
            factors.append(f"{sentiment} Twitter/X sentiment from @WatcherGuru")

        if abs(macro_score) > 0.2:
            direction = "Supportive" if macro_score > 0 else "Unfavorable"
            factors.append(f"{direction} macroeconomic conditions")

        if market_context:
            price_change_24h = market_context.get('price_change_24h', 0)
            if abs(price_change_24h) > 5:
                factors.append(f"Significant 24h price movement ({price_change_24h:+.1f}%)")

            sentiment_up = market_context.get('sentiment_up_percentage', 50)
            if sentiment_up > 60:
                factors.append("High community bullish sentiment")
            elif sentiment_up < 40:
                factors.append("Low community sentiment")

        if not factors:
            factors.append("Neutral market conditions")

        return factors

    def run(self, days_back: int = 3) -> SignalOutput:
        """
        Main execution method - runs the full sentiment analysis pipeline

        Args:
            days_back: Number of days to look back for news

        Returns:
            SignalOutput with trading signal and reasoning
        """
        logger.info("=" * 60)
        logger.info("Starting Sentiment Agent Analysis")
        logger.info("=" * 60)

        try:
            # Step 1: Fetch data
            news_items = self.fetch_news_data(days_back=days_back)
            twitter_items = self.fetch_twitter_data(days_back=days_back)
            macro_indicators = self.fetch_macro_data()
            market_context = self.market_client.fetch_btc_market_context()

            # Step 2: Analyze
            news_sentiment = self.analyze_sentiment(news_items)
            twitter_sentiment = self.analyze_sentiment(twitter_items)
            macro_analysis = self.analyze_macro(macro_indicators)

            # Step 3: Generate signal
            signal_output = self.generate_signal(
                news_sentiment,
                twitter_sentiment,
                macro_analysis,
                market_context
            )

            logger.info("=" * 60)
            logger.info(f"Analysis complete: {signal_output.signal.upper()}")
            logger.info("=" * 60)

            return signal_output

        except Exception as e:
            logger.error(f"Error in sentiment agent execution: {e}", exc_info=True)

            # Return a safe default signal
            return SignalOutput(
                agent_type='sentiment_analysis',
                signal='hold',
                confidence=0.0,
                sentiment_score=0.0,
                news_sentiment=0.0,
                twitter_sentiment=0.0,
                macro_score=0.0,
                key_factors=['Error in analysis'],
                reasoning=f"Unable to complete analysis due to error: {str(e)}",
                timestamp=datetime.now(),
                data_sources=['Error']
            )


def main():
    """Example usage of the Sentiment Agent"""

    print("Bitcoin Sentiment Analysis Agent")
    print("=" * 60)

    # Initialize agent
    agent = SentimentAgent()

    # Run analysis
    signal = agent.run(days_back=3)

    # Display results
    print("\nANALYSIS RESULTS")
    print("=" * 60)
    print(f"Signal: {signal.signal.upper()}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Overall Sentiment Score: {signal.sentiment_score:.3f}")
    print(f"\nNews Sentiment: {signal.news_sentiment:.3f}")
    print(f"Macro Score: {signal.macro_score:.3f}")
    print(f"\nKey Factors:")
    for factor in signal.key_factors:
        print(f"  • {factor}")
    print(f"\nReasoning:")
    print(f"  {signal.reasoning}")
    print(f"\nTimestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Export as JSON for coordinator agent
    print("\nJSON Output for Coordinator:")
    print(signal.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

