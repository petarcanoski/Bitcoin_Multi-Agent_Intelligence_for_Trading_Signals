import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from datetime import datetime, timedelta
import numpy as np
from config import Config
from models import NewsItem, MacroIndicator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Sentiment analysis using FinBERT for financial text"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.SENTIMENT_MODEL
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the pre-trained FinBERT model"""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Sentiment analysis will use fallback method")

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        Returns: dict with 'score' (-1 to 1) and 'confidence'
        """
        if not text or not text.strip():
            return {'score': 0.0, 'confidence': 0.0}

        if not self.model or not self.tokenizer:
            return self._fallback_sentiment(text)

        try:
            # Tokenize and get prediction
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: [positive, negative, neutral]
            scores = predictions[0].tolist()
            positive, negative, neutral = scores

            # Convert to -1 to 1 scale
            sentiment_score = positive - negative
            confidence = max(scores)  # Confidence is the highest probability

            return {
                'score': sentiment_score,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._fallback_sentiment(text)

    def _fallback_sentiment(self, text: str) -> Dict[str, float]:
        """Simple keyword-based sentiment analysis as fallback"""
        text_lower = text.lower()

        positive_keywords = [
            'bullish', 'surge', 'gain', 'rally', 'up', 'increase', 'growth',
            'positive', 'adoption', 'breakthrough', 'success', 'profit', 'rise',
            'strong', 'momentum', 'boost', 'optimistic'
        ]

        negative_keywords = [
            'bearish', 'crash', 'drop', 'fall', 'decline', 'loss', 'down',
            'negative', 'concern', 'fear', 'risk', 'uncertainty', 'volatile',
            'weak', 'plunge', 'selloff', 'pessimistic', 'warning'
        ]

        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        total = pos_count + neg_count

        if total == 0:
            return {'score': 0.0, 'confidence': 0.3}

        score = (pos_count - neg_count) / total
        confidence = min(total / 5, 0.8)  # Cap confidence at 0.8 for fallback

        return {'score': score, 'confidence': confidence}

    def analyze_news_batch(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """
        Analyze sentiment for a batch of news items
        Applies time decay for news relevance
        """
        if not news_items:
            return []

        logger.info(f"Analyzing sentiment for {len(news_items)} news items")

        for news in news_items:
            # Combine title and description for analysis
            text = f"{news.title}. {news.description or ''}"
            sentiment_result = self.analyze_text(text)
            news.sentiment = sentiment_result['score']

            # Calculate relevance score based on time decay
            # Handle timezone-aware and naive datetimes
            now = datetime.now()
            published_at = news.published_at

            # Make both timezone-naive for comparison
            if published_at.tzinfo is not None:
                published_at = published_at.replace(tzinfo=None)
            if now.tzinfo is not None:
                now = now.replace(tzinfo=None)

            hours_old = (now - published_at).total_seconds() / 3600
            decay_factor = max(0.0, 1.0 - (hours_old / Config.NEWS_RELEVANCE_DECAY))
            news.relevance_score = decay_factor * sentiment_result['confidence']

        return news_items

    def aggregate_news_sentiment(self, news_items: List[NewsItem]) -> Dict[str, float]:
        """
        Aggregate sentiment from multiple news items
        Uses weighted average based on relevance
        """
        if not news_items:
            return {'average_sentiment': 0.0, 'confidence': 0.0, 'count': 0}

        weighted_sentiment = 0.0
        total_weight = 0.0

        for news in news_items:
            if news.sentiment is not None:
                weight = news.relevance_score
                weighted_sentiment += news.sentiment * weight
                total_weight += weight

        if total_weight == 0:
            return {'average_sentiment': 0.0, 'confidence': 0.0, 'count': len(news_items)}

        average_sentiment = weighted_sentiment / total_weight
        confidence = min(total_weight / len(news_items), 1.0)

        logger.info(f"News sentiment: {average_sentiment:.3f} (confidence: {confidence:.3f})")

        return {
            'average_sentiment': average_sentiment,
            'confidence': confidence,
            'count': len(news_items)
        }

    def analyze_macro_indicators(self, indicators: List[MacroIndicator]) -> Dict[str, float]:
        """
        Analyze macroeconomic indicators and return aggregate score
        """
        if not indicators:
            return {'macro_score': 0.0, 'confidence': 0.0, 'count': 0}

        weighted_score = 0.0
        total_weight = 0.0

        for indicator in indicators:
            # Convert impact to numeric score
            impact_map = {
                'positive': 1.0,
                'neutral': 0.0,
                'negative': -1.0
            }

            impact_score = impact_map.get(indicator.impact, 0.0)
            weight = indicator.weight

            weighted_score += impact_score * weight
            total_weight += weight

        if total_weight == 0:
            return {'macro_score': 0.0, 'confidence': 0.0, 'count': len(indicators)}

        macro_score = weighted_score / total_weight
        confidence = min(len(indicators) / 3, 1.0)  # Full confidence with 3+ indicators

        logger.info(f"Macro score: {macro_score:.3f} (confidence: {confidence:.3f})")

        return {
            'macro_score': macro_score,
            'confidence': confidence,
            'count': len(indicators)
        }

