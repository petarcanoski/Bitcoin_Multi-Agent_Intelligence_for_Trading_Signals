"""
Geopolitical Events Client
Monitors wars, conflicts, sanctions, economic crises, and political instability
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from config import RiskConfig
from models import GeopoliticalEvent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeopoliticalAnalyzer:
    """Analyzes geopolitical events and their impact on crypto markets"""

    def __init__(self, newsapi_key: str = None):
        self.newsapi_key = newsapi_key or RiskConfig.NEWSAPI_KEY
        if not self.newsapi_key:
            logger.warning("NewsAPI key not found. Using mock geopolitical data.")

    def fetch_geopolitical_events(self, days_back: int = None) -> List[GeopoliticalEvent]:
        """Fetch recent geopolitical events"""
        days_back = days_back or RiskConfig.GEOPOLITICAL_EVENTS_DAYS

        if not self.newsapi_key:
            return self._get_mock_geopolitical_events()

        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            events = []

            # Search for different types of events
            for keyword in RiskConfig.GEOPOLITICAL_KEYWORDS[:5]:  # Limit to avoid rate limits
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': keyword,
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': 5,
                    'apiKey': self.newsapi_key
                }

                response = requests.get(url, params=params, timeout=RiskConfig.REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()

                for article in data.get('articles', []):
                    event = self._parse_event_from_article(article, keyword)
                    if event:
                        events.append(event)

            logger.info(f"Fetched {len(events)} geopolitical events")
            return events[:20]  # Limit to most recent 20

        except Exception as e:
            logger.error(f"Error fetching geopolitical events: {e}")
            return self._get_mock_geopolitical_events()

    def _parse_event_from_article(self, article: Dict, keyword: str) -> Optional[GeopoliticalEvent]:
        """Parse news article into geopolitical event"""
        try:
            title = article.get('title', '')
            description = article.get('description', '')

            # Determine event type based on keyword
            event_type = self._classify_event_type(title + ' ' + description)

            # Determine severity
            severity = self._assess_severity(title, description)

            # Determine impact on crypto
            impact = self._assess_crypto_impact(title, description, event_type)

            # Determine region
            region = self._detect_region(title + ' ' + description)

            return GeopoliticalEvent(
                event_type=event_type,
                title=title[:200],
                description=description[:500] if description else title,
                region=region,
                severity=severity,
                impact_on_crypto=impact,
                confidence=0.7,
                timestamp=datetime.fromisoformat(
                    article.get('publishedAt', '').replace('Z', '+00:00')
                ),
                source=article.get('source', {}).get('name', 'Unknown')
            )
        except Exception as e:
            logger.debug(f"Error parsing article: {e}")
            return None

    def _classify_event_type(self, text: str) -> str:
        """Classify the type of geopolitical event"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['war', 'conflict', 'invasion', 'attack']):
            return 'war'
        elif any(word in text_lower for word in ['sanction', 'embargo', 'ban']):
            return 'sanctions'
        elif any(word in text_lower for word in ['crisis', 'collapse', 'recession']):
            return 'economic_crisis'
        elif any(word in text_lower for word in ['trade war', 'tariff', 'trade dispute']):
            return 'trade_war'
        else:
            return 'political_instability'

    def _assess_severity(self, title: str, description: str) -> str:
        """Assess event severity"""
        text = (title + ' ' + (description or '')).lower()

        critical_words = ['nuclear', 'world war', 'invasion', 'collapse', 'crash']
        high_words = ['war', 'conflict', 'crisis', 'sanction', 'attack']
        medium_words = ['tension', 'dispute', 'concern', 'warning']

        if any(word in text for word in critical_words):
            return 'critical'
        elif any(word in text for word in high_words):
            return 'high'
        elif any(word in text for word in medium_words):
            return 'medium'
        else:
            return 'low'

    def _assess_crypto_impact(self, title: str, description: str, event_type: str) -> str:
        """Assess impact on cryptocurrency markets"""
        text = (title + ' ' + (description or '')).lower()

        # Wars and crises can be positive (safe haven) or negative (risk-off)
        if event_type in ['war', 'economic_crisis', 'sanctions']:
            # Bitcoin often seen as hedge against instability
            if any(word in text for word in ['dollar', 'currency', 'inflation']):
                return 'positive'  # Potential safe haven
            else:
                return 'negative'  # Risk-off sentiment

        elif event_type == 'trade_war':
            return 'negative'  # Generally negative for risk assets

        else:
            return 'neutral'

    def _detect_region(self, text: str) -> str:
        """Detect geographic region of event"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['us', 'usa', 'united states', 'america']):
            return 'usa'
        elif any(word in text_lower for word in ['china', 'chinese', 'beijing']):
            return 'china'
        elif any(word in text_lower for word in ['europe', 'eu', 'european']):
            return 'europe'
        elif any(word in text_lower for word in ['russia', 'russian', 'moscow']):
            return 'russia'
        elif any(word in text_lower for word in ['middle east', 'israel', 'iran', 'saudi']):
            return 'middle_east'
        elif any(word in text_lower for word in ['asia', 'japan', 'korea', 'india']):
            return 'asia'
        else:
            return 'other'

    def calculate_geopolitical_risk_score(self, events: List[GeopoliticalEvent]) -> float:
        """Calculate overall geopolitical risk score (0.0 to 1.0)"""
        if not events:
            return 0.3  # Low baseline risk

        severity_map = {'low': 0.2, 'medium': 0.5, 'high': 0.7, 'critical': 1.0}
        impact_map = {'positive': -0.2, 'neutral': 0.0, 'negative': 0.3}

        total_risk = 0.0
        total_weight = 0.0

        for event in events:
            severity_score = severity_map.get(event.severity, 0.5)
            impact_adjustment = impact_map.get(event.impact_on_crypto, 0.0)
            region_weight = RiskConfig.REGION_WEIGHTS.get(event.region, 0.3)

            event_risk = (severity_score + impact_adjustment) * region_weight * event.confidence
            total_risk += event_risk
            total_weight += region_weight

        if total_weight == 0:
            return 0.3

        normalized_risk = min(total_risk / total_weight, 1.0)
        logger.info(f"Geopolitical risk score: {normalized_risk:.3f}")
        return normalized_risk

    def identify_key_risks(self, events: List[GeopoliticalEvent]) -> List[str]:
        """Identify key geopolitical risks from events"""
        risks = []

        # Group by severity and type
        critical_events = [e for e in events if e.severity == 'critical']
        high_events = [e for e in events if e.severity == 'high']

        for event in critical_events[:2]:
            risks.append(f"Critical: {event.event_type.replace('_', ' ').title()} in {event.region}")

        for event in high_events[:3]:
            risks.append(f"High: {event.event_type.replace('_', ' ').title()} - {event.title[:60]}")

        if not risks:
            risks.append("No major geopolitical risks detected")

        return risks

    def _get_mock_geopolitical_events(self) -> List[GeopoliticalEvent]:
        """Mock geopolitical events for testing"""
        logger.info("Using mock geopolitical events")

        return [
            GeopoliticalEvent(
                event_type='trade_war',
                title='US imposes new tariffs on Chinese imports',
                description='United States announces 25% tariffs on $200B of Chinese goods, escalating trade tensions',
                region='usa',
                severity='high',
                impact_on_crypto='negative',
                confidence=0.9,
                timestamp=datetime.now() - timedelta(days=2),
                source='Reuters'
            ),
            GeopoliticalEvent(
                event_type='sanctions',
                title='New sanctions imposed on Russia energy sector',
                description='Western nations coordinate new sanctions targeting Russian oil exports',
                region='russia',
                severity='high',
                impact_on_crypto='positive',  # May drive adoption
                confidence=0.85,
                timestamp=datetime.now() - timedelta(days=5),
                source='Bloomberg'
            ),
            GeopoliticalEvent(
                event_type='economic_crisis',
                title='Inflation concerns rise in Europe amid energy crisis',
                description='European Central Bank warns of stagflation risks as energy prices surge',
                region='europe',
                severity='medium',
                impact_on_crypto='positive',  # Bitcoin as inflation hedge
                confidence=0.75,
                timestamp=datetime.now() - timedelta(days=7),
                source='Financial Times'
            ),
            GeopoliticalEvent(
                event_type='war',
                title='Ongoing conflict in Middle East raises oil concerns',
                description='Regional tensions escalate, impacting global energy markets',
                region='middle_east',
                severity='high',
                impact_on_crypto='negative',  # Risk-off sentiment
                confidence=0.8,
                timestamp=datetime.now() - timedelta(days=10),
                source='CNN'
            )
        ]


