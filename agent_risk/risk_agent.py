from typing import List, Dict
from datetime import datetime
import logging

from config import RiskConfig
from models import RiskSignalOutput, OnChainMetric, GeopoliticalEvent, VolatilityMetric
from onchain_clients import OnChainAnalyzer
from geopolitical_clients import GeopoliticalAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAgent:
    """
    Risk & Volatility Analysis Agent for Bitcoin Trading

    Analyzes:
    - On-chain metrics (transaction volume, whale movements, hash rate)
    - Geopolitical events (wars, sanctions, economic crises)
    - Market volatility and risk indicators

    Generates risk-adjusted signals with confidence scores
    """

    def __init__(self):
        self.onchain_analyzer = OnChainAnalyzer()
        self.geopolitical_analyzer = GeopoliticalAnalyzer()
        logger.info("Risk Agent initialized")

    def fetch_onchain_data(self) -> List[OnChainMetric]:
        """Fetch on-chain blockchain metrics"""
        logger.info("Fetching on-chain data")
        return self.onchain_analyzer.get_all_onchain_metrics()

    def fetch_geopolitical_data(self, days_back: int = None) -> List[GeopoliticalEvent]:
        """Fetch geopolitical events"""
        logger.info(f"Fetching geopolitical events from last {days_back or RiskConfig.GEOPOLITICAL_EVENTS_DAYS} days")
        return self.geopolitical_analyzer.fetch_geopolitical_events(days_back)

    def calculate_volatility_score(self, onchain_metrics: List[OnChainMetric]) -> float:
        """Calculate market volatility score based on on-chain metrics"""
        # Volatility indicated by rapid changes in volume, mempool size, etc.
        volatility_indicators = []

        for metric in onchain_metrics:
            if metric.metric_type == 'transaction_volume':
                # High or low volume indicates volatility
                if metric.risk_level in ['high', 'extreme']:
                    volatility_indicators.append(0.7)
                else:
                    volatility_indicators.append(0.3)

            elif metric.metric_type == 'mempool_size':
                # Large mempool indicates congestion/volatility
                if metric.risk_level == 'high':
                    volatility_indicators.append(0.6)
                else:
                    volatility_indicators.append(0.2)

        if not volatility_indicators:
            return 0.5  # Neutral

        volatility_score = sum(volatility_indicators) / len(volatility_indicators)
        logger.info(f"Volatility score: {volatility_score:.3f}")
        return volatility_score

    def generate_signal(self,
                       onchain_metrics: List[OnChainMetric],
                       geopolitical_events: List[GeopoliticalEvent]) -> RiskSignalOutput:
        """
        Generate risk-adjusted trading signal

        Signal logic:
        - Risk score < 0.3: LOW_RISK (bullish)
        - Risk score > 0.7: HIGH_RISK (bearish)
        - Otherwise: MEDIUM_RISK (neutral)
        """
        # Calculate component scores
        onchain_risk = self.onchain_analyzer.calculate_onchain_risk_score(onchain_metrics)
        geopolitical_risk = self.geopolitical_analyzer.calculate_geopolitical_risk_score(geopolitical_events)
        volatility_score = self.calculate_volatility_score(onchain_metrics)

        # Combined risk score (weighted)
        combined_risk = (
            onchain_risk * RiskConfig.ONCHAIN_WEIGHT +
            geopolitical_risk * RiskConfig.GEOPOLITICAL_WEIGHT
        )

        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(onchain_metrics, geopolitical_events)

        # Determine signal
        if combined_risk < RiskConfig.LOW_RISK_THRESHOLD:
            signal = 'low_risk'
        elif combined_risk > RiskConfig.HIGH_RISK_THRESHOLD:
            signal = 'high_risk'
        else:
            signal = 'medium_risk'

        # Identify key risks and opportunities
        key_risks = self._identify_key_risks(onchain_metrics, geopolitical_events, combined_risk)
        key_opportunities = self._identify_opportunities(onchain_metrics, geopolitical_events, combined_risk)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            signal, onchain_risk, geopolitical_risk, volatility_score,
            combined_risk, len(onchain_metrics), len(geopolitical_events)
        )

        # Create signal output
        signal_output = RiskSignalOutput(
            agent_type='risk_volatility',
            signal=signal,
            confidence=confidence,
            risk_score=combined_risk,
            onchain_risk=onchain_risk,
            geopolitical_risk=geopolitical_risk,
            volatility_score=volatility_score,
            key_risks=key_risks,
            key_opportunities=key_opportunities,
            onchain_metrics=onchain_metrics,
            geopolitical_events=geopolitical_events,
            reasoning=reasoning,
            timestamp=datetime.now(),
            data_sources=['Blockchain.info', 'NewsAPI', 'On-Chain Analysis']
        )

        logger.info(f"Generated signal: {signal.upper()} (risk: {combined_risk:.2f}, confidence: {confidence:.2f})")
        return signal_output

    def _calculate_confidence(self, onchain_metrics: List[OnChainMetric],
                             geopolitical_events: List[GeopoliticalEvent]) -> float:
        """Calculate confidence based on data availability"""
        onchain_conf = min(len(onchain_metrics) / RiskConfig.MIN_ONCHAIN_DATAPOINTS, 1.0)
        geo_conf = min(len(geopolitical_events) / RiskConfig.MIN_GEOPOLITICAL_EVENTS, 1.0)

        overall_conf = (onchain_conf + geo_conf) / 2
        return overall_conf

    def _identify_key_risks(self, onchain_metrics: List[OnChainMetric],
                           geopolitical_events: List[GeopoliticalEvent],
                           combined_risk: float) -> List[str]:
        """Identify key risk factors"""
        risks = []

        # On-chain risks
        for metric in onchain_metrics:
            if metric.risk_level in ['high', 'extreme']:
                risks.append(f"High {metric.metric_type.replace('_', ' ')} risk")

        # Geopolitical risks
        geo_risks = self.geopolitical_analyzer.identify_key_risks(geopolitical_events)
        risks.extend(geo_risks[:2])  # Top 2 geopolitical risks

        if not risks:
            risks.append("No major risks identified")

        return risks[:5]  # Top 5 risks

    def _identify_opportunities(self, onchain_metrics: List[OnChainMetric],
                               geopolitical_events: List[GeopoliticalEvent],
                               combined_risk: float) -> List[str]:
        """Identify potential opportunities"""
        opportunities = []

        # Low risk = opportunity
        if combined_risk < 0.3:
            opportunities.append("Low overall risk presents buying opportunity")

        # Strong network metrics
        for metric in onchain_metrics:
            if metric.metric_type == 'hash_rate' and metric.risk_level == 'low':
                opportunities.append("Strong network security (high hash rate)")
            if metric.metric_type == 'transaction_volume' and metric.risk_level == 'normal':
                opportunities.append("Healthy transaction volume")

        # Positive geopolitical impacts
        positive_events = [e for e in geopolitical_events if e.impact_on_crypto == 'positive']
        if positive_events:
            opportunities.append(f"Potential safe-haven demand from {len(positive_events)} events")

        if not opportunities:
            opportunities.append("Limited clear opportunities in current environment")

        return opportunities[:5]

    def _generate_reasoning(self, signal: str, onchain_risk: float,
                           geopolitical_risk: float, volatility_score: float,
                           combined_risk: float, onchain_count: int,
                           geopolitical_count: int) -> str:
        """Generate human-readable explanation"""
        signal_text = {
            'low_risk': 'LOW RISK',
            'medium_risk': 'MEDIUM RISK',
            'high_risk': 'HIGH RISK'
        }

        reasoning_parts = [
            f"{signal_text[signal]} signal generated with combined risk score of {combined_risk:.2f}."
        ]

        # On-chain component
        if onchain_count > 0:
            onchain_level = "low" if onchain_risk < 0.4 else "high" if onchain_risk > 0.6 else "moderate"
            reasoning_parts.append(
                f"On-chain analysis ({onchain_count} metrics) shows {onchain_level} risk "
                f"with score {onchain_risk:.2f}."
            )

        # Geopolitical component
        if geopolitical_count > 0:
            geo_level = "low" if geopolitical_risk < 0.4 else "high" if geopolitical_risk > 0.6 else "moderate"
            reasoning_parts.append(
                f"Geopolitical analysis ({geopolitical_count} events) indicates {geo_level} risk "
                f"with score {geopolitical_risk:.2f}."
            )

        # Volatility
        vol_level = "low" if volatility_score < 0.4 else "high" if volatility_score > 0.6 else "moderate"
        reasoning_parts.append(
            f"Market volatility is {vol_level} (score: {volatility_score:.2f})."
        )

        # Recommendation
        if signal == 'low_risk':
            reasoning_parts.append(
                "Overall risk environment is favorable for bullish positions or entry."
            )
        elif signal == 'high_risk':
            reasoning_parts.append(
                "Elevated risk levels suggest caution, defensive positioning, or reduced exposure."
            )
        else:
            reasoning_parts.append(
                "Mixed risk signals suggest maintaining current positions with close monitoring."
            )

        return " ".join(reasoning_parts)

    def run(self, geopolitical_days_back: int = None) -> RiskSignalOutput:
        """
        Main execution method - runs the full risk analysis pipeline

        Args:
            geopolitical_days_back: Days to look back for geopolitical events

        Returns:
            RiskSignalOutput with risk assessment and signal
        """
        logger.info("=" * 60)
        logger.info("Starting Risk Agent Analysis")
        logger.info("=" * 60)

        try:
            # Step 1: Fetch data
            onchain_metrics = self.fetch_onchain_data()
            geopolitical_events = self.fetch_geopolitical_data(geopolitical_days_back)

            # Step 2: Generate signal
            signal_output = self.generate_signal(onchain_metrics, geopolitical_events)

            logger.info("=" * 60)
            logger.info(f"Analysis complete: {signal_output.signal.upper()}")
            logger.info("=" * 60)

            return signal_output

        except Exception as e:
            logger.error(f"Error in risk agent execution: {e}", exc_info=True)

            # Return safe default
            return RiskSignalOutput(
                agent_type='risk_volatility',
                signal='medium_risk',
                confidence=0.0,
                risk_score=0.5,
                onchain_risk=0.5,
                geopolitical_risk=0.5,
                volatility_score=0.5,
                key_risks=['Error in analysis'],
                key_opportunities=[],
                reasoning=f"Unable to complete analysis due to error: {str(e)}",
                timestamp=datetime.now(),
                data_sources=['Error']
            )

