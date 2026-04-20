import requests
from typing import List, Dict
from datetime import datetime, timedelta
from config import RiskConfig
from models import OnChainMetric, WhaleTransaction
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockchainInfoClient:
    """Client for Blockchain.info API (free, no API key needed)"""

    BASE_URL = "https://blockchain.info"

    def fetch_network_stats(self) -> Dict:
        """Fetch basic Bitcoin network statistics"""
        try:
            url = f"{self.BASE_URL}/stats?format=json"
            response = requests.get(url, timeout=RiskConfig.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            logger.info("Fetched blockchain network stats")
            return {
                'hash_rate': data.get('hash_rate', 0),
                'difficulty': data.get('difficulty', 0),
                'total_btc': data.get('totalbc', 0) / 1e8,
                'market_price_usd': data.get('market_price_usd', 0),
                'trade_volume_24h': data.get('trade_volume_usd', 0),
                'mempool_size': data.get('n_tx_mempool', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching blockchain stats: {e}")
            return self._get_mock_network_stats()

    def fetch_recent_blocks(self, limit: int = 10) -> List[Dict]:
        """Fetch recent block data"""
        try:
            url = f"{self.BASE_URL}/blocks/{int(datetime.now().timestamp() * 1000)}?format=json"
            response = requests.get(url, timeout=RiskConfig.REQUEST_TIMEOUT)
            response.raise_for_status()
            blocks = response.json()

            logger.info(f"Fetched {len(blocks[:limit])} recent blocks")
            return blocks[:limit]
        except Exception as e:
            logger.error(f"Error fetching recent blocks: {e}")
            return []

    def _get_mock_network_stats(self) -> Dict:
        """Mock data for testing"""
        logger.info("Using mock blockchain network stats")
        return {
            'hash_rate': 350_000_000_000_000_000,  # 350 EH/s
            'difficulty': 50_000_000_000_000,
            'total_btc': 19_500_000,
            'market_price_usd': 68000,
            'trade_volume_24h': 25_000_000_000,
            'mempool_size': 50000
        }


class OnChainAnalyzer:
    """Analyzes on-chain metrics and generates risk signals"""

    def __init__(self):
        self.blockchain_client = BlockchainInfoClient()

    def analyze_transaction_volume(self) -> OnChainMetric:
        """Analyze Bitcoin transaction volume"""
        try:
            stats = self.blockchain_client.fetch_network_stats()
            volume_24h = stats.get('trade_volume_24h', 0)

            # Determine risk level based on volume
            if volume_24h > RiskConfig.HIGH_VOLUME_THRESHOLD:
                risk_level = 'high'  # High volume can mean volatility
            elif volume_24h < RiskConfig.LOW_VOLUME_THRESHOLD:
                risk_level = 'high'  # Low volume can mean illiquidity
            else:
                risk_level = 'normal'

            return OnChainMetric(
                metric_type='transaction_volume',
                value=volume_24h,
                timestamp=datetime.now(),
                risk_level=risk_level
            )
        except Exception as e:
            logger.error(f"Error analyzing transaction volume: {e}")
            return self._get_mock_volume_metric()

    def analyze_hash_rate(self) -> OnChainMetric:
        """Analyze Bitcoin network hash rate (security indicator)"""
        try:
            stats = self.blockchain_client.fetch_network_stats()
            hash_rate = stats.get('hash_rate', 0)

            # Lower hash rate = higher security risk
            # Typical range: 300-400 EH/s
            if hash_rate < 250_000_000_000_000_000:  # < 250 EH/s
                risk_level = 'high'
            elif hash_rate > 350_000_000_000_000_000:  # > 350 EH/s
                risk_level = 'low'
            else:
                risk_level = 'normal'

            return OnChainMetric(
                metric_type='hash_rate',
                value=hash_rate,
                timestamp=datetime.now(),
                risk_level=risk_level
            )
        except Exception as e:
            logger.error(f"Error analyzing hash rate: {e}")
            return self._get_mock_hash_rate_metric()

    def analyze_mempool_size(self) -> OnChainMetric:
        """Analyze mempool size (network congestion indicator)"""
        try:
            stats = self.blockchain_client.fetch_network_stats()
            mempool_size = stats.get('mempool_size', 0)

            # Large mempool = network congestion
            if mempool_size > 100000:
                risk_level = 'high'
            elif mempool_size < 20000:
                risk_level = 'low'
            else:
                risk_level = 'normal'

            return OnChainMetric(
                metric_type='mempool_size',
                value=mempool_size,
                timestamp=datetime.now(),
                risk_level=risk_level
            )
        except Exception as e:
            logger.error(f"Error analyzing mempool: {e}")
            return self._get_mock_mempool_metric()

    def detect_whale_movements(self) -> List[WhaleTransaction]:
        """Detect large whale transactions (>100 BTC)"""
        try:
            # This would require a more advanced API (like Glassnode)
            # For now, return mock data
            logger.info("Analyzing whale movements (mock data)")
            return self._get_mock_whale_transactions()
        except Exception as e:
            logger.error(f"Error detecting whale movements: {e}")
            return []

    def get_all_onchain_metrics(self) -> List[OnChainMetric]:
        """Get all on-chain metrics for comprehensive analysis"""
        metrics = []

        metrics.append(self.analyze_transaction_volume())
        metrics.append(self.analyze_hash_rate())
        metrics.append(self.analyze_mempool_size())

        logger.info(f"Collected {len(metrics)} on-chain metrics")
        return metrics

    def calculate_onchain_risk_score(self, metrics: List[OnChainMetric]) -> float:
        """Calculate overall on-chain risk score (0.0 to 1.0)"""
        if not metrics:
            return 0.5  # Neutral if no data

        risk_map = {'low': 0.2, 'normal': 0.5, 'high': 0.8, 'extreme': 1.0}
        risk_scores = [risk_map.get(m.risk_level, 0.5) for m in metrics]

        avg_risk = sum(risk_scores) / len(risk_scores)
        logger.info(f"On-chain risk score: {avg_risk:.3f}")
        return avg_risk

    # Mock data methods
    def _get_mock_volume_metric(self) -> OnChainMetric:
        return OnChainMetric(
            metric_type='transaction_volume',
            value=25_000_000_000,
            timestamp=datetime.now(),
            risk_level='normal'
        )

    def _get_mock_hash_rate_metric(self) -> OnChainMetric:
        return OnChainMetric(
            metric_type='hash_rate',
            value=350_000_000_000_000_000,
            timestamp=datetime.now(),
            risk_level='low'
        )

    def _get_mock_mempool_metric(self) -> OnChainMetric:
        return OnChainMetric(
            metric_type='mempool_size',
            value=45000,
            timestamp=datetime.now(),
            risk_level='normal'
        )

    def _get_mock_whale_transactions(self) -> List[WhaleTransaction]:
        """Mock whale transactions"""
        return [
            WhaleTransaction(
                transaction_hash='abc123...',
                amount_btc=150.5,
                amount_usd=10_234_000,
                from_address='wallet_x',
                to_address='binance_deposit',
                transaction_type='exchange_deposit',
                timestamp=datetime.now() - timedelta(hours=2),
                risk_implication='bearish'  # Large exchange deposit = potential sell
            ),
            WhaleTransaction(
                transaction_hash='def456...',
                amount_btc=200.0,
                amount_usd=13_600_000,
                from_address='coinbase_withdrawal',
                to_address='cold_wallet',
                transaction_type='exchange_withdrawal',
                timestamp=datetime.now() - timedelta(hours=5),
                risk_implication='bullish'  # Withdrawal to cold storage = HODLing
            )
        ]

