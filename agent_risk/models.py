from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class OnChainMetric(BaseModel):
    """Model for on-chain blockchain metrics"""

    metric_type: str  # e.g., 'transaction_volume', 'whale_movements'
    value: float
    timestamp: datetime
    change_24h: Optional[float] = None  # Percentage change in 24h
    change_7d: Optional[float] = None   # Percentage change in 7d
    risk_level: str = 'normal'  # 'low', 'normal', 'high', 'extreme'


class GeopoliticalEvent(BaseModel):
    """Model for geopolitical events"""

    event_type: str  # 'war', 'sanctions', 'crisis', etc.
    title: str
    description: str
    region: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    impact_on_crypto: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    source: str


class VolatilityMetric(BaseModel):
    """Model for volatility measurements"""

    period: str  # 'short_term', 'medium_term', 'long_term'
    volatility_index: float  # 0.0 to 1.0+
    price_range_high: float
    price_range_low: float
    average_volume: float
    timestamp: datetime


class RiskSignalOutput(BaseModel):
    """Standardized output format for Risk Agent signals"""

    agent_type: str = Field(default='risk_volatility', description="Type of agent")
    signal: str = Field(description="Risk signal: 'low_risk', 'medium_risk', or 'high_risk'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    risk_score: float = Field(ge=0.0, le=1.0, description="Overall risk score (0=low, 1=high)")

    # Component breakdowns
    onchain_risk: float = Field(ge=0.0, le=1.0, description="On-chain risk component")
    geopolitical_risk: float = Field(ge=0.0, le=1.0, description="Geopolitical risk component")
    volatility_score: float = Field(ge=0.0, le=1.0, description="Market volatility score")

    # Detailed metrics
    key_risks: List[str] = Field(description="List of identified key risks")
    key_opportunities: List[str] = Field(description="List of identified opportunities")

    # Analysis details
    onchain_metrics: List[OnChainMetric] = Field(default_factory=list)
    geopolitical_events: List[GeopoliticalEvent] = Field(default_factory=list)
    volatility_metrics: List[VolatilityMetric] = Field(default_factory=list)

    reasoning: str = Field(description="Human-readable explanation of the risk assessment")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    data_sources: List[str] = Field(default_factory=list, description="APIs/sources used")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WhaleTransaction(BaseModel):
    """Model for large whale transactions"""

    transaction_hash: str
    amount_btc: float
    amount_usd: float
    from_address: str
    to_address: str
    transaction_type: str  # 'exchange_deposit', 'exchange_withdrawal', 'wallet_transfer'
    timestamp: datetime
    risk_implication: str  # 'bearish', 'bullish', 'neutral'

