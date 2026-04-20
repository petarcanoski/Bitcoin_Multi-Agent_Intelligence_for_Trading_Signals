from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class SignalOutput(BaseModel):
    """Standardized output format for agent signals"""

    agent_type: str = Field(description="Type of agent (e.g., 'sentiment_analysis')")
    signal: str = Field(description="Trading signal: 'buy', 'sell', or 'hold'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Overall sentiment score")

    # Detailed breakdown
    news_sentiment: float = Field(ge=-1.0, le=1.0, description="News sentiment component")
    twitter_sentiment: float = Field(ge=-1.0, le=1.0, description="Twitter/X sentiment component", default=0.0)
    macro_score: float = Field(ge=-1.0, le=1.0, description="Macro analysis component")

    key_factors: List[str] = Field(description="List of key factors influencing the signal")
    reasoning: str = Field(description="Human-readable explanation of the signal")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    data_sources: List[str] = Field(default_factory=list, description="APIs/sources used")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NewsItem(BaseModel):
    """Model for news articles"""

    title: str
    description: Optional[str] = None
    source: str
    published_at: datetime
    url: Optional[str] = None
    sentiment: Optional[float] = None
    relevance_score: float = 1.0


class MacroIndicator(BaseModel):
    """Model for macroeconomic indicators"""

    indicator_type: str  # 'cpi', 'fed_decision', 'tariff', etc.
    value: Optional[float] = None
    description: str
    impact: str  # 'positive', 'negative', 'neutral'
    weight: float
    published_at: datetime

