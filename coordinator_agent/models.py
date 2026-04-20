from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class TechnicalSignal(BaseModel):
    agent_type: str = "technical_analysis"
    signal: str
    confidence: float = Field(ge=0.0, le=1.0)
    trade_probability: float = Field(ge=0.0, le=1.0)
    long_probability: float = Field(ge=0.0, le=1.0)
    sequence_length: int
    feature_count: int
    timestamp: datetime = Field(default_factory=datetime.now)
    reasoning: str


class FinalCoordinatorSignal(BaseModel):
    signal: str
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=-1.0, le=1.0)
    risk_level: str
    key_factors: List[str]
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data_sources: List[str]

