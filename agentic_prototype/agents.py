from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from pydantic import BaseModel, Field

from .llm_backend import NarrativeLLMScorer
from .skills import SkillRegistry


class AgentOutput(BaseModel):
    signal: str
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    used_skills: List[str]


@dataclass
class BaseAgent:
    name: str
    skills: SkillRegistry


class TechnicalAgent(BaseAgent):
    def analyze(self, case: Dict) -> AgentOutput:
        momentum = self.skills.run("normalize_score", float(case["technical_momentum"]))
        trend_strength = self.skills.run("normalize_score", float(case["trend_strength"]))
        volatility = float(case["volatility"])
        volatility_penalty = -0.25 * max(0.0, min(1.0, volatility))
        score = self.skills.run(
            "weighted_vote",
            [
                (0.50, momentum),
                (0.35, trend_strength),
                (0.15, volatility_penalty),
            ],
        )
        signal = self.skills.run("signal_from_score", score)
        confidence = self.skills.run("confidence_from_score", score)
        reasoning = self.skills.run(
            "concise_reasoning",
            self.name,
            [
                f"momentum={momentum:+.2f}",
                f"trend_strength={trend_strength:+.2f}",
                f"volatility_penalty={volatility_penalty:+.2f}",
            ],
            score,
            signal,
        )
        return AgentOutput(
            signal=signal,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            used_skills=["normalize_score", "weighted_vote", "signal_from_score", "confidence_from_score", "concise_reasoning"],
        )


class SentimentMacroAgent(BaseAgent):
    def __init__(self, name: str, skills: SkillRegistry, llm_scorer: NarrativeLLMScorer | None = None):
        super().__init__(name=name, skills=skills)
        self.llm_scorer = llm_scorer or NarrativeLLMScorer()

    def analyze(self, case: Dict) -> AgentOutput:
        news = self.skills.run("normalize_score", float(case["news_sentiment"]))
        social = self.skills.run("normalize_score", float(case["social_sentiment"]))
        macro = self.skills.run("normalize_score", float(case["macro_sentiment"]))
        narrative_bias = self.llm_scorer.score_narrative(str(case.get("narrative", "")))

        score = self.skills.run(
            "weighted_vote",
            [
                (0.45, news),
                (0.30, social),
                (0.20, macro),
                (0.05, narrative_bias),
            ],
        )
        signal = self.skills.run("signal_from_score", score)
        confidence = self.skills.run("confidence_from_score", score)
        reasoning = self.skills.run(
            "concise_reasoning",
            self.name,
            [
                f"news={news:+.2f}",
                f"social={social:+.2f}",
                f"macro={macro:+.2f}",
                f"narrative_bias={narrative_bias:+.2f}",
            ],
            score,
            signal,
        )
        return AgentOutput(
            signal=signal,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            used_skills=["normalize_score", "weighted_vote", "signal_from_score", "confidence_from_score", "concise_reasoning"],
        )


class RiskAgent(BaseAgent):
    def analyze(self, case: Dict) -> AgentOutput:
        onchain_risk = max(0.0, min(1.0, float(case["onchain_risk"])))
        geopolitical_risk = max(0.0, min(1.0, float(case["geopolitical_risk"])))
        risk_score = self.skills.run("weighted_vote", [(0.6, onchain_risk), (0.4, geopolitical_risk)])

        # Convert risk to action score (higher risk suppresses risk-taking).
        score = 1.0 - 2.0 * risk_score
        signal = self.skills.run("signal_from_score", score)
        confidence = self.skills.run("confidence_from_score", score)
        reasoning = self.skills.run(
            "concise_reasoning",
            self.name,
            [f"onchain_risk={onchain_risk:.2f}", f"geopolitical_risk={geopolitical_risk:.2f}"],
            score,
            signal,
        )
        return AgentOutput(
            signal=signal,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            used_skills=["weighted_vote", "signal_from_score", "confidence_from_score", "concise_reasoning"],
        )


class CoordinatorAgent(BaseAgent):
    def decide(self, technical: AgentOutput, sentiment: AgentOutput, risk: AgentOutput) -> AgentOutput:
        score = self.skills.run(
            "weighted_vote",
            [
                (0.45, technical.score),
                (0.35, sentiment.score),
                (0.20, risk.score),
            ],
        )
        signal = self.skills.run("signal_from_score", score)
        confidence = min(
            1.0,
            (0.45 * technical.confidence + 0.35 * sentiment.confidence + 0.20 * risk.confidence),
        )
        reasoning = self.skills.run(
            "concise_reasoning",
            self.name,
            [
                f"technical={technical.score:+.2f}",
                f"sentiment={sentiment.score:+.2f}",
                f"risk={risk.score:+.2f}",
            ],
            score,
            signal,
        )
        return AgentOutput(
            signal=signal,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            used_skills=["weighted_vote", "signal_from_score", "concise_reasoning"],
        )


