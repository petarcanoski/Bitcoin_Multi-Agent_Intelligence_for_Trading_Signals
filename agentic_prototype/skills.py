from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Skill:

    name: str
    description: str
    run: Callable[..., float | str | dict]


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def run(self, skill_name: str, *args, **kwargs):
        if skill_name not in self._skills:
            raise KeyError(f"Unknown skill: {skill_name}")
        return self._skills[skill_name].run(*args, **kwargs)


def normalize_score(value: float, min_value: float = -1.0, max_value: float = 1.0) -> float:
    if max_value <= min_value:
        raise ValueError("max_value must be greater than min_value")
    value = max(min_value, min(max_value, value))
    center = (min_value + max_value) / 2.0
    scale = (max_value - min_value) / 2.0
    return (value - center) / scale


def weighted_vote(weighted_scores: Iterable[Tuple[float, float]]) -> float:
    weighted_scores = list(weighted_scores)
    if not weighted_scores:
        return 0.0
    total_weight = sum(weight for weight, _ in weighted_scores)
    if total_weight == 0:
        return 0.0
    return sum(weight * score for weight, score in weighted_scores) / total_weight


def signal_from_score(score: float, buy_threshold: float = 0.20, sell_threshold: float = -0.20) -> str:
    if score >= buy_threshold:
        return "buy"
    if score <= sell_threshold:
        return "sell"
    return "hold"


def confidence_from_score(score: float, floor: float = 0.35, ceiling: float = 0.95) -> float:
    confidence = floor + abs(score) * (ceiling - floor)
    return max(floor, min(ceiling, confidence))


def concise_reasoning(agent_name: str, factors: List[str], score: float, signal: str) -> str:
    factors_txt = "; ".join(factors)
    return f"{agent_name} computed score {score:+.2f} -> {signal.upper()}. Factors: {factors_txt}."


def build_default_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(Skill("normalize_score", "Clip and normalize a score to [-1, 1].", normalize_score))
    registry.register(Skill("weighted_vote", "Compute weighted average score.", weighted_vote))
    registry.register(Skill("signal_from_score", "Map score to BUY/SELL/HOLD signal.", signal_from_score))
    registry.register(Skill("confidence_from_score", "Convert score magnitude to confidence.", confidence_from_score))
    registry.register(Skill("concise_reasoning", "Create one-line explainable reasoning.", concise_reasoning))
    return registry

