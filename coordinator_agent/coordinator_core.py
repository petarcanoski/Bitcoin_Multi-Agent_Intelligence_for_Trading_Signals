import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

try:
    from .models import FinalCoordinatorSignal
    from .technical_agent import TechnicalAgent
except ImportError:
    from models import FinalCoordinatorSignal
    from technical_agent import TechnicalAgent


class CoordinatorAgent:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.sentiment_dir = repo_root / "sentiment_analysis"
        self.risk_dir = repo_root / "agent_risk"
        self.technical_agent = TechnicalAgent(repo_root=repo_root)

    @staticmethod
    def _signal_to_score(signal: str) -> int:
        mapping = {"buy": 1, "hold": 0, "sell": -1}
        return mapping.get(signal.lower(), 0)

    def _run_json_script(self, cwd: Path, script_name: str) -> Dict:
        completed = subprocess.run(
            [sys.executable, script_name],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"{script_name} failed with code {completed.returncode}. stderr={completed.stderr.strip()}"
            )

        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)

        raise RuntimeError(f"No JSON payload found in output of {script_name}.")

    def _combine_signals(self, technical: Dict, sentiment: Dict, risk: Dict) -> Tuple[str, float, float, str]:
        tech_score = self._signal_to_score(technical["signal"]) * float(technical.get("confidence", 0.0))
        sentiment_score = self._signal_to_score(sentiment["signal"]) * float(sentiment.get("confidence", 0.0))

        # Higher weight for technical model because it is directly trained on OHLCV-derived features.
        combined_score = 0.60 * tech_score + 0.40 * sentiment_score

        risk_signal = str(risk.get("signal", "medium_risk")).lower()
        risk_multiplier = {
            "low_risk": 1.10,
            "medium_risk": 1.00,
            "high_risk": 0.65,
        }.get(risk_signal, 1.0)

        adjusted_score = max(-1.0, min(1.0, combined_score * risk_multiplier))

        # Conservative risk gate: high risk suppresses BUY unless conviction is very high.
        if risk_signal == "high_risk" and adjusted_score > 0.55:
            final_signal = "hold"
        elif adjusted_score >= 0.25:
            final_signal = "buy"
        elif adjusted_score <= -0.25:
            final_signal = "sell"
        else:
            final_signal = "hold"

        confidence = min(1.0, abs(adjusted_score) + 0.15 * float(risk.get("confidence", 0.0)))
        return final_signal, confidence, adjusted_score, risk_signal

    def run(self) -> FinalCoordinatorSignal:
        technical = self.technical_agent.run().model_dump()
        sentiment = self._run_json_script(self.sentiment_dir, "run_agent_json.py")
        risk = self._run_json_script(self.risk_dir, "run_agent_json.py")

        signal, confidence, score, risk_level = self._combine_signals(technical, sentiment, risk)

        key_factors = [
            f"Technical: {technical['signal'].upper()} ({technical['confidence']:.2f})",
            f"Sentiment: {sentiment.get('signal', 'hold').upper()} ({float(sentiment.get('confidence', 0.0)):.2f})",
            f"Risk: {risk_level.upper()} ({float(risk.get('risk_score', 0.5)):.2f})",
        ]

        reasoning = (
            f"Coordinator combined technical and sentiment scores into {score:+.2f}, then adjusted using risk level "
            f"{risk_level}. Final signal is {signal.upper()} with confidence {confidence:.2f}. "
            f"Technical reason: {technical['reasoning']} Sentiment reason: {sentiment.get('reasoning', 'N/A')} "
            f"Risk reason: {risk.get('reasoning', 'N/A')}"
        )

        return FinalCoordinatorSignal(
            signal=signal,
            confidence=confidence,
            score=score,
            risk_level=risk_level,
            key_factors=key_factors,
            reasoning=reasoning,
            data_sources=[
                "bitcoin-predictor-dev models",
                "bitcoin-predictor-dev features_1h.parquet",
                "sentiment_analysis",
                "agent_risk",
            ],
        )



