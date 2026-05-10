from __future__ import annotations

import json
import os
from dataclasses import dataclass

import requests


@dataclass
class NarrativeLLMScorer:

    model: str = "llama-3.1-8b-instant"

    POSITIVE_CUES = ("etf", "adoption", "soft landing", "rate cut", "inflows")
    NEGATIVE_CUES = ("ban", "lawsuit", "liquidation", "war", "rate hike")

    def score_narrative(self, narrative: str) -> float:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return self._fallback_score(narrative)

        prompt = (
            "Return only JSON with one key 'bias' in range [-0.25, 0.25]. "
            "Positive macro/market narrative => positive bias, negative narrative => negative bias. "
            f"Narrative: {narrative}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 60,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=12,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            parsed = json.loads(content)
            bias = float(parsed.get("bias", 0.0))
            return max(-0.25, min(0.25, bias))
        except Exception:
            return self._fallback_score(narrative)

    def _fallback_score(self, narrative: str) -> float:
        text = narrative.lower()
        bias = 0.0
        if any(cue in text for cue in self.POSITIVE_CUES):
            bias += 0.12
        if any(cue in text for cue in self.NEGATIVE_CUES):
            bias -= 0.12
        return max(-0.25, min(0.25, bias))

