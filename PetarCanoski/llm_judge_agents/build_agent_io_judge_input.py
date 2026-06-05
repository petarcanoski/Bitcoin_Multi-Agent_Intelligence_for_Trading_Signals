from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "project-context" / "evaluation_agents_raw_1.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump(mode="json"))
        except Exception:
            return _to_jsonable(value.model_dump())
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _run_subprocess_json(cwd: Path, script_name: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, script_name],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{script_name} failed with code {completed.returncode}. stderr={completed.stderr.strip()}")

    lines = [ln.strip() for ln in (completed.stdout or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"No JSON object found in output of {script_name}")


def _capture_technical() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from coordinator_agent.technical_agent import TechnicalAgent

    agent = TechnicalAgent(repo_root=REPO_ROOT)
    output = _to_jsonable(agent.run().model_dump())

    features_path = REPO_ROOT / "technical_analysis" / "data" / "processed" / "features_1h.parquet"
    latest_feature_ts = None
    try:
        import pandas as pd

        df = pd.read_parquet(features_path)
        if len(df.index) > 0:
            latest_feature_ts = str(df.index[-1])
    except Exception:
        pass

    input_context = {
        "invocation": {"trade_threshold": 0.45},
        "data_paths": {
            "trade_model_path": str(agent.trade_model_path),
            "direction_model_path": str(agent.direction_model_path),
            "features_path": str(agent.features_path),
        },
        "latest_feature_timestamp": latest_feature_ts,
        "expected_behavior": "Output buy/sell/hold with confidence, trade_probability, long_probability, and concise reasoning.",
    }
    return input_context, output


def _capture_sentiment() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    output = _run_subprocess_json(REPO_ROOT / "sentiment_analysis", "run_agent_json.py")
    input_context = {
        "invocation": {"days_back": 3},
        "entrypoint": "sentiment_analysis/run_agent_json.py",
        "expected_behavior": "Fetch recent crypto/news+macro context, output buy/sell/hold with confidence and reasoning.",
    }
    return input_context, _to_jsonable(output)


def _capture_risk() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    output = _run_subprocess_json(REPO_ROOT / "agent_risk", "run_agent_json.py")
    input_context = {
        "invocation": {"geopolitical_days_back": 14},
        "entrypoint": "agent_risk/run_agent_json.py",
        "expected_behavior": "Output low_risk/medium_risk/high_risk with confidence, risk scores, and reasoning.",
    }
    return input_context, _to_jsonable(output)


def _capture_coordinator() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from coordinator_agent.coordinator_core import CoordinatorAgent

    agent = CoordinatorAgent(repo_root=REPO_ROOT)
    output = _to_jsonable(agent.run(mode="agentic").model_dump())
    input_context = {
        "invocation": {"mode": "agentic"},
        "entrypoint": "coordinator_agent/coordinator_core.py::CoordinatorAgent.run",
        "depends_on": ["technical", "sentiment", "risk", "agentic_prototype.workflow"],
        "expected_behavior": "Fuse technical/sentiment/risk into a final trading signal with confidence and reasoning.",
    }
    return input_context, output


def _build_sample(agent_name: str, question: str, capture_fn) -> Dict[str, Any]:
    started = time.time()
    timestamp = _utc_now_iso()
    try:
        agent_input, agent_output = capture_fn()
        status = "ok"
        error = None
    except Exception as exc:
        agent_input = {}
        agent_output = {}
        status = "error"
        error = str(exc)

    duration = round(time.time() - started, 3)
    sample_id = f"{agent_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    sample = {
        "sample_id": sample_id,
        "task": "Evaluate if the agent output is good quality and if the agent appears to function correctly for its role.",
        "question": question,
        "agent_input": agent_input,
        "agent_prediction": agent_output,
        "final_agent_prompt": "Judge this agent by the appropriateness of its output for the provided input context and expected behavior.",
        "reference_answer": None,
        "metadata": {
            "agent_name": agent_name,
            "status": status,
            "error": error,
            "duration_seconds": duration,
            "timestamp_utc": timestamp,
        },
    }
    return sample


def build_payload() -> Dict[str, Any]:
    samples = [
        _build_sample(
            "technical",
            "Technical Agent: Given latest feature window and model outputs, assess whether signal/confidence/reasoning are coherent and useful.",
            _capture_technical,
        ),
        _build_sample(
            "sentiment",
            "Sentiment Agent: Given recent sentiment and macro sources, assess whether output signal/confidence/reasoning are plausible and aligned.",
            _capture_sentiment,
        ),
        _build_sample(
            "risk",
            "Risk Agent: Given on-chain/geopolitical or model-based risk inputs, assess whether risk level and explanation are coherent and useful.",
            _capture_risk,
        ),
        _build_sample(
            "coordinator",
            "Coordinator Agent: Given sub-agent signals, assess whether final fused decision is coherent, risk-aware, and operationally useful.",
            _capture_coordinator,
        ),
    ]

    return {
        "meta": {
            "schema": "llm_judge_agent_io_samples_v1",
            "generated_at_utc": _utc_now_iso(),
            "sample_count": len(samples),
            "notes": "Each sample contains actual live agent input context and output for direct LLM judging.",
        },
        "samples": samples,
    }


def main() -> int:
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {payload['meta']['sample_count']} agent IO samples to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

