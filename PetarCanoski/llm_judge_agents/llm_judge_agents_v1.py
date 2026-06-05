from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "PetarCanoski" / "llm_judge_agents" /"evaluation_agents_raw_1.json"
DEFAULT_OUTPUT = REPO_ROOT / "PetarCanoski" / "llm_judge_agents" /"llm_judgement_agents_v3_live.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _extract_json_candidate(text: str) -> Optional[Any]:
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{\[]", cleaned):
        start = match.start()
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return obj
        except Exception:
            continue

    return None


def parse_judge_response(text: str) -> Dict[str, Any]:
    parsed = _extract_json_candidate(text)
    if isinstance(parsed, dict):
        parsed["parse_status"] = "ok"
        parsed["raw_response"] = text
        return parsed

    return {
        "parse_status": "failed",
        "parse_error": "json_not_found_or_incomplete",
        "raw_response": text,
    }


def _normalize_recommendations(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def normalize_single_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    score_value = parsed.get("score", parsed.get("overall", parsed.get("rating", 0)))
    try:
        score = int(round(float(str(score_value)))) if score_value is not None else 0
    except Exception:
        score = 0
    score = max(1, min(10, score))

    return {
        "parse_status": parsed.get("parse_status", "failed"),
        "score": score,
        "reasoning": str(parsed.get("reasoning", parsed.get("explanation", ""))).strip(),
        "recommendations": _normalize_recommendations(parsed.get("recommendations", parsed.get("improvements"))),
        "raw_response": parsed.get("raw_response"),
    }


def _call_gemini(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 4096) -> str:
    try:
        import google.genai as genai
    except Exception as exc:
        raise RuntimeError("Install google-genai to use live Gemini judging.") from exc

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    text = response.text if hasattr(response, "text") else str(response)
    return text.strip()


def build_judge_prompt(sample: Dict[str, Any]) -> str:
    payload = {
        "task": sample.get("task"),
        "question": sample.get("question"),
        "agent_input": sample.get("agent_input"),
        "agent_prediction": sample.get("agent_prediction"),
        "reference_answer": sample.get("reference_answer"),
        "metadata": sample.get("metadata", {}),
    }

    schema = {
        "score": "integer 1-10",
        "reasoning": "concise explanation",
        "recommendations": ["short actionable recommendation", "...", "..."],
    }

    return (
        "You are an impartial LLM-as-a-Judge for multi-agent trading systems.\n"
        "Evaluate this ONE agent using its actual input context and actual output.\n"
        "Score from 1 to 10 for correctness, relevance, usefulness, and whether behavior appears functionally sound for this agent role.\n"
        "If the sample status is error/failure, score low and explain the failure impact.\n"
        "Return STRICT JSON only matching this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Agent sample:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _heuristic_judge(sample: Dict[str, Any]) -> Dict[str, Any]:
    status = str(sample.get("metadata", {}).get("status", "ok")).lower()
    if status != "ok":
        return {
            "parse_status": "heuristic",
            "score": 2,
            "reasoning": "Agent execution failed, so output quality and functional correctness are poor for this run.",
            "recommendations": [
                "Fix runtime errors before evaluating model quality.",
                "Add retry and fallback handling for external dependencies.",
                "Log structured error diagnostics per agent run.",
            ],
            "raw_response": None,
        }

    return {
        "parse_status": "heuristic",
        "score": 6,
        "reasoning": "Agent executed successfully; assign neutral baseline without live LLM judging.",
        "recommendations": [
            "Run live LLM judging to get richer qualitative assessment.",
            "Track output stability across multiple runs.",
            "Add reference targets where possible for stronger evaluation.",
        ],
        "raw_response": None,
    }


def _extract_retry_delay_seconds(msg: str) -> Optional[int]:
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", msg, flags=re.IGNORECASE)
    if not match:
        return None
    return max(1, int(float(match.group(1))) + 1)


def judge_one_sample(sample: Dict[str, Any], model: str, dry_run: bool, max_attempts: int) -> Dict[str, Any]:
    prompt = build_judge_prompt(sample)
    if dry_run or os.environ.get("GOOGLE_API_KEY") is None:
        return {"judge": _heuristic_judge(sample), "prompt": prompt}

    retry_prompt = (
        "Your previous answer was invalid or incomplete. "
        "Return a COMPLETE and VALID JSON object only, matching the required schema, with no extra text."
    )

    last_parsed: Dict[str, Any] = {"parse_status": "failed", "parse_error": "no_attempt"}
    for attempt in range(1, max_attempts + 1):
        current_prompt = prompt if attempt == 1 else f"{prompt}\n\n{retry_prompt}"
        try:
            text = _call_gemini(current_prompt, model=model)
        except Exception as exc:
            err_msg = str(exc)
            delay = _extract_retry_delay_seconds(err_msg)
            if attempt < max_attempts:
                if delay is not None:
                    print(f"Rate-limit detected. Sleeping {delay}s before retry...")
                    time.sleep(delay)
                    continue

                # Generic transient retry path (e.g., 503 high demand).
                backoff = min(30, 3 * attempt)
                print(f"Transient Gemini error: {err_msg[:140]}... Retrying in {backoff}s.")
                time.sleep(backoff)
                continue
            raise

        parsed = parse_judge_response(text)
        normalized = normalize_single_result(parsed)
        last_parsed = normalized

        if normalized.get("parse_status") == "ok" and normalized.get("reasoning"):
            return {"judge": normalized, "prompt": current_prompt}

        if attempt < max_attempts:
            time.sleep(2)

    return {"judge": last_parsed, "prompt": prompt}


def aggregate_by_agent(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        agent = str(s.get("metadata", {}).get("agent_name", "unknown"))
        grouped[agent].append(s)

    out: Dict[str, Any] = {}
    for agent, rows in grouped.items():
        scores: List[float] = []
        ok_count = 0
        for row in rows:
            judge = row.get("judge", {})
            if judge.get("parse_status") == "ok":
                ok_count += 1
            sc = judge.get("score")
            try:
                if sc is not None:
                    scores.append(float(sc))
            except Exception:
                pass

        out[agent] = {
            "sample_count": len(rows),
            "llm_parse_ok_count": ok_count,
            "average_score": round(sum(scores) / len(scores), 3) if scores else None,
            "min_score": round(min(scores), 3) if scores else None,
            "max_score": round(max(scores), 3) if scores else None,
        }
    return out


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="LLM judge for 4-agent input/output live samples.")
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", dest="model", default="gemini-2.5-flash")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--max-attempts", dest="max_attempts", type=int, default=3)
    parser.add_argument("--sleep-after", dest="sleep_after", type=int, default=4, help="Sleep after N live requests")
    parser.add_argument("--sleep-seconds", dest="sleep_seconds", type=int, default=65, help="Sleep duration for quota protection")
    args = parser.parse_args(argv)

    data = load_json(args.in_path)
    samples_in = list(data.get("samples", []))
    meta = dict(data.get("meta", {}))

    dry = args.dry_run or os.environ.get("GOOGLE_API_KEY") is None

    judged: List[Dict[str, Any]] = []
    live_counter = 0
    for sample in samples_in:
        if not dry and live_counter and (live_counter % max(1, args.sleep_after) == 0):
            print(f"Throttling: sleeping {args.sleep_seconds}s to reduce quota errors...")
            time.sleep(args.sleep_seconds)

        result = judge_one_sample(sample, model=args.model, dry_run=dry, max_attempts=max(1, args.max_attempts))
        if not dry:
            live_counter += 1

        judged.append(
            {
                "sample_id": sample.get("sample_id"),
                "task": sample.get("task"),
                "question": sample.get("question"),
                "metadata": sample.get("metadata", {}),
                "input": sample,
                "judge": result["judge"],
                "judge_prompt": result["prompt"],
            }
        )

    output = {
        "meta": {
            **meta,
            "generated_at_utc": _utc_now_iso(),
            "mode": "dry-run" if dry else args.model,
            "input_path": str(Path(args.in_path).resolve()),
            "output_path": str(Path(args.out_path).resolve()),
        },
        "samples": judged,
        "aggregate_by_agent": aggregate_by_agent(judged),
    }

    save_json(args.out_path, output)
    print(f"Saved agent-IO judge report to: {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



