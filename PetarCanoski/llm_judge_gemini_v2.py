
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def load_input(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_output(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_prompt_for_agent(agent_name: str, data: Dict[str, Any]) -> str:
    summary = json.dumps(data, ensure_ascii=False, indent=2)
    prompt = (
        f"You are an impartial judge. Given the following automatic evaluation summary for the '{agent_name}' agent, "
        "provide a numeric overall score from 0 (poor) to 100 (excellent), and a short justification. "
        "Return JSON only with keys: overall (int 0-100), reasoning (string).\n\n"
        f"Evaluation summary:\n{summary}\n\n"
        "Be concise (max 100 words). Return ONLY valid JSON."
    )
    return prompt


def call_gemini_v2(prompt: str, model: str = "gemini-2.0-flash", temperature: float = 0.0, max_tokens: int = 4096) -> str:
    """Call Gemini 2.0 using the new google-genai v2 SDK."""
    try:
        import google.genai as genai
    except ImportError:
        raise RuntimeError("Install google-genai: pip install google-genai") from None

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment")

    # v2 SDK client
    client = genai.Client(api_key=api_key)

    # Use the v2 SDK
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    # Extract text from response object
    text = resp.text if hasattr(resp, "text") else str(resp)
    return text.strip()


def heuristic_score(agent_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback heuristic scoring (for testing without LLM calls)."""
    out = {"agent": agent_name, "scores": {}, "reasoning": "heuristic"}

    if agent_name == "technical":
        acc = data.get("stage1", {}).get("accuracy", 0.0)
        f1_s2 = data.get("stage2", {}).get("report", {}).get("macro avg", {}).get("f1-score", 0.0)
        overall = (acc * 0.6 + f1_s2 * 0.4) * 100.0
        out["scores"]["overall"] = round(min(100, max(0, overall)), 2)
        out["reason"] = f"Accuracy={acc:.3f}, F1={f1_s2:.3f}"
    elif agent_name == "sentiment":
        acc = data.get("accuracy", 0.0)
        out["scores"]["overall"] = round(acc * 100.0, 2)
        out["reason"] = f"Accuracy={acc:.3f}"
    elif agent_name == "risk":
        acc = data.get("binary_classification", {}).get("report", {}).get("accuracy", 0.0)
        out["scores"]["overall"] = round(acc * 100.0, 2)
        out["reason"] = f"Binary accuracy={acc:.3f}"
    elif agent_name == "backtest":
        ta_pnl = data.get("ta_only", {}).get("cumulative_pnl_pct", 0.0)
        # normalize PnL to 0–100 range
        score = 50.0 + (min(1000.0, max(0.0, ta_pnl)) / 1000.0) * 45.0
        out["scores"]["overall"] = round(min(100, max(0, score)), 2)
        out["reason"] = f"TA PnL={ta_pnl:.2f}%"
    else:
        out["scores"]["overall"] = 50.0
        out["reason"] = "default"

    return out


def evaluate_with_gemini(data: Dict[str, Any], dry_run: bool = True, model: str = "gemini-2.0-flash") -> Dict[str, Any]:
    results = {"meta": {"mode": "dry-run" if dry_run else "gemini-2.0"}, "judgements": {}}
    targets = ["technical", "sentiment", "risk", "backtest"]

    for key in targets:
        if key not in data:
            continue

        if dry_run:
            results["judgements"][key] = heuristic_score(key, data[key])
        else:
            prompt = build_prompt_for_agent(key, data[key])
            try:
                content = call_gemini_v2(prompt, model=model)
                try:
                    parsed = json.loads(content)
                    results["judgements"][key] = parsed
                except json.JSONDecodeError:
                    results["judgements"][key] = {"raw": content}
            except Exception as e:
                results["judgements"][key] = {"error": str(e)}

    return results


def main(argv: list) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="evaluation_raw.json")
    p.add_argument("--out", dest="out_path", default="llm_judgement_gemini.json")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--model", dest="model", default="gemini-2.0-flash")
    args = p.parse_args(argv)

    data = load_input(args.in_path)
    dry = args.dry_run or (os.environ.get("GOOGLE_API_KEY") is None)

    if not dry:
        print(f"Calling Gemini v2 model {args.model}...")

    results = evaluate_with_gemini(data, dry_run=dry, model=args.model)
    save_output(args.out_path, results)
    print(f"Saved judgements to: {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


