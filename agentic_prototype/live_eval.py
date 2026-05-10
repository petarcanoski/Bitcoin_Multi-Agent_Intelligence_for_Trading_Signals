from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List

from coordinator_agent.coordinator_core import CoordinatorAgent


def run_live_benchmark(repo_root: Path, runs: int, interval_seconds: float) -> Dict:
    agent = CoordinatorAgent(repo_root=repo_root)
    entries: List[Dict] = []

    for idx in range(1, runs + 1):
        started_at = datetime.now(timezone.utc).isoformat()
        legacy = agent.run(mode="legacy")
        agentic = agent.run(mode="agentic")

        entries.append(
            {
                "run": idx,
                "started_at_utc": started_at,
                "legacy": legacy.model_dump(mode="json"),
                "agentic": agentic.model_dump(mode="json"),
                "signal_match": legacy.signal == agentic.signal,
                "score_delta": round(abs(legacy.score - agentic.score), 6),
                "confidence_delta": round(abs(legacy.confidence - agentic.confidence), 6),
            }
        )

        if idx < runs and interval_seconds > 0:
            time.sleep(interval_seconds)

    agreement = mean(1.0 if row["signal_match"] else 0.0 for row in entries) if entries else 0.0
    avg_score_delta = mean(row["score_delta"] for row in entries) if entries else 0.0
    avg_conf_delta = mean(row["confidence_delta"] for row in entries) if entries else 0.0

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "interval_seconds": interval_seconds,
        "agreement_rate": round(agreement, 4),
        "avg_abs_score_delta": round(avg_score_delta, 4),
        "avg_abs_confidence_delta": round(avg_conf_delta, 4),
        "entries": entries,
    }


def save_report(report: Dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"live_eval_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark agentic coordinator against legacy coordinator on live agent outputs.")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs.")
    parser.add_argument("--interval-seconds", type=float, default=1.0, help="Pause between runs.")
    parser.add_argument(
        "--output-dir",
        default="agentic_prototype/reports",
        help="Directory for JSON benchmark artifacts.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    report = run_live_benchmark(repo_root=repo_root, runs=args.runs, interval_seconds=args.interval_seconds)
    report_path = save_report(report, repo_root / args.output_dir)

    print("=" * 70)
    print("AGENTIC LIVE BENCHMARK")
    print("=" * 70)
    print(f"Runs                  : {report['runs']}")
    print(f"Signal agreement      : {report['agreement_rate']:.2%}")
    print(f"Avg |score delta|     : {report['avg_abs_score_delta']:.4f}")
    print(f"Avg |confidence delta|: {report['avg_abs_confidence_delta']:.4f}")
    print(f"Report saved          : {report_path}")


if __name__ == "__main__":
    main()

