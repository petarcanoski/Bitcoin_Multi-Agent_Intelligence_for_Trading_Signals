from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .workflow import run_case

SIGNALS = ("buy", "hold", "sell")


def load_eval_set(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate(cases: List[Dict]) -> Dict:
    predictions = [run_case(case) for case in cases]
    total = len(predictions)
    correct = sum(1 for p in predictions if p["predicted_signal"] == p["expected_signal"])
    accuracy = correct / total if total else 0.0

    confusion = {label: Counter() for label in SIGNALS}
    for p in predictions:
        confusion[p["expected_signal"]][p["predicted_signal"]] += 1

    per_class = {}
    for label in SIGNALS:
        tp = confusion[label][label]
        fp = sum(confusion[e][label] for e in SIGNALS if e != label)
        fn = sum(confusion[label][p] for p in SIGNALS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4)}

    return {
        "total_cases": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "confusion_matrix": {
            expected: {pred: confusion[expected][pred] for pred in SIGNALS} for expected in SIGNALS
        },
        "per_class": per_class,
        "predictions": predictions,
    }


def save_report(report: Dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"evaluation_{ts}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the LangGraph skill-based agentic prototype.")
    parser.add_argument(
        "--output-dir",
        default="agentic_prototype/reports",
        help="Directory where JSON evaluation reports are stored.",
    )
    args = parser.parse_args()

    dataset_path = Path(__file__).resolve().parent / "eval_set.jsonl"
    cases = load_eval_set(dataset_path)
    report = evaluate(cases)
    report_path = save_report(report, Path(__file__).resolve().parents[1] / args.output_dir)

    print("=" * 70)
    print("AGENTIC PROTOTYPE EVALUATION")
    print("=" * 70)
    print(f"Total cases : {report['total_cases']}")
    print(f"Correct     : {report['correct']}")
    print(f"Accuracy    : {report['accuracy']:.2%}")
    print("\nConfusion Matrix (expected -> predicted counts):")
    for expected, row in report["confusion_matrix"].items():
        print(f"  {expected:>4}: {row}")
    print("\nPer-class metrics:")
    for label, metric in report["per_class"].items():
        print(f"  {label:>4}: precision={metric['precision']:.2%}, recall={metric['recall']:.2%}")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()

