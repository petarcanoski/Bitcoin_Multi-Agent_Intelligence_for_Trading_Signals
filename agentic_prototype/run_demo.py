from __future__ import annotations

import json
from pathlib import Path

from .workflow import run_case


def main() -> None:
    dataset_path = Path(__file__).resolve().parent / "eval_set.jsonl"
    with dataset_path.open("r", encoding="utf-8") as f:
        first_case = json.loads(f.readline())

    result = run_case(first_case)
    print("=" * 70)
    print("AGENTIC PROTOTYPE DEMO")
    print("=" * 70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

