from __future__ import annotations

import unittest
from pathlib import Path

from agentic_prototype.evaluate import evaluate, load_eval_set
from agentic_prototype.workflow import run_case


class AgenticPrototypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_path = Path(__file__).resolve().parent / "eval_set.jsonl"
        self.cases = load_eval_set(self.dataset_path)

    def test_single_case_output_schema(self) -> None:
        output = run_case(self.cases[0])
        self.assertIn(output["predicted_signal"], {"buy", "sell", "hold"})
        self.assertTrue(-1.0 <= output["predicted_score"] <= 1.0)
        self.assertTrue(0.0 <= output["predicted_confidence"] <= 1.0)

    def test_eval_accuracy_threshold(self) -> None:
        report = evaluate(self.cases)
        self.assertGreaterEqual(report["accuracy"], 0.80)


if __name__ == "__main__":
    unittest.main()

