from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt


SIGNALS = ["buy", "hold", "sell"]


def load_latest_json(report_dir: Path, prefix: str) -> Optional[Dict]:
    files = sorted(report_dir.glob(f"{prefix}_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def plot_confusion_matrix(report: Dict, out_path: Path) -> None:
    matrix = [[report["confusion_matrix"][exp][pred] for pred in SIGNALS] for exp in SIGNALS]

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(SIGNALS)), SIGNALS)
    ax.set_yticks(range(len(SIGNALS)), SIGNALS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    ax.set_title("Agentic Prototype Confusion Matrix")

    for i in range(len(SIGNALS)):
        for j in range(len(SIGNALS)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_class(report: Dict, out_path: Path) -> None:
    precision = [report["per_class"][label]["precision"] for label in SIGNALS]
    recall = [report["per_class"][label]["recall"] for label in SIGNALS]

    x = range(len(SIGNALS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar([i - width / 2 for i in x], precision, width=width, label="Precision")
    ax.bar([i + width / 2 for i in x], recall, width=width, label="Recall")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(list(x), SIGNALS)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_live_benchmark(live_report: Dict, out_path: Path) -> None:
    labels = ["Agreement", "1-ScoreDelta", "1-ConfDelta"]
    values = [
        float(live_report.get("agreement_rate", 0.0)),
        max(0.0, 1.0 - float(live_report.get("avg_abs_score_delta", 1.0))),
        max(0.0, 1.0 - float(live_report.get("avg_abs_confidence_delta", 1.0))),
    ]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized score")
    ax.set_title("Live Benchmark Consistency")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_markdown(eval_report: Dict, live_report: Optional[Dict]) -> str:
    accuracy = float(eval_report["accuracy"])
    lines = [
        "# 05 - Agentic Prototype Evaluation old_Reports",
        "",
        "## Objective",
        "",
        "Evaluate the LangGraph + skill-based multi-agent prototype and compare behavior with the legacy coordinator.",
        "",
        "## Methodology",
        "",
        "- Offline labeled set: `agentic_prototype/eval_set.jsonl` (12 scenarios)",
        "- Online benchmark: live runs comparing `legacy` vs `agentic` coordinator outputs",
        "- Main metrics: accuracy, confusion matrix, precision/recall, agreement rate",
        "",
        "## Offline Results",
        "",
        f"- Total cases: **{eval_report['total_cases']}**",
        f"- Correct predictions: **{eval_report['correct']}**",
        f"- Accuracy: **{accuracy:.2%}**",
        "",
        "![Confusion Matrix](assets/agentic_confusion_matrix.png)",
        "",
        "![Per-Class Metrics](assets/agentic_per_class_metrics.png)",
        "",
    ]

    if live_report:
        lines.extend(
            [
                "## Live Benchmark (Real Agent Outputs)",
                "",
                f"- Runs: **{live_report['runs']}**",
                f"- Signal agreement (agentic vs legacy): **{float(live_report['agreement_rate']):.2%}**",
                f"- Avg absolute score delta: **{float(live_report['avg_abs_score_delta']):.4f}**",
                f"- Avg absolute confidence delta: **{float(live_report['avg_abs_confidence_delta']):.4f}**",
                "",
                "![Live Benchmark](assets/agentic_live_benchmark.png)",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The agentic coordinator keeps high directional quality on the labeled set.",
            "- The one sell->hold miss indicates conservative behavior in borderline bearish cases.",
            "- Live benchmark highlights where agentic fusion agrees or diverges from legacy heuristics.",
            "",
            "## Next Improvements",
            "",
            "1. Increase labeled set size with true historical outcomes.",
            "2. Calibrate thresholds per market regime (trend vs range).",
            "3. Add rolling weekly benchmark reports for presentation updates.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report_dir = repo_root / "agentic_prototype" / "reports"
    eval_report = load_latest_json(report_dir, "evaluation")
    if not eval_report:
        raise FileNotFoundError("No evaluation report found. Run 'python -m agentic_prototype.evaluate' first.")

    live_report = load_latest_json(report_dir, "live_eval")

    assets_dir = Path(__file__).resolve().parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(eval_report, assets_dir / "agentic_confusion_matrix.png")
    plot_per_class(eval_report, assets_dir / "agentic_per_class_metrics.png")
    if live_report:
        plot_live_benchmark(live_report, assets_dir / "agentic_live_benchmark.png")

    report_md = build_markdown(eval_report, live_report)
    report_path = Path(__file__).resolve().parent / "05_Agentic_Evaluation_Report.md"
    report_path.write_text(report_md, encoding="utf-8")

    print("Generated assets and markdown report:")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()


