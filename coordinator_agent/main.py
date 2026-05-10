try:
    import env_loader  # ensure root .env is loaded
except Exception:
    print("Warning: env_loader failed to import. Make sure .env file is present and env_loader.py is in the path.")

import argparse
from pathlib import Path

try:
    from coordinator_agent import CoordinatorAgent
except ImportError:
    from coordinator_core import CoordinatorAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Bitcoin coordinator agent.")
    parser.add_argument(
        "--mode",
        choices=["agentic", "legacy"],
        default="agentic",
        help="Execution mode. 'agentic' uses LangGraph+skills fusion, 'legacy' uses rule-based fusion.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    agent = CoordinatorAgent(repo_root=repo_root)
    result = agent.run(mode=args.mode)

    print("=" * 70)
    print(f"BITCOIN COORDINATOR AGENT ({args.mode.upper()} MODE)")
    print("=" * 70)
    print(f"Signal      : {result.signal.upper()}")
    print(f"Confidence  : {result.confidence:.2%}")
    print(f"Score       : {result.score:+.3f}")
    print(f"Risk Level  : {result.risk_level.upper()}")
    print("\nKey factors:")
    for idx, factor in enumerate(result.key_factors, 1):
        print(f"  {idx}. {factor}")
    print("\nReasoning:")
    print(result.reasoning)
    print("\nJSON:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()


