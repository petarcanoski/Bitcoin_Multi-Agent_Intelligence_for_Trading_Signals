from pathlib import Path

try:
    from coordinator_agent import CoordinatorAgent
except ImportError:
    from coordinator_core import CoordinatorAgent


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    agent = CoordinatorAgent(repo_root=repo_root)
    result = agent.run()

    print("=" * 70)
    print("BITCOIN COORDINATOR AGENT")
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


