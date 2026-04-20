from pathlib import Path

try:
    from coordinator_agent import CoordinatorAgent
except ImportError:
    from coordinator_core import CoordinatorAgent


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    agent = CoordinatorAgent(repo_root=repo_root)
    result = agent.run()

    assert result.signal in {"buy", "sell", "hold"}
    assert 0.0 <= result.confidence <= 1.0
    assert -1.0 <= result.score <= 1.0
    print("Smoke test passed.")
    print(result.model_dump_json())


if __name__ == "__main__":
    main()


