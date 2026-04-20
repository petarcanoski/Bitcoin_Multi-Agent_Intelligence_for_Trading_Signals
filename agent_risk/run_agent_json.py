from risk_agent import RiskAgent


def main() -> None:
    agent = RiskAgent()
    result = agent.run(geopolitical_days_back=14)
    print(result.model_dump_json())


if __name__ == "__main__":
    main()

