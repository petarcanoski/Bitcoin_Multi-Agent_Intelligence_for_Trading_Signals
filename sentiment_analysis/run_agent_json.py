from sentiment_agent import SentimentAgent


def main() -> None:
    agent = SentimentAgent()
    result = agent.run(days_back=3)
    print(result.model_dump_json())


if __name__ == "__main__":
    main()

