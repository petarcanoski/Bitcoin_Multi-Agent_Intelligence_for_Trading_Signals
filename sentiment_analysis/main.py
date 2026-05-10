try:
    import env_loader
except Exception:
    print("Warning: env_loader failed to import. Make sure .env file is present and env_loader.py is in the path.")

from sentiment_agent import SentimentAgent


def main():
    print(" Starting Bitcoin Sentiment Analysis Agent")
    print("=" * 70)

    agent = SentimentAgent()
    signal = agent.run(days_back=3)

    # Display results
    print("\n" + "=" * 70)
    print(" SENTIMENT AGENT RESULTS")
    print("=" * 70)
    print(f"\n Signal: {signal.signal.upper()}")
    print(f" Confidence: {signal.confidence:.2%}")
    print(f" Sentiment Score: {signal.sentiment_score:+.3f}")
    print(f" News Sentiment: {signal.news_sentiment:+.3f}")
    print(f" Twitter/X Sentiment: {signal.twitter_sentiment:+.3f}")
    print(f" Macro Score: {signal.macro_score:+.3f}")

    print(f"\n Key Factors:")
    for i, factor in enumerate(signal.key_factors, 1):
        print(f"   {i}. {factor}")

    print(f"\n Reasoning:")
    print(f"   {signal.reasoning}")

    print(f"\n Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Data Sources: {', '.join(signal.data_sources)}")

    print("\n" + "=" * 70)
    print(" Analysis Complete!")
    print("=" * 70)

    return signal


if __name__ == "__main__":
    main()

