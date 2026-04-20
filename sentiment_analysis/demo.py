from sentiment_agent import SentimentAgent


def demo_basic_usage():
    """Basic usage of the sentiment agent"""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Usage")
    print("=" * 70)

    agent = SentimentAgent()
    signal = agent.run(days_back=3)

    print(f"\n Signal: {signal.signal.upper()}")
    print(f" Confidence: {signal.confidence:.2%}")
    print(f" Reasoning: {signal.reasoning[:100]}...")


def demo_json_output():
    """Show JSON output for coordinator agent"""
    print("\n" + "=" * 70)
    print("DEMO 2: JSON Output for Coordinator Agent")
    print("=" * 70)

    agent = SentimentAgent()
    signal = agent.run(days_back=5)

    # Export as JSON
    json_output = signal.model_dump_json(indent=2)
    print("\n JSON Output:")
    print(json_output)

    # You can also parse it
    signal_dict = signal.model_dump()
    print(f"\n As Dictionary:")
    print(f"Signal Type: {signal_dict['agent_type']}")
    print(f"Trading Signal: {signal_dict['signal']}")
    print(f"Confidence: {signal_dict['confidence']}")


def demo_component_breakdown():
    print("\n" + "=" * 70)
    print("DEMO 3: Component Breakdown")
    print("=" * 70)

    agent = SentimentAgent()
    signal = agent.run(days_back=7)

    print(f"\n News Component:")
    print(f"   Score: {signal.news_sentiment:+.3f}")
    print(f"   Weight: 40%")
    print(f"   Contribution: {signal.news_sentiment * 0.4:+.3f}")

    print(f"\n Twitter/X Component:")
    print(f"   Score: {signal.twitter_sentiment:+.3f}")
    print(f"   Weight: 30%")
    print(f"   Contribution: {signal.twitter_sentiment * 0.3:+.3f}")

    print(f"\n Macro Component:")
    print(f"   Score: {signal.macro_score:+.3f}")
    print(f"   Weight: 30%")
    print(f"   Contribution: {signal.macro_score * 0.3:+.3f}")

    print(f"\n Combined Score: {signal.sentiment_score:+.3f}")

    print(f"\n Key Factors:")
    for factor in signal.key_factors:
        print(f"   • {factor}")


def demo_integration_example():
    """Show how coordinator might use this agent"""
    print("\n" + "=" * 70)
    print("DEMO 4: Multi-Agent Integration Example")
    print("=" * 70)

    # Get sentiment signal
    sentiment_agent = SentimentAgent()
    sentiment_signal = sentiment_agent.run()

    print("\n Sentiment Agent Output:")
    print(f"   Signal: {sentiment_signal.signal}")
    print(f"   Confidence: {sentiment_signal.confidence:.2f}")

    # Simulate other agents (these would be real in your full system)
    print("\n Technical Agent Output: (simulated)")
    technical_signal = "buy"
    technical_confidence = 0.75
    print(f"   Signal: {technical_signal}")
    print(f"   Confidence: {technical_confidence:.2f}")

    print("\n Risk Agent Output: (simulated)")
    risk_signal = "hold"
    risk_confidence = 0.65
    print(f"   Signal: {risk_signal}")
    print(f"   Confidence: {risk_confidence:.2f}")

    # Simple coordinator logic
    print("\n Coordinator Decision:")

    # Weight each agent by confidence
    signal_values = {
        'buy': 1.0,
        'hold': 0.0,
        'sell': -1.0
    }

    weighted_score = (
        signal_values[sentiment_signal.signal] * sentiment_signal.confidence * 0.4 +
        signal_values[technical_signal] * technical_confidence * 0.4 +
        signal_values[risk_signal] * risk_confidence * 0.2
    )

    if weighted_score > 0.3:
        final_signal = "BUY"
    elif weighted_score < -0.3:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    print(f"   Weighted Score: {weighted_score:+.3f}")
    print(f"   Final Signal: {final_signal}")
    print(f"\n   Reasoning:")
    print(f"   - Sentiment suggests: {sentiment_signal.signal}")
    print(f"   - Technical suggests: {technical_signal}")
    print(f"   - Risk suggests: {risk_signal}")
    print(f"   - Combined analysis suggests: {final_signal}")


def demo_custom_analysis():
    """Show how to customize analysis parameters"""
    print("\n" + "=" * 70)
    print("DEMO 5: Custom Analysis Period")
    print("=" * 70)

    agent = SentimentAgent()

    # Try different time periods
    for days in [1, 3, 7]:
        signal = agent.run(days_back=days)
        print(f"\n Last {days} day(s):")
        print(f"   Signal: {signal.signal.upper()} (confidence: {signal.confidence:.2%})")
        print(f"   Sentiment: {signal.sentiment_score:+.3f}")


def main():
    """Run all demos"""
    print(" SENTIMENT AGENT DEMO SUITE")
    print("=" * 70)

    try:
        demo_basic_usage()
        demo_json_output()
        demo_component_breakdown()
        demo_integration_example()
        demo_custom_analysis()

        print("\n" + "=" * 70)
        print(" All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

