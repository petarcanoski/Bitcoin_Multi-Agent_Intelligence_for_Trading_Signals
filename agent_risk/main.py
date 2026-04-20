from risk_agent import RiskAgent


def main():

    print("\n" + "=" * 70)
    print("Bitcoin Risk & Volatility Analysis Agent")
    print("=" * 70)

    # Initialize agent
    agent = RiskAgent()

    # Run analysis
    signal = agent.run(geopolitical_days_back=14)

    # Display results
    print("\n" + "=" * 70)
    print("RISK AGENT RESULTS")
    print("=" * 70)

    print(f"\nSignal: {signal.signal.upper()}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Overall Risk Score: {signal.risk_score:.3f}")
    print(f"On-Chain Risk: {signal.onchain_risk:.3f}")
    print(f"Geopolitical Risk: {signal.geopolitical_risk:.3f}")
    print(f"Volatility Score: {signal.volatility_score:.3f}")

    print(f"\n  Key Risks:")
    for i, risk in enumerate(signal.key_risks, 1):
        print(f"   {i}. {risk}")

    print(f"\n Opportunities:")
    for i, opp in enumerate(signal.key_opportunities, 1):
        print(f"   {i}. {opp}")

    print(f"\n Reasoning:")
    print(f"   {signal.reasoning}")

    print(f"\n Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📡 Data Sources: {', '.join(signal.data_sources)}")

    print("\n" + "=" * 70)
    print(" Analysis Complete!")
    print("=" * 70)

    # Optional: Show detailed metrics
    print(f"\n Detailed On-Chain Metrics ({len(signal.onchain_metrics)}):")
    for metric in signal.onchain_metrics:
        print(f"   • {metric.metric_type}: {metric.value:,.0f} (risk: {metric.risk_level})")

    print(f"\n Geopolitical Events ({len(signal.geopolitical_events)}):")
    for event in signal.geopolitical_events[:5]:  # Show top 5
        print(f"   • [{event.severity.upper()}] {event.title[:60]}...")
        print(f"     Region: {event.region}, Impact: {event.impact_on_crypto}")

    # Export as JSON for coordinator agent
    print(f"\n JSON Output for Coordinator:")
    print(signal.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

