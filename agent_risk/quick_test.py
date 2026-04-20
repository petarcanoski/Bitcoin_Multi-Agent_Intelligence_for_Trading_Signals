print("=" * 60)
print("Testing Risk Agent with REAL DATA")
print("=" * 60)

# Test 1: Check if .env file exists
print("\n1. Checking .env file...")
import os
if os.path.exists('.env'):
    print("   .env file found!")
else:
    print("   .env file not found")

# Test 2: Load config
print("\n2. Loading configuration...")
try:
    from config import RiskConfig
    if RiskConfig.NEWSAPI_KEY:
        print(f"   NewsAPI key loaded: {RiskConfig.NEWSAPI_KEY[:10]}...")
    else:
        print("    NewsAPI key empty (will use mock data)")
    print("   Configuration loaded!")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Test Blockchain.info (FREE, no key needed)
print("\n3. Testing Blockchain.info (FREE - no API key needed)...")
try:
    from onchain_clients import BlockchainInfoClient
    client = BlockchainInfoClient()
    stats = client.fetch_network_stats()
    print(f"    Hash rate: {stats['hash_rate']:,.0f}")
    print(f"    BTC price: ${stats['market_price_usd']:,.0f}")
    print(f"    Volume 24h: ${stats['trade_volume_24h']:,.0f}")
    print("    Real blockchain data working!")
except Exception as e:
    print(f"    Error: {e}")

# Test 4: Test NewsAPI (FREE with key)
print("\n4. Testing NewsAPI (FREE with your key)...")
try:
    from geopolitical_clients import GeopoliticalAnalyzer
    analyzer = GeopoliticalAnalyzer()
    events = analyzer.fetch_geopolitical_events(days_back=7)
    print(f"    Fetched {len(events)} geopolitical events")
    if events:
        print(f"    Latest: {events[0].title[:50]}...")
    print("    Real geopolitical data working!")
except Exception as e:
    print(f"  ️  Using mock data (expected if no key): {e}")

# Test 5: Run full agent
print("\n5. Running full Risk Agent...")
try:
    from risk_agent import RiskAgent
    agent = RiskAgent()
    signal = agent.run()
    print(f"    Signal: {signal.signal.upper()}")
    print(f"    Risk Score: {signal.risk_score:.3f}")
    print(f"    Confidence: {signal.confidence:.2%}")
    print("    Agent completed successfully!")
except Exception as e:
    print(f"    Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print(" TEST COMPLETE!")
print("=" * 60)

