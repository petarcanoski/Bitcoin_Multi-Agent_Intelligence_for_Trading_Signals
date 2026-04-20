import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("Testing Risk Agent...")
print("=" * 60)

try:
    print("1. Importing config...")
    from config import RiskConfig
    print("    Config imported")

    print("2. Importing models...")
    from models import RiskSignalOutput, OnChainMetric
    print("    Models imported")

    print("3. Importing onchain clients...")
    from onchain_clients import OnChainAnalyzer
    print("    OnChain clients imported")

    print("4. Importing geopolitical clients...")
    from geopolitical_clients import GeopoliticalAnalyzer
    print("    Geopolitical clients imported")

    print("5. Importing risk agent...")
    from risk_agent import RiskAgent
    print("    Risk agent imported")

    print("\n6. Initializing agent...")
    agent = RiskAgent()
    print("    Agent initialized!")

    print("\n7. Running analysis...")
    signal = agent.run()
    print("    Analysis complete!")

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Signal: {signal.signal.upper()}")
    print(f"Risk Score: {signal.risk_score:.3f}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Key Risks: {len(signal.key_risks)}")
    print("=" * 60)
    print("\n ALL TESTS PASSED!")

except Exception as e:
    print(f"\n ERROR: {e}")
    import traceback
    traceback.print_exc()

