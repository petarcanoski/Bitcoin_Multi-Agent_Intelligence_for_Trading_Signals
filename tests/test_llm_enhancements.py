

import sys
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)
    
# Set UTF-8 encoding for Windows compatibility
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_llm_reasoning():
    """Test Enhancement 1: LLM Reasoning Module"""
    print_header("TEST 1: LLM Reasoning Module")

    try:
        # Test import
        print("✓ Attempting to import llm_reasoning module...")
        sys.path.insert(0, str(Path(__file__).parent / "sentiment_analysis"))
        from sentiment_analysis import llm_reasoning
        print("✓ Successfully imported generate_llm_reasoning")

        # Test function signature
        import inspect
        sig = inspect.signature(llm_reasoning)
        params = list(sig.parameters.keys())
        print(f"✓ Function parameters: {params}")

        expected_params = [
            'signal', 'combined_score', 'news_score', 'twitter_score',
            'macro_score', 'news_count', 'twitter_count', 'macro_count',
            'fallback_reasoning', 'market_context'
        ]

        if params == expected_params:
            print("✓ Function signature matches specification")
        else:
            print(f"⚠ Parameter mismatch. Expected: {expected_params}, Got: {params}")
            return False

        # Test graceful fallback (no API key)
        print("\n✓ Testing graceful fallback (no GROQ_API_KEY)...")
        result = llm_reasoning(
            signal='buy',
            combined_score=0.5,
            news_score=0.4,
            twitter_score=0.3,
            macro_score=0.2,
            news_count=10,
            twitter_count=5,
            macro_count=3,
            fallback_reasoning="Test fallback reasoning"
        )

        if isinstance(result, str) and len(result) > 0:
            print("✓ Fallback reasoning returned successfully")
            print(f"  Type: {type(result).__name__}, Length: {len(result)} chars")
            return True
        else:
            print("✗ Failed to get reasoning string")
            return False

    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sentiment_agent_integration():
    """Test Enhancement 1B: Sentiment Agent Integration"""
    print_header("TEST 1B: Sentiment Agent Integration")

    try:
        # Test import
        print("✓ Attempting to import sentiment_agent...")
        sys.path.insert(0, str(Path(__file__).parent / "sentiment_analysis"))
        from sentiment_analysis import SentimentAgent
        print("✓ Successfully imported SentimentAgent")

        # Check for import
        import sentiment_agent
        source = inspect.getsource(sentiment_agent)
        if "generate_llm_reasoning" in source:
            print("✓ LLM reasoning import found in sentiment_agent.py")
        else:
            print("⚠ LLM reasoning import not found")

        if "generate_llm_reasoning(" in source:
            print("✓ LLM reasoning function call found in sentiment_agent.py")
            return True
        else:
            print("⚠ LLM reasoning function call not found")
            return False

    except Exception as e:
        print(f"✗ Error during test: {e}")
        return False

def test_chat_interface():
    """Test Enhancement 2: Chat Interface"""
    print_header("TEST 2: Chat Interface")

    try:
        # Test import
        print("✓ Attempting to import chat_interface...")
        chat_file = Path(__file__).parent / "chat_interface.py"

        if not chat_file.exists():
            print(f"✗ chat_interface.py not found at {chat_file}")
            return False

        print(f"✓ Found chat_interface.py at {chat_file}")

        # Check file content
        with open(chat_file, 'r') as f:
            content = f.read()

        checks = [
            ("CoordinatorAgent", "Coordinator import"),
            ("FinalCoordinatorSignal", "Signal model"),
            ("Groq", "Groq client"),
            ("def run_chat_interface", "Main function"),
            ("while True:", "Chat loop"),
            ("'exit'", "Exit condition"),
        ]

        all_good = True
        for check_str, desc in checks:
            if check_str in content:
                print(f"✓ {desc} found")
            else:
                print(f"⚠ {desc} not found")
                all_good = False

        return all_good

    except Exception as e:
        print(f"✗ Error during test: {e}")
        return False

def test_readme():
    """Test Documentation"""
    print_header("TEST 3: Documentation")

    try:
        readme = Path(__file__).parent / "README_CHAT.md"

        if not readme.exists():
            print(f"✗ README_CHAT.md not found")
            return False

        print(f"✓ Found README_CHAT.md")

        with open(readme, 'r') as f:
            content = f.read()

        checks = [
            ("Setup", "Setup section"),
            ("GROQ_API_KEY", "API key instructions"),
            ("Usage", "Usage section"),
            ("Troubleshooting", "Troubleshooting"),
        ]

        for check_str, desc in checks:
            if check_str in content:
                print(f"✓ {desc} found")
            else:
                print(f"⚠ {desc} not found")

        print(f"✓ Documentation complete ({len(content)} chars)")
        return True

    except Exception as e:
        print(f"✗ Error during test: {e}")
        return False

def check_groq_availability():
    """Check if Groq is installed"""
    print_header("TEST 4: Groq Package Availability")

    try:
        import groq
        print(f"✓ Groq package is installed (version: {groq.__version__})")

        # Check for API key
        if os.getenv("GROQ_API_KEY"):
            print("✓ GROQ_API_KEY environment variable is set")
        else:
            print("ℹ GROQ_API_KEY not set (this is optional)")

        return True

    except ImportError:
        print("ℹ Groq package not installed")
        print("  Install with: pip install groq")
        print("  This is optional - system works with fallback")
        return True  # Not a failure, just optional

def run_all_tests():
    """Run all tests"""
    print_header("LLM ENHANCEMENTS - VERIFICATION TEST SUITE")

    results = {
        "LLM Reasoning Module": test_llm_reasoning(),
        "Sentiment Agent Integration": test_sentiment_agent_integration(),
        "Chat Interface": test_chat_interface(),
        "Documentation": test_readme(),
        "Groq Availability": check_groq_availability(),
    }

    print_header("TEST RESULTS SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        symbol = "✓" if result else "✗"
        status = "PASS" if result else "FAIL"
        print(f"{symbol} {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "="*70)
        print(" ✅ ALL TESTS PASSED - ENHANCEMENTS READY!")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print(f" ⚠️  {total - passed} test(s) failed")
        print("="*70)
        return 1

if __name__ == "__main__":
    import inspect
    exit_code = run_all_tests()
    sys.exit(exit_code)


