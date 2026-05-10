"""
LLM-Powered Reasoning Module for Sentiment Agent

Uses Groq API with llama3-8b-8192 model to generate intelligent reasoning
for trading signals. Includes graceful fallback to basic reasoning on API errors.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def generate_llm_reasoning(
    signal: str,
    combined_score: float,
    news_score: float,
    twitter_score: float,
    macro_score: float,
    news_count: int,
    twitter_count: int,
    macro_count: int,
    fallback_reasoning: str,
    market_context: dict = None
) -> str:
    """
    Generate intelligent reasoning using Groq LLM.

    Args:
        signal: Trading signal ("buy", "sell", or "hold")
        combined_score: Combined sentiment score (-1 to 1)
        news_score: News sentiment component (-1 to 1)
        twitter_score: Twitter sentiment component (-1 to 1)
        macro_score: Macro analysis component (-1 to 1)
        news_count: Number of news articles analyzed
        twitter_count: Number of tweets analyzed
        macro_count: Number of macro indicators analyzed
        fallback_reasoning: Basic reasoning to use if LLM call fails
        market_context: Optional dict with market data

    Returns:
        LLM-generated reasoning string or fallback if unavailable
    """

    try:
        # Lazy import - only import if used
        from groq import Groq
    except ImportError:
        logger.warning("Groq package not installed. Falling back to basic reasoning.")
        return fallback_reasoning

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set. Falling back to basic reasoning.")
        return fallback_reasoning

    try:
        # Initialize Groq client
        client = Groq(api_key=api_key)

        # Build context from data
        context_parts = [
            f"News sentiment analysis: {news_count} articles analyzed, score: {news_score:.3f}",
            f"Twitter/X sentiment analysis: {twitter_count} tweets analyzed, score: {twitter_score:.3f}",
            f"Macroeconomic indicators: {macro_count} factors analyzed, score: {macro_score:.3f}",
            f"Combined weighted sentiment score: {combined_score:.3f}",
        ]

        if market_context:
            price_change = market_context.get('price_change_24h', 0)
            if price_change:
                context_parts.append(f"24-hour Bitcoin price change: {price_change:+.2f}%")

            sentiment_pct = market_context.get('sentiment_up_percentage', 50)
            if sentiment_pct:
                context_parts.append(f"Community bullish sentiment: {sentiment_pct:.0f}%")

        context_str = "\n".join(context_parts)

        system_prompt = """You are an expert cryptocurrency trading analyst. 
Your task is to provide concise, intelligent reasoning for trading signals.
Generate a 3-5 sentence explanation that:
1. Clearly explains the signal (BUY, SELL, or HOLD)
2. Identifies the dominant factors driving this signal
3. Highlights what traders should be aware of
4. Is direct and actionable, not verbose

Keep technical jargon minimal and focus on clarity."""

        user_prompt = f"""Based on the following sentiment analysis data, explain this {signal.upper()} signal:

{context_str}

Provide clear, actionable reasoning for this signal."""

        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=256,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        reasoning = response.choices[0].message.content.strip()
        logger.info(f"LLM reasoning generated successfully ({len(reasoning)} chars)")
        return reasoning

    except Exception as e:
        logger.warning(f"LLM reasoning failed: {type(e).__name__}: {str(e)}. Using fallback.")
        return fallback_reasoning

