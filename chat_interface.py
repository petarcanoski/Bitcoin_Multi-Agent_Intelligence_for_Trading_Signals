try:
    import env_loader
except Exception:
    print("Warning: env_loader failed to import. Make sure you have a .env file with GROQ_API_KEY if you want LLM functionality.")

import json
import logging
import os
import pickle
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DOCS_ROOT_NAMES = ("md", "Presentation")
CHAT_HISTORY_SUMMARY_THRESHOLD = 10
CHAT_HISTORY_KEEP_LAST = 4
SUMMARY_MAX_CHARS = 1200
CACHE_MAX_CHARS = 1400
TFIDF_SNIPPET_CHARS = 900
TFIDF_CHUNK_CHARS = 1200
TFIDF_CHUNK_OVERLAP = 180
OUT_OF_SCOPE_REFUSAL = (
    "That question is outside the Bitcoin decision system context. "
    "Please ask about the signal, supporting evidence, agents, retrieved docs, or the app workflow."
)

DOMAIN_KEYWORDS = {
    "bitcoin", "btc", "signal", "risk", "technical", "sentiment", "macro", "on-chain",
    "coordinator", "agent", "workflow", "analysis", "trading", "market", "docs", "document",
    "retrieval", "rag", "history", "memory", "prompt", "reasoning", "confidence", "score",
    "price", "volume", "volatility", "trend", "buy", "sell", "hold", "long", "short",
}

_DOC_INDEX = {
    "ready": False,
    "error": None,
    "items": [],
    "vectorizer": None,
    "matrix": None,
}


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_text(text: str, max_chars: int) -> str:
    """Trim text to a manageable length."""
    text = _normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _safe_get(obj, key: str, default=""):
    """Safely read attributes or dict keys from signal objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _format_key_factors(signal_obj) -> str:
    factors = _safe_get(signal_obj, "key_factors", []) or []
    if not isinstance(factors, (list, tuple)):
        factors = [str(factors)]
    return "\n".join(f"- {factor}" for factor in factors) or "- None provided"


def _format_history_for_summary(history: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for index, message in enumerate(history, start=1):
        role = str(message.get("role", "unknown")).upper()
        content = _truncate_text(message.get("content", ""), 240)
        if content:
            lines.append(f"{index}. {role}: {content}")
    return "\n".join(lines).strip()


def _local_history_summary(history: List[Dict[str, str]], existing_memory: str = "") -> str:
    """Create a concise local fallback summary when LLM summarization is unavailable."""
    pieces: List[str] = []
    if existing_memory.strip():
        pieces.append("Previous memory:")
        pieces.append(_truncate_text(existing_memory.strip(), 500))

    if history:
        pieces.append("Recent conversation:")
        for message in history:
            role = str(message.get("role", "unknown")).upper()
            content = _truncate_text(message.get("content", ""), 200)
            if content:
                pieces.append(f"- {role}: {content}")

    return _truncate_text("\n".join(pieces), SUMMARY_MAX_CHARS)


def _chunk_text(text: str, max_chars: int = TFIDF_CHUNK_CHARS, overlap: int = TFIDF_CHUNK_OVERLAP) -> List[str]:
    """Split long markdown into overlapping chunks for retrieval."""
    text = _normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        start = 0
        step = max(1, max_chars - overlap)
        while start < len(paragraph):
            chunks.append(paragraph[start : start + max_chars])
            if start + max_chars >= len(paragraph):
                current = ""
                break
            start += step
        else:
            current = ""

    if current:
        chunks.append(current)
    return chunks


def load_local_docs() -> List[Dict[str, str]]:
    """Read all markdown files from `md/` and `Presentation/` into retrievable chunks."""
    repo_root = Path(__file__).parent
    docs: List[Dict[str, str]] = []

    for folder_name in DOCS_ROOT_NAMES:
        folder = repo_root / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                logger.warning("Failed to read markdown file %s: %s", path, exc)
                continue

            for chunk_index, chunk in enumerate(_chunk_text(text), start=1):
                docs.append(
                    {
                        "path": str(path.relative_to(repo_root)),
                        "chunk_index": str(chunk_index),
                        "text": chunk,
                    }
                )

    return docs


def _ensure_doc_index() -> None:
    """Build the TF-IDF index once on demand."""
    if _DOC_INDEX["ready"]:
        return

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401  # Imported for parity with retrieval usage
    except ImportError as exc:
        _DOC_INDEX["error"] = exc
        _DOC_INDEX["ready"] = True
        logger.warning("scikit-learn is not available; local document retrieval is disabled.")
        return

    docs = load_local_docs()
    _DOC_INDEX["items"] = docs

    if not docs:
        _DOC_INDEX["vectorizer"] = None
        _DOC_INDEX["matrix"] = None
        _DOC_INDEX["ready"] = True
        return

    corpus = [item["text"] for item in docs]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=6000)
    matrix = vectorizer.fit_transform(corpus)

    _DOC_INDEX["vectorizer"] = vectorizer
    _DOC_INDEX["matrix"] = matrix
    _DOC_INDEX["ready"] = True


def retrieve_docs(query: str, top_k: int = 3) -> str:
    """Return the most relevant local markdown snippets for a query."""
    query = _normalize_whitespace(query)
    if not query:
        return ""

    _ensure_doc_index()
    if _DOC_INDEX["error"] is not None or _DOC_INDEX["vectorizer"] is None or _DOC_INDEX["matrix"] is None:
        return ""

    vectorizer = _DOC_INDEX["vectorizer"]
    matrix = _DOC_INDEX["matrix"]
    items = _DOC_INDEX["items"]

    if not items:
        return ""

    query_vector = vectorizer.transform([query])
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return ""

    scores = cosine_similarity(query_vector, matrix)[0]
    ranked_indices = scores.argsort()[::-1][: max(1, top_k)]

    snippets: List[str] = []
    for rank, index in enumerate(ranked_indices, start=1):
        score = float(scores[index])
        if score <= 0:
            continue
        item = items[index]
        snippet = _truncate_text(item["text"], TFIDF_SNIPPET_CHARS)
        snippets.append(
            f"[{rank}] {item['path']} (chunk {item['chunk_index']}, score={score:.3f})\n{snippet}"
        )

    return "\n\n".join(snippets).strip()


def get_signal_summary(signal_obj) -> str:
    """Convert FinalCoordinatorSignal to a formatted summary."""
    signal = _safe_get(signal_obj, "signal", "unknown")
    confidence = float(_safe_get(signal_obj, "confidence", 0.0) or 0.0)
    score = float(_safe_get(signal_obj, "score", 0.0) or 0.0)
    risk_level = _safe_get(signal_obj, "risk_level", "unknown")
    reasoning = _safe_get(signal_obj, "reasoning", "") or ""

    return (
        f"FINAL SIGNAL: {str(signal).upper()}\n"
        f"CONFIDENCE: {confidence:.2%}\n"
        f"NUMERIC SCORE: {score:+.3f}\n"
        f"RISK LEVEL: {risk_level}\n\n"
        f"KEY FACTORS:\n"
        f"{_format_key_factors(signal_obj)}\n\n"
        f"FULL REASONING:\n{reasoning}"
    )


def build_system_prompt(signal_obj, retrieved_docs: str = "", memory_summary: str = "") -> str:
    """Build the system prompt with current signal context and optional retrieved context."""
    signal = _safe_get(signal_obj, "signal", "unknown")
    confidence = float(_safe_get(signal_obj, "confidence", 0.0) or 0.0)
    score = float(_safe_get(signal_obj, "score", 0.0) or 0.0)
    risk_level = _safe_get(signal_obj, "risk_level", "unknown")
    key_factors = _format_key_factors(signal_obj)
    reasoning = _safe_get(signal_obj, "reasoning", "") or ""

    prompt = f"""You are an intelligent Bitcoin trading assistant. You have just analyzed the market using a multi-agent AI system and produced the following signal:

FINAL SIGNAL: {str(signal).upper()}
CONFIDENCE: {confidence:.2%}
NUMERIC SCORE: {score:+.3f}
RISK LEVEL: {risk_level}

KEY FACTORS:
{key_factors}

FULL REASONING:
{reasoning}
"""

    if memory_summary.strip():
        prompt += f"""

CONVERSATION MEMORY SUMMARY:
{memory_summary.strip()}
"""

    if retrieved_docs.strip():
        prompt += f"""

RELEVANT LOCAL DOCUMENTS:
{retrieved_docs.strip()}
"""

    prompt += """

Your job is to help the user understand this signal, answer their questions about the analysis, and explain the reasoning in plain language. Be concise, honest, and helpful. If asked something outside the scope of this analysis, say so clearly.

Scope rules:
- Only answer questions that are directly related to the Bitcoin decision system, the current signal, the retrieved local documents, or the application's workflow.
- Do not answer unrelated general-knowledge questions, trivia, or hypothetical prompts that are not tied to the application context.
- If the user asks something out of scope, briefly say it is outside the app/context and redirect them to ask about the signal, supporting evidence, agents, data sources, or workflow.
- Do not use general world knowledge to improvise an answer when the question is unrelated.

When explaining components:
- Technical: PyTorch models analyzing OHLCV price action
- Sentiment: FinBERT analysis of news, Twitter/X, and macro indicators
- Risk: On-chain metrics and geopolitical event assessment
- Coordinator: Weighted fusion with risk-adjusted scoring

Required response format:
1. TL;DR — 1 to 3 sentences only.
2. Structured explanation — use clear headings or bullet points.
3. What would flip this signal? — provide measurable triggers, thresholds, or concrete conditions.
4. Next steps / caveats — briefly note practical implications and uncertainty.

If you use retrieved documents or memory, keep the answer grounded and do not invent details. If the question is ambiguous, ask one concise clarifying question before answering."""
    return prompt


def _compress_with_llm(history: List[Dict[str, str]], client, model: str, existing_memory: str) -> Optional[str]:
    """Try to compress conversation history using the LLM."""
    if client is None:
        return None

    summary_source = history[:-CHAT_HISTORY_KEEP_LAST] if len(history) > CHAT_HISTORY_KEEP_LAST else history
    if not summary_source and not existing_memory.strip():
        return None

    prompt_parts: List[str] = []
    if existing_memory.strip():
        prompt_parts.append(f"Previous memory summary:\n{existing_memory.strip()}")
    if summary_source:
        prompt_parts.append("Older conversation turns to compress:\n" + _format_history_for_summary(summary_source))

    prompt = "\n\n".join(prompt_parts).strip()
    if not prompt:
        return None

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You compress chat history into a short, factual memory for future turns. Keep important facts, decisions, numbers, user intent, and unresolved questions. Do not add new information.",
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize the conversation below into a compact memory string under 1200 characters. "
                        "Use plain text only, preferably short bullets. Preserve key facts and unresolved questions.\n\n"
                        f"{prompt}"
                    ),
                },
            ],
            max_tokens=220,
            temperature=0.2,
        )
        summary = response.choices[0].message.content.strip()
        return _truncate_text(summary, SUMMARY_MAX_CHARS) if summary else None
    except Exception as exc:
        logger.warning("Conversation summarization failed; falling back to local compression: %s", exc)
        return None


def compress_conversation_history(history: List[Dict[str, str]], client=None, existing_memory: str = "") -> str:
    """Summarize older messages into a compact memory string when history grows too long."""
    if len(history) <= CHAT_HISTORY_SUMMARY_THRESHOLD and not existing_memory.strip():
        return existing_memory.strip()

    llm_summary = _compress_with_llm(history, client=client, model="llama-3.1-8b-instant", existing_memory=existing_memory)
    if llm_summary:
        return llm_summary

    summary_source = history[:-CHAT_HISTORY_KEEP_LAST] if len(history) > CHAT_HISTORY_KEEP_LAST else history
    return _local_history_summary(summary_source, existing_memory=existing_memory)


def _extract_keywords(*parts: str) -> List[str]:
    """Extract lightweight keywords to rank cached payloads."""
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "have", "has", "had", "are", "was",
        "were", "you", "your", "about", "what", "would", "signal", "assistant", "chat", "please",
        "into", "when", "then", "than", "but", "not", "can", "could", "should", "will", "just",
        "market", "bitcoin", "btc", "price", "signal", "analysis",
    }
    keywords = set()
    for part in parts:
        if not part:
            continue
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{2,}", str(part).lower()):
            if token not in stopwords:
                keywords.add(token)
    return sorted(keywords)


def _looks_like_in_scope_question(user_input: str, conversation_history: List[Dict[str, str]]) -> bool:
    """Heuristic guard to keep the assistant focused on the Bitcoin app context."""
    text = user_input.lower().strip()
    tokens = set(re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", text))

    if any(keyword in text for keyword in DOMAIN_KEYWORDS):
        return True

    # Allow short follow-up questions that rely on the recent conversation context.
    short_follow_up_phrases = (
        "why", "how", "what about", "what changed", "and why", "tell me more",
        "explain that", "clarify", "more detail", "can you explain", "why is that",
    )
    if len(text) <= 40 and any(text.startswith(phrase) or phrase in text for phrase in short_follow_up_phrases):
        return bool(conversation_history)

    # If the user uses pronouns or vague references after prior turns, keep it in scope.
    vague_follow_up_tokens = {"it", "that", "this", "those", "these", "same", "above", "below"}
    if tokens & vague_follow_up_tokens and len(conversation_history) > 0:
        return True

    return False


def _decode_cached_payload(value) -> str:
    """Decode a cache row into text if possible."""
    try:
        payload = pickle.loads(value)
    except Exception:
        return ""

    if isinstance(payload, dict) and "_content" in payload:
        content = payload["_content"]
    else:
        content = payload

    if isinstance(content, (bytes, bytearray)):
        return content.decode("utf-8", errors="ignore")
    return str(content)


def _summarize_cached_text(text: str) -> str:
    """Turn cached payloads into a short, human-readable fallback."""
    text = text.strip()
    if not text:
        return "No cached payload text was available."

    try:
        data = json.loads(text)
    except Exception:
        return _truncate_text(text, CACHE_MAX_CHARS)

    if isinstance(data, dict):
        if isinstance(data.get("articles"), list):
            lines = [
                f"Cached news payload: status={data.get('status', 'unknown')}, totalResults={data.get('totalResults', '?')}"
            ]
            for article in data.get("articles", [])[:3]:
                title = article.get("title") or "Untitled article"
                description = article.get("description") or article.get("content") or ""
                lines.append(f"- {title}: {_truncate_text(description, 220)}")
            return "\n".join(lines)

        if isinstance(data.get("data"), list):
            latest = data.get("data", [{}])[0] if data.get("data") else {}
            name = data.get("name") or "Cached time series"
            unit = data.get("unit") or "unknown unit"
            return (
                f"Cached time series: {name} ({unit}). Latest point: "
                f"{latest.get('date', 'unknown date')} = {latest.get('value', 'unknown value')}"
            )

        if "reasoning" in data:
            return _truncate_text(str(data.get("reasoning", "")), CACHE_MAX_CHARS)

        return _truncate_text(json.dumps(data, ensure_ascii=False), CACHE_MAX_CHARS)

    if isinstance(data, list):
        items = [f"- {str(item)}" for item in data[:5]]
        return "Cached list payload:\n" + "\n".join(items)

    return _truncate_text(text, CACHE_MAX_CHARS)


def load_cached_reasoning(signal_obj, user_input: str, repo_root: Path) -> str:
    """Load a cached fallback response from sentiment_agent_cache.sqlite."""
    cache_path = repo_root / "sentiment_agent_cache.sqlite"
    signal_summary = get_signal_summary(signal_obj)

    if not cache_path.exists():
        return (
            "Assistant (cached fallback unavailable):\n\n"
            f"{signal_summary}\n\n"
            "No cache database was found, so I could not load an offline response."
        )

    try:
        conn = sqlite3.connect(str(cache_path))
        cur = conn.cursor()
        cur.execute("SELECT key, value, expires FROM responses ORDER BY expires DESC")
        rows = cur.fetchall()
    except Exception as exc:
        logger.error("Failed to open cache database: %s", exc, exc_info=True)
        return (
            "Assistant (cached fallback unavailable):\n\n"
            f"{signal_summary}\n\n"
            f"I could not read the cache database: {exc}"
        )

    keywords = _extract_keywords(
        user_input,
        _safe_get(signal_obj, "signal", ""),
        _safe_get(signal_obj, "reasoning", ""),
        "bitcoin btc market signal sentiment risk technical macro coordinator",
    )

    scored_rows = []
    for key, value, expires in rows:
        decoded_text = _decode_cached_payload(value)
        if not decoded_text:
            continue
        lower_text = decoded_text.lower()
        score = sum(lower_text.count(keyword) for keyword in keywords)
        if "bitcoin" in lower_text or "btc" in lower_text:
            score += 3
        if score > 0:
            scored_rows.append((score, expires or 0, key, decoded_text))

    if scored_rows:
        scored_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_text = scored_rows[0][3]
    else:
        best_text = _decode_cached_payload(rows[0][1]) if rows else ""

    cached_summary = _summarize_cached_text(best_text)
    if not cached_summary:
        cached_summary = "A cached payload was found, but it could not be summarized cleanly."

    return (
        "Assistant (cached fallback from sentiment_agent_cache.sqlite):\n\n"
        f"{signal_summary}\n\n"
        f"Cached context:\n{cached_summary}\n\n"
        "Note: Groq was unavailable or failed, so this response was served from cache."
    )


def run_chat_interface():
    """Main chat interface loop."""

    # Import coordinator after we know we're in the right place
    try:
        from coordinator_agent.coordinator_agent import CoordinatorAgent
    except (ImportError, ModuleNotFoundError):
        print("ERROR: Cannot import coordinator agent. Make sure you're in the project root.")
        print("Expected structure: Code/ (project root)")
        sys.exit(1)

    # Try to import groq, but allow offline fallback if unavailable.
    Groq = None
    try:
        from groq import Groq as GroqClient

        Groq = GroqClient
    except ImportError:
        Groq = None

    # Get repo root
    try:
        repo_root = Path(__file__).parent
        coordinator = CoordinatorAgent(repo_root=repo_root)
    except Exception as e:
        print(f"ERROR: Failed to initialize coordinator: {e}")
        logger.error("Coordinator initialization failed", exc_info=True)
        sys.exit(1)

    # Run the coordinator to get latest signal
    print("\n" + "=" * 70)
    print("BITCOIN TRADING ASSISTANT - Fetching latest signal...")
    print("=" * 70 + "\n")

    try:
        signal = coordinator.run()
    except Exception as e:
        print(f"ERROR: Failed to run coordinator agent: {e}")
        logger.error("Coordinator execution failed", exc_info=True)
        sys.exit(1)

    # Display signal summary
    print(get_signal_summary(signal))
    print("\n" + "=" * 70)
    print("Starting Interactive Chat Session")
    print("=" * 70)
    print("\nType 'exit' or 'quit' to end the conversation.\n")

    # Initialize Groq client if possible; otherwise, rely on cache fallback.
    api_key = os.getenv("GROQ_API_KEY")
    client = None
    if Groq is None:
        print("WARNING: Groq package is not installed. Offline cache fallback will be used.")
    elif not api_key:
        print("WARNING: GROQ_API_KEY is not set. Offline cache fallback will be used.")
    else:
        try:
            client = Groq(api_key=api_key)
        except Exception as e:
            print(f"WARNING: Failed to initialize Groq client: {e}")
            logger.error("Groq initialization failed", exc_info=True)
            client = None

    # Conversation state
    conversation_history: List[Dict[str, str]] = []
    conversation_memory = ""

    # Preload docs once so retrieval is ready and failures surface early in logs.
    try:
        _ensure_doc_index()
    except Exception as exc:
        logger.warning("Local document index initialization failed: %s", exc, exc_info=True)

    # Chat loop
    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue

            if not _looks_like_in_scope_question(user_input, conversation_history):
                print(f"Assistant: {OUT_OF_SCOPE_REFUSAL}\n")
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": OUT_OF_SCOPE_REFUSAL,
                    }
                )
                continue

            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Assistant: Goodbye! Good luck with your trading. Remember to always manage risk!")
                break

            # Add user message to history
            conversation_history.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            # If the conversation is getting long, compress older turns into a memory string.
            if len(conversation_history) > CHAT_HISTORY_SUMMARY_THRESHOLD:
                conversation_memory = compress_conversation_history(
                    conversation_history,
                    client=client,
                    existing_memory=conversation_memory,
                )
                conversation_history = conversation_history[-CHAT_HISTORY_KEEP_LAST:]

            # Build system prompt with signal context plus optional memory and retrieval context.
            retrieval_query = "\n".join(
                [
                    user_input,
                    str(_safe_get(signal, "signal", "")),
                    str(_safe_get(signal, "reasoning", "")),
                    " ".join(map(str, _safe_get(signal, "key_factors", []) or [])),
                ]
            )
            retrieved_docs = retrieve_docs(retrieval_query, top_k=3)
            system_prompt = build_system_prompt(
                signal,
                retrieved_docs=retrieved_docs,
                memory_summary=conversation_memory,
            )

            # Call Groq LLM or fall back to cached response.
            if client is None:
                assistant_message = load_cached_reasoning(signal, user_input, repo_root)
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": assistant_message,
                    }
                )
                print(f"Assistant: {assistant_message}\n")
                continue

            try:
                messages = [{"role": "system", "content": system_prompt}] + conversation_history
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7,
                )

                # Extract assistant message
                assistant_message = response.choices[0].message.content.strip()

                # Add to history
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": assistant_message,
                    }
                )

                # Display response
                print(f"Assistant: {assistant_message}\n")

            except Exception as e:
                logger.error("LLM call failed: %s", e, exc_info=True)
                assistant_message = load_cached_reasoning(signal, user_input, repo_root)
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": assistant_message,
                    }
                )
                print(f"Assistant: {assistant_message}\n")

    except KeyboardInterrupt:
        print("\n\nChat interrupted. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.error("Unexpected error in chat loop", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_chat_interface()

