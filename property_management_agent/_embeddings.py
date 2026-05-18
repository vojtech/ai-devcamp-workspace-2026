"""
Embedding helper — wraps Google's text-embedding-004 model into a single
`embed_text(text)` function used by the archive ingestion + search pipeline.

Uses the same GOOGLE_API_KEY the rest of the agent uses (Tier 1 Postpay,
inputs are NOT used for training).
"""
import logging
import os
from typing import Optional

from google import genai

logger = logging.getLogger(__name__)

# gemini-embedding-001 is stable + GA on AI Studio (text-embedding-004 was
# retired from public Gemini API). It defaults to 3072 dims but supports
# Matryoshka — we request 768 to keep the sqlite-vec table small and search
# fast at almost no quality cost.
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

# Hard cap on input length to keep latency predictable. The model accepts
# 2048 tokens (~8000 chars of English); we truncate well under that.
MAX_CHARS = 7000

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to "
                "property_management_agent/.env or your shell environment."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def embed_text(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> list[float]:
    """Return a 768-dim embedding vector for `text`.

    task_type guidance:
      RETRIEVAL_DOCUMENT  — when embedding documents to be stored/indexed
      RETRIEVAL_QUERY     — when embedding a user query for retrieval
      SEMANTIC_SIMILARITY — generic; default and fine for both sides
    """
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM

    body = text[:MAX_CHARS]
    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=body,
        config={
            "task_type": task_type,
            "output_dimensionality": EMBEDDING_DIM,
        },
    )
    return list(result.embeddings[0].values)


def embed_for_storage(text: str) -> list[float]:
    """Embed a document body for long-term storage / archival indexing."""
    return embed_text(text, task_type="RETRIEVAL_DOCUMENT")


def embed_for_query(text: str) -> list[float]:
    """Embed a user query for similarity lookup against stored documents."""
    return embed_text(text, task_type="RETRIEVAL_QUERY")
