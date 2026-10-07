import os
from typing import List
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIM = 1536

BATCH_SIZE = 100


def embed_texts(texts: List[str]) -> List[List[float]]:
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        all_embeddings.extend(_embed_batch(batch))
    return all_embeddings


def _embed_batch(texts: List[str]) -> List[List[float]]:
    response = httpx.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()["data"]
    # Response items aren't guaranteed to preserve input order; sort by the
    # index OpenAI-compatible APIs include per item.
    data.sort(key=lambda item: item["index"])
    return [item["embedding"] for item in data]
