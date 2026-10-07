import os
import uuid
from typing import Any, Dict, List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from qdrant_client.http.exceptions import UnexpectedResponse
from app.embeddings.embedder import EMBEDDING_DIM

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

TEXT_COLLECTION = "textbook_text"
IMAGE_COLLECTION = "textbook_images"

_client = None


def stable_id(name: str) -> str:
    """Deterministic point ID derived from a natural key (e.g. book_id +
    image_id), so re-running ingestion/indexing upserts the same point
    instead of creating a duplicate."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return _client


def ensure_collections():
    client = get_client()
    for name in (TEXT_COLLECTION, IMAGE_COLLECTION):
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        # Filtering on a field (as retrieval does for book_id/chapter_number)
        # requires a payload index. Creating one that already exists is a
        # no-op, so this is safe to call every time.
        client.create_payload_index(name, "book_id", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "chapter_number", field_schema=PayloadSchemaType.INTEGER)


BATCH_SIZE = 64


def upsert_points(
    collection_name: str,
    ids: List[str],
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
):
    client = get_client()
    points = [
        PointStruct(id=point_id, vector=vector, payload=payload)
        for point_id, vector, payload in zip(ids, vectors, payloads)
    ]
    # Batch uploads: a single request with hundreds of embeddings can exceed
    # the write timeout over a real network (fine on localhost, not on a
    # remote cluster).
    for i in range(0, len(points), BATCH_SIZE):
        client.upsert(collection_name=collection_name, points=points[i : i + BATCH_SIZE])


def set_point_payload(collection_name: str, point_id: str, payload: Dict[str, Any]) -> bool:
    """Update only the given payload fields on an existing point, leaving its
    vector untouched. Used to refresh fields that go stale (e.g. signed image
    URLs) without re-embedding.

    Returns False (instead of raising) if the point doesn't exist - e.g. the
    book/chapter hasn't been indexed yet - since that's an expected state for
    a catalog where not every published book has been run through
    build_index.py yet.
    """
    client = get_client()
    try:
        client.set_payload(collection_name=collection_name, payload=payload, points=[point_id])
        return True
    except UnexpectedResponse as e:
        if e.status_code == 404:
            return False
        raise
