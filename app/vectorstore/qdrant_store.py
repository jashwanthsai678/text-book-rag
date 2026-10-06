import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBEDDING_DIM = 384

TEXT_COLLECTION = "textbook_text"
IMAGE_COLLECTION = "textbook_images"

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def ensure_collections():
    client = get_client()
    for name in (TEXT_COLLECTION, IMAGE_COLLECTION):
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )


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
    client.upsert(collection_name=collection_name, points=points)
