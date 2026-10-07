from typing import List, Optional
from pydantic import BaseModel
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.embeddings.embedder import embed_texts
from app.vectorstore.qdrant_store import get_client, TEXT_COLLECTION, IMAGE_COLLECTION


class TextResult(BaseModel):
    chunk_id: str
    book_id: str
    chapter_number: int
    chapter_title: str
    page_number: int
    section_title: Optional[str]
    text: str
    score: float


class ImageResult(BaseModel):
    image_id: str
    book_id: str
    chapter_number: int
    chapter_title: str
    page_number: Optional[int]
    caption: str
    url: str
    score: float


class RetrieveRequest(BaseModel):
    book_id: Optional[str] = None
    chapter: Optional[int] = None
    query: str
    top_k_text: int = 5
    top_k_images: int = 3


class RetrieveResponse(BaseModel):
    text: List[TextResult]
    images: List[ImageResult]


def _build_filter(book_id: Optional[str], chapter: Optional[int]) -> Optional[Filter]:
    conditions = []
    if book_id is not None:
        conditions.append(FieldCondition(key="book_id", match=MatchValue(value=book_id)))
    if chapter is not None:
        conditions.append(
            FieldCondition(key="chapter_number", match=MatchValue(value=chapter))
        )
    return Filter(must=conditions) if conditions else None


def retrieve_content(request: RetrieveRequest) -> RetrieveResponse:
    query_vector = embed_texts([request.query])[0]
    query_filter = _build_filter(request.book_id, request.chapter)
    client = get_client()

    text_hits = client.query_points(
        collection_name=TEXT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=request.top_k_text,
    ).points

    image_hits = client.query_points(
        collection_name=IMAGE_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=request.top_k_images,
    ).points

    text_results = [
        TextResult(
            chunk_id=hit.payload["chunk_id"],
            book_id=hit.payload["book_id"],
            chapter_number=hit.payload["chapter_number"],
            chapter_title=hit.payload["chapter_title"],
            page_number=hit.payload["page_number"],
            section_title=hit.payload["section_title"],
            text=hit.payload["text"],
            score=hit.score,
        )
        for hit in text_hits
    ]

    image_results = [
        ImageResult(
            image_id=hit.payload["image_id"],
            book_id=hit.payload["book_id"],
            chapter_number=hit.payload["chapter_number"],
            chapter_title=hit.payload["chapter_title"],
            page_number=hit.payload["page_number"],
            caption=hit.payload["caption"],
            url=hit.payload["url"],
            score=hit.score,
        )
        for hit in image_hits
    ]

    return RetrieveResponse(text=text_results, images=image_results)
