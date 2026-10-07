from typing import List, Optional
from pydantic import BaseModel
from qdrant_client.models import Filter, FieldCondition, MatchValue, HasIdCondition

from app.embeddings.embedder import embed_texts
from app.vectorstore.qdrant_store import get_client, stable_id, TEXT_COLLECTION, IMAGE_COLLECTION


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


def _image_result(point_id, payload: dict, score: float) -> ImageResult:
    return ImageResult(
        image_id=payload["image_id"],
        book_id=payload["book_id"],
        chapter_number=payload["chapter_number"],
        chapter_title=payload["chapter_title"],
        page_number=payload["page_number"],
        caption=payload["caption"],
        url=payload["url"],
        score=score,
    )


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

    # Prefer images the textbook itself placed next to the matched passages
    # (via <img id> tags resolved during parsing) over an independent
    # caption-only search - a short caption can coincidentally resemble the
    # query even when it's from an unrelated book/subject, but an image
    # sitting right next to a passage that itself matched is ground-truth
    # relevant.
    linked_image_results: List[ImageResult] = []
    seen_point_ids = set()

    if request.top_k_images > 0:
        linked_lookup: List[tuple] = []  # (point_id, inherited_score)
        for hit in text_hits:
            book_id = hit.payload["book_id"]
            for image_id in hit.payload.get("image_ids", []):
                point_id = stable_id(f"{book_id}_{image_id}")
                if point_id in seen_point_ids:
                    continue
                seen_point_ids.add(point_id)
                linked_lookup.append((point_id, hit.score))
                if len(linked_lookup) >= request.top_k_images:
                    break
            if len(linked_lookup) >= request.top_k_images:
                break

        if linked_lookup:
            points = client.retrieve(
                collection_name=IMAGE_COLLECTION,
                ids=[p[0] for p in linked_lookup],
                with_payload=True,
            )
            points_by_id = {str(p.id): p for p in points}
            for point_id, inherited_score in linked_lookup:
                point = points_by_id.get(point_id)
                if point is None:
                    continue  # e.g. excluded at index time (no url/caption)
                linked_image_results.append(_image_result(point_id, point.payload, inherited_score))

    image_results = list(linked_image_results)
    remaining = request.top_k_images - len(image_results)

    if remaining > 0:
        fallback_filter_conditions = list(query_filter.must) if query_filter else []
        fallback_filter = Filter(
            must=fallback_filter_conditions,
            must_not=[HasIdCondition(has_id=list(seen_point_ids))] if seen_point_ids else None,
        )

        image_hits = client.query_points(
            collection_name=IMAGE_COLLECTION,
            query=query_vector,
            query_filter=fallback_filter,
            limit=remaining,
        ).points

        for hit in image_hits:
            image_results.append(_image_result(str(hit.id), hit.payload, hit.score))

    return RetrieveResponse(text=text_results, images=image_results)
