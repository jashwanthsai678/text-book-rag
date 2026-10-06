import os
import time
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv
from app.ingestion.models import RawChapterResponse

load_dotenv()

API_BASE_URL = os.environ["API_BASE_URL"]

MAX_RETRIES = 3
TIMEOUT_SECONDS = 60
RETRY_BACKOFF_SECONDS = 5


def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = httpx.get(url, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"  attempt {attempt} failed ({e}), retrying...")
                time.sleep(RETRY_BACKOFF_SECONDS)
        except httpx.TimeoutException as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"  attempt {attempt} failed ({e}), retrying...")
                time.sleep(RETRY_BACKOFF_SECONDS)

    raise last_error


def _unwrap_list(data: Any) -> List[Dict[str, Any]]:
    """The published/books and .../chapters endpoints may return either a
    bare list or a dict wrapping the list under a key like "books"/"chapters".
    Handle both shapes defensively since we haven't confirmed the exact schema."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("books", "chapters", "items", "data", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Unexpected response shape, couldn't find a list: {data!r}")


def list_books(
    board: Optional[str] = None,
    grade: Optional[int] = None,
    subject: Optional[str] = None,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params = {
        k: v
        for k, v in {
            "board": board,
            "grade": grade,
            "subject": subject,
            "language": language,
        }.items()
        if v is not None
    }
    data = _get_json(f"{API_BASE_URL}/published/books", params=params)
    return _unwrap_list(data)


def list_chapters(book_id: str) -> List[Dict[str, Any]]:
    data = _get_json(f"{API_BASE_URL}/published/books/{book_id}/chapters")
    return _unwrap_list(data)


def fetch_chapter(book_id: str, chapter_number: int) -> RawChapterResponse:
    url = f"{API_BASE_URL}/published/books/{book_id}/chapters/{chapter_number}"
    data = _get_json(url)
    return RawChapterResponse.model_validate(data)
