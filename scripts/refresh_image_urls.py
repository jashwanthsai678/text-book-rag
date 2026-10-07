"""
Refreshes image URLs stored in Qdrant without re-embedding anything.

The textbook API's image URLs are Supabase signed URLs valid for ~6 hours.
Captions, page numbers, section titles, and embeddings never change - only
the `url` field goes stale. Re-running the full ingest/index pipeline just
to fix a URL would waste time re-downloading and re-embedding content that
hasn't changed. Instead, this script re-fetches each chapter (the only way
to get a fresh signed URL, since there's no endpoint for a single image) and
does a payload-only update on the matching Qdrant point.

Run this on a schedule comfortably inside the 6-hour expiry window (e.g.
every 3-4 hours) so served URLs are never stale.
"""

import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.api_client import list_books, list_chapters, fetch_chapter
from app.vectorstore.qdrant_store import IMAGE_COLLECTION, set_point_payload


def stable_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def refresh_book(book_id: str):
    print(f"\n=== Refreshing image URLs for {book_id} ===")
    chapters_meta = list_chapters(book_id)

    updated = 0
    not_indexed = 0
    for chapter_meta in chapters_meta:
        chapter_number = chapter_meta["chapter_number"]
        try:
            raw = fetch_chapter(book_id, chapter_number)
        except Exception as e:
            print(f"  skipped chapter {chapter_number} ({e})")
            continue

        for image in raw.images:
            if not image.url:
                continue
            point_id = stable_id(f"{book_id}_{image.image_id}")
            if set_point_payload(IMAGE_COLLECTION, point_id, {"url": image.url}):
                updated += 1
            else:
                not_indexed += 1

    print(f"  refreshed {updated} image URL(s), {not_indexed} not yet indexed")


def main():
    books = list_books()
    book_ids = [b["book_id"] for b in books]
    print(f"Found {len(book_ids)} published book(s).")

    for book_id in book_ids:
        refresh_book(book_id)

    print("\nDone refreshing image URLs.")


if __name__ == "__main__":
    main()
