import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from app.ingestion.api_client import list_books, list_chapters, fetch_chapter
from app.ingestion.parser import parse_chapter

load_dotenv()

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional filters matching the API's own query params. Leave unset to ingest
# every published book in the catalog.
BOARD_FILTER = os.environ.get("BOARD_FILTER")
GRADE_FILTER = os.environ.get("GRADE_FILTER")
SUBJECT_FILTER = os.environ.get("SUBJECT_FILTER")
LANGUAGE_FILTER = os.environ.get("LANGUAGE_FILTER")


def ingest_book(book_id: str):
    print(f"\n=== Ingesting {book_id} ===")
    chapters_meta = list_chapters(book_id)
    chapter_numbers = [c["chapter_number"] for c in chapters_meta]
    print(f"  {len(chapter_numbers)} chapter(s) listed: {chapter_numbers}")

    chapters = []
    for chapter_number in chapter_numbers:
        print(f"  fetching chapter {chapter_number}...")
        try:
            raw = fetch_chapter(book_id, chapter_number)
        except Exception as e:
            print(f"    skipped (chapter {chapter_number} failed: {e})")
            continue
        parsed = parse_chapter(raw)
        chapters.append(parsed.model_dump())

    out_path = OUT_DIR / f"{book_id}.json"
    out_path.write_text(json.dumps(chapters, indent=2), encoding="utf-8")
    print(f"  saved {len(chapters)} chapters to {out_path}")


def main():
    grade = int(GRADE_FILTER) if GRADE_FILTER else None
    books = list_books(
        board=BOARD_FILTER,
        grade=grade,
        subject=SUBJECT_FILTER,
        language=LANGUAGE_FILTER,
    )
    book_ids = [b["book_id"] for b in books]
    print(f"Found {len(book_ids)} published book(s): {book_ids}")

    for book_id in book_ids:
        ingest_book(book_id)

    print(f"\nDone. Ingested {len(book_ids)} book(s).")


if __name__ == "__main__":
    main()
