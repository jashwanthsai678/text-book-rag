import json
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.embedder import embed_texts
from app.vectorstore.qdrant_store import (
    ensure_collections,
    upsert_points,
    TEXT_COLLECTION,
    IMAGE_COLLECTION,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def stable_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def build_book_records(book_id: str, chapters: list):
    text_ids, text_texts, text_payloads = [], [], []
    image_ids, image_texts, image_payloads = [], [], []

    for chapter in chapters:
        chapter_number = chapter["chapter_number"]
        chapter_title = chapter["chapter_title"]

        image_location = {}
        for section in chapter["sections"]:
            for image_id in section["image_ids"]:
                image_location[image_id] = (
                    section["page_number"],
                    section["section_title"],
                )

        for idx, section in enumerate(chapter["sections"]):
            if not section["text"]:
                continue
            chunk_external_id = f"{book_id}_ch{chapter_number}_pg{section['page_number']}_{idx}"
            text_ids.append(stable_id(chunk_external_id))
            text_texts.append(section["text"])
            text_payloads.append(
                {
                    "chunk_id": chunk_external_id,
                    "book_id": book_id,
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "page_number": section["page_number"],
                    "section_title": section["section_title"],
                    "text": section["text"],
                    "image_ids": section["image_ids"],
                }
            )

        for image in chapter["images"]:
            if not image["url"]:
                continue
            page_number, section_title = image_location.get(
                image["image_id"], (None, None)
            )
            image_ids.append(stable_id(f"{book_id}_{image['image_id']}"))
            image_texts.append(image["caption"])
            image_payloads.append(
                {
                    "image_id": image["image_id"],
                    "book_id": book_id,
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "page_number": page_number,
                    "section_title": section_title,
                    "caption": image["caption"],
                    "url": image["url"],
                    "usage": image["usage"],
                    "relevance": image["relevance"],
                }
            )

    return text_ids, text_texts, text_payloads, image_ids, image_texts, image_payloads


def main():
    ensure_collections()
    processed_files = sorted(PROCESSED_DIR.glob("*.json"))
    print(f"Found {len(processed_files)} processed book file(s).")

    for path in processed_files:
        book_id = path.stem
        chapters = json.loads(path.read_text(encoding="utf-8"))

        (
            text_ids,
            text_texts,
            text_payloads,
            image_ids,
            image_texts,
            image_payloads,
        ) = build_book_records(book_id, chapters)

        print(f"[{book_id}] embedding {len(text_texts)} text chunks, {len(image_texts)} images...")

        if text_texts:
            upsert_points(TEXT_COLLECTION, text_ids, embed_texts(text_texts), text_payloads)
        if image_texts:
            upsert_points(IMAGE_COLLECTION, image_ids, embed_texts(image_texts), image_payloads)

    print("Done indexing all books.")


if __name__ == "__main__":
    main()
