import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.embeddings.embedder import embed_texts
from app.vectorstore.qdrant_store import (
    ensure_collections,
    upsert_points,
    stable_id,
    TEXT_COLLECTION,
    IMAGE_COLLECTION,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ~5000 tokens at a conservative ~4 chars/token, comfortably under
# text-embedding-3-small's 8191-token limit. Needed for real data:
# ts_scert_class5_english_en (reconstructed from an early session, with
# headings sparse enough that one section between them can span an entire
# chapter) produced a 46,768-char chunk that OpenRouter's embeddings API
# rejected outright (400 Bad Request) for exceeding the model's token limit.
_MAX_CHUNK_CHARS = 20000


def _split_long_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Splits oversized text on paragraph boundaries where possible, to avoid
    cutting a sentence in half. Most sections never hit this at all -- it's a
    safety net for the rare section too large for a single embedding call,
    not the primary chunking strategy (that's parser.py's job, splitting on
    headings/page markers)."""
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            parts.append(text[start:])
            break
        split_at = text.rfind("\n\n", start, end)
        if split_at <= start:
            split_at = end
        else:
            split_at += 2
        parts.append(text[start:split_at])
        start = split_at
    return parts


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
            sub_texts = _split_long_text(section["text"])
            for sub_idx, sub_text in enumerate(sub_texts):
                suffix = f"_{sub_idx}" if len(sub_texts) > 1 else ""
                chunk_external_id = f"{book_id}_ch{chapter_number}_pg{section['page_number']}_{idx}{suffix}"
                text_ids.append(stable_id(chunk_external_id))
                text_texts.append(sub_text)
                text_payloads.append(
                    {
                        "chunk_id": chunk_external_id,
                        "book_id": book_id,
                        "chapter_number": chapter_number,
                        "chapter_title": chapter_title,
                        "page_number": section["page_number"],
                        "section_title": section["section_title"],
                        "text": sub_text,
                        # parser.py already strips <img id="..."/> placeholders out of
                        # section["text"] (they live only in image_ids), so there's no
                        # positional signal left to split them across sub-chunks --
                        # all of this section's images go on its first slice only,
                        # not duplicated onto every split, which only matters for the
                        # rare oversized section in the first place.
                        "image_ids": section["image_ids"] if sub_idx == 0 else [],
                    }
                )

        for image in chapter["images"]:
            if not image["url"] or not image["caption"].strip():
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
