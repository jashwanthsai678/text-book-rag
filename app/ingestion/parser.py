import re
from typing import List, Optional
from app.ingestion.models import RawChapterResponse, ParsedChapter, ParsedSection

IMG_TAG_RE = re.compile(r'<img id="([^"]+)"\s*/>')
MARKER_RE = re.compile(
    r'(?m)^(?:<!--\s*page\s*(?P<page>\d+)\s*-->|(?P<hashes>#{1,6})\s+(?P<heading>.+))$'
)


def parse_chapter(raw: RawChapterResponse) -> ParsedChapter:
    content = raw.content
    markers = list(MARKER_RE.finditer(content))

    sections: List[ParsedSection] = []
    current_page = raw.page_start
    current_title: Optional[str] = None
    section_start = 0

    def flush(end_pos: int):
        text_block = content[section_start:end_pos]
        if text_block.strip():
            sections.append(_build_section(current_title, current_page, text_block))

    for m in markers:
        if m.group("page") is not None:
            flush(m.start())
            section_start = m.end()
            current_page = int(m.group("page"))
        else:
            # level-1 heading (chapter title banner) is noise, not a section
            if len(m.group("hashes")) == 1:
                flush(m.start())
                section_start = m.end()
                continue
            flush(m.start())
            section_start = m.end()
            current_title = m.group("heading").strip()

    flush(len(content))

    return ParsedChapter(
        book_id=raw.book_id,
        chapter_number=raw.chapter_number,
        chapter_title=raw.chapter_title,
        page_start=raw.page_start,
        page_end=raw.page_end,
        sections=sections,
        images=raw.images,
    )


def _build_section(title: Optional[str], page: int, text_block: str) -> ParsedSection:
    image_ids = IMG_TAG_RE.findall(text_block)
    clean_text = IMG_TAG_RE.sub("", text_block)
    clean_text = re.sub(r"\n{2,}", "\n\n", clean_text).strip()
    return ParsedSection(
        section_title=title, page_number=page, text=clean_text, image_ids=image_ids
    )
