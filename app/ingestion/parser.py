import re
from typing import List, Optional
from app.ingestion.models import RawChapterResponse, ParsedChapter, ParsedSection

IMG_TAG_RE = re.compile(r'<img id="([^"]+)"\s*/>')

# Format A (seen so far only in the EVS Class 5 book, from an earlier,
# separate pipeline): Markdown, '#'-style headings, with real per-page
# markers, no bracket tags.
MARKDOWN_MARKER_RE = re.compile(
    r'(?m)^(?:<!--\s*page\s*(?P<page>\d+)\s*-->|(?P<hashes>#{1,6})\s+(?P<heading>.+))$'
)

# Format B (every other book in the catalog): bracket tags like [CONCEPT],
# [ACTIVITY], [HEADING], [FIGURE DESCRIPTION]. [HEADING] is the
# section-title analogue to a Markdown heading; the other tags are left
# inline as part of the section body. Books ingested by eduteach-ingest-
# service from 2026-10-08 onward also interleave real <!-- page N -->
# markers with these tags (same marker syntax as format A, just mixed with
# bracket tags instead of '#' headings); older bracket-tag books have none,
# so this format's parser falls back to the chapter's page_start for them,
# same as it always did.
BRACKET_TAG_RE = re.compile(
    r'(?m)^\[(?:HEADING|CONCEPT|ACTIVITY|KEY WORDS|WHAT HAVE WE LEARNT|TEXTBOOK QUESTION)\]'
)
BRACKET_MARKER_RE = re.compile(
    r'(?m)^(?:<!--\s*page\s*(?P<page>\d+)\s*-->|\[HEADING\]\s*(?P<heading>.+))$'
)


def parse_chapter(raw: RawChapterResponse) -> ParsedChapter:
    # Routed by which tag style is actually present, not by whether page
    # markers exist -- a bracket-tag-format book can now carry page markers
    # too (see BRACKET_MARKER_RE), and format A never uses bracket tags, so
    # checking for bracket tags first is the one signal that can't misroute
    # either direction.
    if BRACKET_TAG_RE.search(raw.content):
        sections = _parse_bracket_tag_format(raw)
    else:
        sections = _parse_markdown_format(raw)

    return ParsedChapter(
        book_id=raw.book_id,
        chapter_number=raw.chapter_number,
        chapter_title=raw.chapter_title,
        page_start=raw.page_start,
        page_end=raw.page_end,
        sections=sections,
        images=raw.images,
    )


def _parse_markdown_format(raw: RawChapterResponse) -> List[ParsedSection]:
    content = raw.content
    markers = list(MARKDOWN_MARKER_RE.finditer(content))

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
    return sections


def _parse_bracket_tag_format(raw: RawChapterResponse) -> List[ParsedSection]:
    """Splits on [HEADING] tags for section boundaries, same as always.

    Also tracks interleaved <!-- page N --> markers when present (books
    ingested from 2026-10-08 onward) to tag each section with its real page
    instead of just the chapter's start page. Older bracket-tag books have
    no such markers, so every section just keeps falling back to
    page_start, identical to this function's behavior before that date.
    """
    content = raw.content
    markers = list(BRACKET_MARKER_RE.finditer(content))

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
            flush(m.start())
            section_start = m.end()
            current_title = m.group("heading").strip()

    flush(len(content))
    if not sections:
        sections.append(_build_section(None, raw.page_start, content))
    return sections


def _build_section(title: Optional[str], page: int, text_block: str) -> ParsedSection:
    image_ids = IMG_TAG_RE.findall(text_block)
    clean_text = IMG_TAG_RE.sub("", text_block)
    clean_text = re.sub(r"\n{2,}", "\n\n", clean_text).strip()
    return ParsedSection(
        section_title=title, page_number=page, text=clean_text, image_ids=image_ids
    )
