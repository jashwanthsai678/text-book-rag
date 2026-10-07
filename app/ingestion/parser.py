import re
from typing import List, Optional
from app.ingestion.models import RawChapterResponse, ParsedChapter, ParsedSection

IMG_TAG_RE = re.compile(r'<img id="([^"]+)"\s*/>')

# Format A (seen so far only in the EVS Class 5 book): Markdown, with real
# per-page markers.
MARKDOWN_MARKER_RE = re.compile(
    r'(?m)^(?:<!--\s*page\s*(?P<page>\d+)\s*-->|(?P<hashes>#{1,6})\s+(?P<heading>.+))$'
)
PAGE_MARKER_RE = re.compile(r'<!--\s*page\s*\d+\s*-->')

# Format B (every other book seen in the catalog so far): bracket tags like
# [CONCEPT], [ACTIVITY], [HEADING], [FIGURE DESCRIPTION], no page markers at
# all. [HEADING] is the section-title analogue to a Markdown heading; the
# other tags are left inline as part of the section body.
BRACKET_HEADING_RE = re.compile(r'(?m)^\[HEADING\]\s*(.+)$')


def parse_chapter(raw: RawChapterResponse) -> ParsedChapter:
    if PAGE_MARKER_RE.search(raw.content):
        sections = _parse_markdown_format(raw)
    else:
        sections = _parse_bracket_tag_format(raw)

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
    # No per-page markers exist in this format, so every section is tagged
    # with the chapter's start page rather than a fabricated exact page.
    content = raw.content
    headings = list(BRACKET_HEADING_RE.finditer(content))

    if not headings:
        return [_build_section(None, raw.page_start, content)]

    sections: List[ParsedSection] = []
    if headings[0].start() > 0:
        sections.append(_build_section(None, raw.page_start, content[: headings[0].start()]))

    for i, m in enumerate(headings):
        title = m.group(1).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
        sections.append(_build_section(title, raw.page_start, content[start:end]))

    return sections


def _build_section(title: Optional[str], page: int, text_block: str) -> ParsedSection:
    image_ids = IMG_TAG_RE.findall(text_block)
    clean_text = IMG_TAG_RE.sub("", text_block)
    clean_text = re.sub(r"\n{2,}", "\n\n", clean_text).strip()
    return ParsedSection(
        section_title=title, page_number=page, text=clean_text, image_ids=image_ids
    )
