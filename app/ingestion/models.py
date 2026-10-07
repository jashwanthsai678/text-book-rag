from typing import List, Optional
from pydantic import BaseModel


class ImageAsset(BaseModel):
    image_id: str
    caption: str
    order_index: int
    url: Optional[str] = None
    usage: str
    relevance: str


class RawChapterResponse(BaseModel):
    book_id: str
    chapter_number: int
    chapter_title: str
    page_start: int
    page_end: int
    content: str
    images: List[ImageAsset]


class ParsedSection(BaseModel):
    section_title: Optional[str]
    page_number: int
    text: str
    image_ids: List[str]


class ParsedChapter(BaseModel):
    book_id: str
    chapter_number: int
    chapter_title: str
    page_start: int
    page_end: int
    sections: List[ParsedSection]
    images: List[ImageAsset]
