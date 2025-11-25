from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

DEFAULT_CONFIG_PATH = Path("config/config.json")


@dataclass
class Config:
    user_agent: str
    http_timeout: int
    log_level: str = "INFO"
    quotes_output_path: Path = Path("output/quotes.jsonl")
    author_output_path: Path = Path("output/authors.jsonl")
    collection_name: str = "quotes_demo_v1"
    auto_increment_collection: bool = False
    crawl_depth: int = 0
    request_delay: float = 1.0
    max_pages: int = 50
    skip_keywords: List[str] = field(default_factory=lambda: ["login", "signin", "signup", "search", "admin"])
    max_retries: int = 1
    include_patterns: List[str] = field(default_factory=list)


@dataclass
class QuoteRecord:
    source_url: str
    quote: str
    author: str
    tags: List[str]
    depth: int
    page_number: int
    position_on_page: int
    language_hint: str | None = None


@dataclass
class AuthorRecord:
    source_url: str
    name: str
    born_date: str
    born_location: str
    description: str
    depth: int


@dataclass
class QuoteContext:
    """Container for quote metadata assembly; used by the pipeline to build payloads."""

    record: QuoteRecord
    author_meta: dict
    text_features: Dict[str, object]
    structural: Dict[str, object]
    dedupe_key: str
    is_duplicate: bool
    source_confidence: float
    parse_status: str
    quality_score: float
    safety_flags: List[str]
    embedding: List[float]
    top_keywords: List[str]
    topic_label: str
    emotion_label: str
    document_id: str
    collection_name: str
    scraped_at: str
    start_url: str
    source_site: str
    tags_normalized: List[str]
    tags_count: int
    primary_tag: str | None
    secondary_tags: List[str]
