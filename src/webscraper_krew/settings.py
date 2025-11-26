"""
Config loading utilities.
"""
from pathlib import Path

from .models import DEFAULT_CONFIG_PATH, Config
import json


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load scraper configuration from a JSON file.

    Args:
        path: Path to the JSON config; defaults to config/config.json.
    Returns:
        Config object with all runtime values normalized and validated. Parsed keys:
        - user_agent: string
        - http_timeout: int seconds
        - log_level: DEBUG/INFO/WARNING/ERROR
        - quotes_output_path: path to quotes JSONL
        - author_output_path: path to authors JSONL
        - collection_name: base collection name
        - auto_increment_collection: bool to auto-bump collection version
        - crawl_depth: int BFS depth
        - request_delay: float seconds between requests
        - max_pages: int hard cap on pages fetched
        - skip_keywords: list of substrings to skip in URLs
        - include_patterns: list of regex/glob patterns to allow URLs (applied to discovered links if follow patterns are not set)
        - follow_include_patterns: list of regex/glob patterns applied to discovered links (overrides include_patterns if set)
        - max_retries: int retries per URL
    Raises:
        FileNotFoundError: if the config file is missing.
        ValueError: for malformed or invalid types.
    """
    config_path = path.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open(encoding="utf-8") as config:
        data = json.load(config)

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object")

    user_agent = str(data.get("user_agent", "webscraper-krew/0.1 (+https://example.com)"))

    try:
        http_timeout = int(data.get("http_timeout", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError("Config http_timeout must be an integer") from exc

    log_level = str(data.get("log_level", "INFO")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("Config log_level must be one of: DEBUG, INFO, WARNING, ERROR")

    quotes_output_path_raw = data.get("quotes_output_path", "output/quotes.jsonl")
    quotes_output_path = Path(quotes_output_path_raw).expanduser()
    author_output_path_raw = data.get("author_output_path", "output/authors.jsonl")
    author_output_path = Path(author_output_path_raw).expanduser()

    try:
        crawl_depth = int(data.get("crawl_depth", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Config crawl_depth must be an integer") from exc
    if crawl_depth < 0:
        raise ValueError("Config crawl_depth must be zero or a positive integer")

    try:
        request_delay = float(data.get("request_delay", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Config request_delay must be a number") from exc
    if request_delay < 0:
        raise ValueError("Config request_delay must be zero or positive")

    try:
        max_pages = int(data.get("max_pages", 50))
    except (TypeError, ValueError) as exc:
        raise ValueError("Config max_pages must be an integer") from exc
    if max_pages <= 0:
        raise ValueError("Config max_pages must be a positive integer")

    try:
        max_retries = int(data.get("max_retries", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Config max_retries must be an integer") from exc
    if max_retries < 0:
        raise ValueError("Config max_retries must be zero or a positive integer")

    include_patterns_raw = data.get("include_patterns", [])
    if include_patterns_raw is None:
        include_patterns = []
    elif isinstance(include_patterns_raw, list):
        include_patterns = [str(pat) for pat in include_patterns_raw if str(pat).strip()]
    else:
        raise ValueError("Config include_patterns must be a list of strings")

    follow_patterns_raw = data.get("follow_include_patterns", include_patterns)
    if follow_patterns_raw is None:
        follow_include_patterns = []
    elif isinstance(follow_patterns_raw, list):
        follow_include_patterns = [str(pat) for pat in follow_patterns_raw if str(pat).strip()]
    else:
        raise ValueError("Config follow_include_patterns must be a list of strings")

    skip_keywords_raw = data.get("skip_keywords")
    if skip_keywords_raw is None:
        skip_keywords = ["login", "signin", "signup", "search", "admin"]
    elif isinstance(skip_keywords_raw, list):
        skip_keywords = [str(item).lower() for item in skip_keywords_raw if str(item).strip()]
    else:
        raise ValueError("Config skip_keywords must be a list of strings")

    collection_name = str(data.get("collection_name", "quotes_demo_v1"))

    return Config(
        user_agent=user_agent,
        http_timeout=http_timeout,
        log_level=log_level,
        quotes_output_path=quotes_output_path,
        author_output_path=author_output_path,
        collection_name=collection_name,
        crawl_depth=crawl_depth,
        request_delay=request_delay,
        max_pages=max_pages,
        skip_keywords=skip_keywords,
        max_retries=max_retries,
        include_patterns=include_patterns,
        follow_include_patterns=follow_include_patterns,
        auto_increment_collection=bool(data.get("auto_increment_collection", False)),
    )
