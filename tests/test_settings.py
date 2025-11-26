import json
from pathlib import Path

from webscraper_krew.settings import load_config


def test_load_config_with_all_fields(tmp_path: Path) -> None:
    cfg = {
        "user_agent": "ua",
        "http_timeout": 5,
        "log_level": "DEBUG",
        "quotes_output_path": "out/quotes.jsonl",
        "author_output_path": "out/authors.jsonl",
        "collection_name": "col_v1",
        "auto_increment_collection": True,
        "crawl_depth": 2,
        "request_delay": 0.2,
        "max_pages": 10,
        "skip_keywords": ["login", "search"],
        "max_retries": 3,
        "include_patterns": [r"/tag/", r"/page/"],
        "follow_include_patterns": [r"/tag/"],
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = load_config(cfg_path)

    assert loaded.user_agent == "ua"
    assert loaded.http_timeout == 5
    assert loaded.log_level == "DEBUG"
    assert loaded.quotes_output_path.name == "quotes.jsonl"
    assert loaded.author_output_path.name == "authors.jsonl"
    assert loaded.collection_name == "col_v1"
    assert loaded.auto_increment_collection is True
    assert loaded.crawl_depth == 2
    assert loaded.request_delay == 0.2
    assert loaded.max_pages == 10
    assert loaded.skip_keywords == ["login", "search"]
    assert loaded.max_retries == 3
    assert loaded.include_patterns == [r"/tag/", r"/page/"]
    assert loaded.follow_include_patterns == [r"/tag/"]


def test_load_config_defaults(tmp_path: Path) -> None:
    cfg = {
        "user_agent": "ua",
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = load_config(cfg_path)

    assert loaded.skip_keywords  # defaults applied
    assert loaded.include_patterns == []
    assert loaded.follow_include_patterns == []
    assert loaded.auto_increment_collection is False
