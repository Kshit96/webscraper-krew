"""
Entry point for running the scraper.
Handles argument parsing, config loading, and invoking the crawler + writers.
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from .models import DEFAULT_CONFIG_PATH, Config
from .scraper import build_author_lookup, scrape, write_authors_jsonl, write_quotes_jsonl
from .settings import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal web scraper that extracts quotes/authors.")
    parser.add_argument("url", nargs="?", default="https://quotes.toscrape.com", help="URL to scrape")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config file (default: config/config.json)",
    )
    args = parser.parse_args()

    start_url = args.url
    config: Config = load_config(args.config)

    logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO), format="%(levelname)s: %(message)s")
    logging.debug("Loaded config from %s", args.config)
    logging.debug("Using User-Agent: %s", config.user_agent)

    quotes, authors = scrape(start_url, config)

    scraped_at = datetime.now(timezone.utc).isoformat()
    author_lookup = build_author_lookup(authors, config.author_output_path)

    if not quotes and not authors:
        logging.info(
            "No quotes or authors found. Writing empty outputs to %s and %s",
            config.quotes_output_path,
            config.author_output_path,
        )
        write_quotes_jsonl(
            [],
            config.quotes_output_path,
            start_url,
            scraped_at,
            author_lookup,
            config.collection_name,
            config.auto_increment_collection,
        )
        write_authors_jsonl([], config.author_output_path, start_url, scraped_at)
        return

    write_quotes_jsonl(
        quotes,
        config.quotes_output_path,
        start_url,
        scraped_at,
        author_lookup,
        config.collection_name,
        config.auto_increment_collection,
    )
    write_authors_jsonl(authors, config.author_output_path, start_url, scraped_at)
    logging.info(
        "Wrote %s quotes to %s; %s authors to %s",
        len(quotes),
        config.quotes_output_path,
        len(authors),
        config.author_output_path,
    )


if __name__ == "__main__":
    main()
