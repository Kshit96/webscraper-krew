import argparse
import json
import logging
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import List
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_CONFIG_PATH = Path("config/config.json")


@dataclass
class Config:
    user_agent: str
    http_timeout: int
    log_level: str = "INFO"
    output_path: Path = Path("output/links.jsonl")
    crawl_depth: int = 0
    request_delay: float = 1.0
    max_pages: int = 50


@dataclass
class LinkRecord:
    source_url: str
    text: str
    href: str
    depth: int


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load scraper configuration from a JSON file."""
    config_path = path.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

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

    output_path_raw = data.get("output_path", "output/links.jsonl")
    output_path = Path(output_path_raw).expanduser()

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

    return Config(
        user_agent=user_agent,
        http_timeout=http_timeout,
        log_level=log_level,
        output_path=output_path,
        crawl_depth=crawl_depth,
        request_delay=request_delay,
        max_pages=max_pages,
    )


def fetch_html(url: str, config: Config) -> str:
    """Fetch raw HTML from a URL."""
    headers = {"User-Agent": config.user_agent}
    response = requests.get(url, headers=headers, timeout=config.http_timeout)
    response.raise_for_status()
    return response.text


def extract_links(html: str, base_url: str, depth: int, allowed_netloc: str | None) -> List[LinkRecord]:
    """Return list of link records found in anchor tags, limited to allowed domain when provided."""
    soup = BeautifulSoup(html, "html.parser")
    links: List[LinkRecord] = []
    for anchor in soup.find_all("a", href=True):
        text = (anchor.get_text() or "").strip()
        href = anchor["href"]
        resolved = urljoin(base_url, href)
        cleaned, _ = urldefrag(resolved)
        if not cleaned.startswith(("http://", "https://")):
            continue
        link_netloc = urlparse(cleaned).netloc.lower()
        if allowed_netloc and link_netloc != allowed_netloc:
            continue
        links.append(LinkRecord(source_url=base_url, text=text, href=cleaned, depth=depth))
    return links


def scrape(url: str, config: Config) -> List[LinkRecord]:
    """High-level scraper: fetch page and extract links with optional breadth-first crawling."""
    start_url = urldefrag(url)[0]
    start_netloc = urlparse(start_url).netloc.lower()
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    visited = {start_url}
    records: List[LinkRecord] = []
    is_first_request = True
    pages_fetched = 0

    while queue:
        current_url, depth = queue.popleft()
        if pages_fetched >= config.max_pages:
            logging.info("Reached max_pages limit (%s); stopping crawl.", config.max_pages)
            break

        if not is_first_request and config.request_delay > 0:
            time.sleep(config.request_delay)

        try:
            html = fetch_html(current_url, config)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping %s due to error: %s", current_url, exc)
            continue

        is_first_request = False
        pages_fetched += 1

        links = extract_links(html, current_url, depth, start_netloc)
        records.extend(links)

        if depth >= config.crawl_depth:
            continue

        next_depth = depth + 1
        for link in links:
            link_netloc = urlparse(link.href).netloc.lower()
            if link_netloc != start_netloc:
                continue
            if link.href not in visited:
                visited.add(link.href)
                queue.append((link.href, next_depth))

    return records


def write_jsonl(records: List[LinkRecord], path: Path) -> None:
    """Write link records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "source_url": record.source_url,
                "text": record.text,
                "href": record.href,
                "depth": record.depth,
            }
            handle.write(json.dumps(payload))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal web scraper that extracts links.")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config file (default: config/config.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO), format="%(levelname)s: %(message)s")
    logging.debug("Loaded config from %s", args.config)
    logging.debug("Using User-Agent: %s", config.user_agent)

    try:
        links = scrape(args.url, config)
    except Exception as exc:  # noqa: BLE001
        logging.error("Scrape failed: %s", exc)
        raise SystemExit(1) from exc

    if not links:
        logging.info("No links found. Writing empty output to %s", config.output_path)
        write_jsonl([], config.output_path)
        return

    write_jsonl(links, config.output_path)
    logging.info("Wrote %s links to %s", len(links), config.output_path)


if __name__ == "__main__":
    main()
