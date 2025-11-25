"""
Scrape quotes/authors from a target site, enrich with metadata, and write JSONL outputs.
Separation of concerns: models/settings/pipeline/utils live in their own modules; this file
focuses on scraping logic, enrichment assembly, and writing outputs.
"""

import argparse
import hashlib
import json
import logging
import math
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .models import DEFAULT_CONFIG_PATH, AuthorRecord, Config, QuoteContext, QuoteRecord
from .pipeline import QuoteMetadataPipeline
from .settings import load_config
from .utils import (
    _ordinal_suffix,
    build_dedupe_key,
    compute_quality_score,
    detect_safety_flags,
    slugify,
)


def normalize_url(url: str) -> str:
    """Normalize URLs to reduce duplicates (strip fragments, trim trailing slash except root)."""
    cleaned, _ = urldefrag(url)
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    path = parsed.path or "/"
    if path.endswith("/") and path != "/":
        path = path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", parsed.query, ""))


def derive_page_number(url: str) -> int:
    """Extract page number from path/query when present; default 1."""
    parsed = urlparse(url)
    match = re.search(r"/page/(\d+)/?", parsed.path)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    query_match = re.search(r"(?:^|[?&])page=(\d+)", parsed.query)
    if query_match:
        try:
            return int(query_match.group(1))
        except ValueError:
            pass
    return 1


def fetch_html(url: str, config: Config) -> str:
    """Fetch raw HTML from a URL with configured headers/timeout."""
    headers = {"User-Agent": config.user_agent}
    response = requests.get(url, headers=headers, timeout=config.http_timeout)
    response.raise_for_status()
    return response.text


def extract_links(html: str, base_url: str, allowed_netloc: str | None) -> List[str]:
    """Return normalized link URLs found in anchor tags."""
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        resolved = urljoin(base_url, href)
        cleaned = normalize_url(resolved)
        if not cleaned:
            continue
        link_netloc = urlparse(cleaned).netloc.lower()
        if allowed_netloc and link_netloc != allowed_netloc:
            continue
        links.append(cleaned)
    return links


def extract_quotes(html: str, page_url: str, depth: int, language_hint: str | None = None) -> List[QuoteRecord]:
    """Extract quotes, authors, and tags from a page if present."""
    soup = BeautifulSoup(html, "html.parser")
    quotes: List[QuoteRecord] = []
    page_number = derive_page_number(page_url)
    for idx, block in enumerate(soup.find_all("div", class_="quote"), start=1):
        quote_text = (block.find("span", class_="text").get_text() if block.find("span", class_="text") else "").strip()
        author = (block.find("small", class_="author").get_text() if block.find("small", class_="author") else "").strip()
        tags = [tag.get_text().strip() for tag in block.find_all("a", class_="tag")]
        if quote_text:
            quotes.append(
                QuoteRecord(
                    source_url=page_url,
                    quote=quote_text,
                    author=author,
                    tags=tags,
                    depth=depth,
                    page_number=page_number,
                    position_on_page=idx,
                    language_hint=language_hint,
                )
            )
    return quotes


def extract_author_details(html: str, page_url: str, depth: int) -> AuthorRecord | None:
    """Extract author details from an author page if present."""
    soup = BeautifulSoup(html, "html.parser")
    details = soup.find("div", class_="author-details")
    if not details:
        return None

    name_el = details.find(["h3", "h1"], class_="author-title") or details.find(["h3", "h1"])
    name = (name_el.get_text() if name_el else "").strip()
    born_date_el = details.find(class_="author-born-date")
    born_location_el = details.find(class_="author-born-location")
    description_el = details.find("div", class_="author-description")

    born_date = (born_date_el.get_text() if born_date_el else "").strip()
    born_location = (born_location_el.get_text() if born_location_el else "").strip()
    description = (description_el.get_text() if description_el else "").strip()

    if not name:
        return None

    return AuthorRecord(
        source_url=page_url,
        name=name,
        born_date=born_date,
        born_location=born_location,
        description=description,
        depth=depth,
    )


def should_skip_url(url: str, skip_keywords: List[str]) -> bool:
    """Return True if URL matches any skip keywords in path/query."""
    parsed = urlparse(url)
    path_and_query = f"{parsed.path}?{parsed.query}".lower()
    return any(keyword in path_and_query for keyword in skip_keywords)


def matches_include_patterns(url: str, patterns: List[str]) -> bool:
    """Return True if URL matches any include pattern; if none provided, allow all."""
    if not patterns:
        return True
    return any(re.search(pattern, url) for pattern in patterns)


def detect_language_from_html(html: str) -> str | None:
    """Extract language code from <html lang> or meta language tags, if present."""
    soup = BeautifulSoup(html, "html.parser")
    lang = soup.html.get("lang") if soup.html else None
    if lang:
        return lang.split("-")[0].lower()
    meta_lang = soup.find("meta", attrs={"http-equiv": "Content-Language"}) or soup.find(
        "meta", attrs={"name": "language"}
    )
    if meta_lang and meta_lang.get("content"):
        return meta_lang["content"].split("-")[0].lower()
    return None


def fallback_detect_language(text: str) -> str:
    """Very lightweight language guess: defaults to en unless many non-ASCII letters."""
    if not text:
        return "unknown"
    ascii_ratio = sum(1 for c in text if c.isascii()) / max(len(text), 1)
    return "en" if ascii_ratio > 0.8 else "unknown"


def load_existing_records(path: Path) -> List[dict]:
    """Load existing JSONL records from disk if present."""
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def compute_text_features(text: str) -> Dict[str, object]:
    """Compute lightweight text features for AI enrichment (counts, flags, language guess)."""
    normalized_text = " ".join(text.split())
    quote_normalized = normalized_text.replace("“", '"').replace("”", '"').replace("’", "'")
    words = normalized_text.split()
    word_count = len(words)
    char_count = len(normalized_text)
    token_count = len(re.findall(r"\S+", normalized_text))
    avg_word_length = (sum(len(w) for w in words) / word_count) if word_count else 0
    ascii_ratio = sum(1 for c in normalized_text if c.isascii()) / max(len(normalized_text), 1)
    language = "en" if ascii_ratio > 0.8 else "unknown"
    language_confidence = round(ascii_ratio, 3)
    has_punctuation_issues = ascii_ratio < 0.7 or any(ord(c) < 32 for c in normalized_text if c not in "\n\t\r")
    has_ellipsis = "..." in normalized_text or "… " in normalized_text
    has_question_mark = "?" in normalized_text
    has_exclamation_mark = "!" in normalized_text

    return {
        "normalized_text": normalized_text.lower(),
        "quote_normalized": quote_normalized,
        "word_count": word_count,
        "char_count": char_count,
        "token_count": token_count,
        "avg_word_length": avg_word_length,
        "language": language,
        "language_confidence": language_confidence,
        "has_punctuation_issues": has_punctuation_issues,
        "has_ellipsis": has_ellipsis,
        "has_question_mark": has_question_mark,
        "has_exclamation_mark": has_exclamation_mark,
    }


def build_mini_embedding(text: str, dims: int = 16) -> List[float]:
    """Tiny deterministic embedding via hashing; suitable as a PoC vector."""
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    while len(digest) < dims * 2:
        digest += hashlib.sha1(digest).digest()
    vec: List[float] = []
    for i in range(dims):
        segment = digest[i * 2 : i * 2 + 2]
        val = int.from_bytes(segment, "big", signed=False) / 65535.0
        vec.append(round(val, 6))
    return vec


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "to",
    "for",
    "with",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "it",
    "this",
    "that",
    "as",
    "but",
    "if",
    "into",
    "about",
    "than",
    "then",
    "so",
    "very",
    "too",
    "can",
    "will",
    "would",
    "could",
    "should",
}


def normalize_tags(tags: List[str]) -> List[str]:
    """Lowercase, slugify, and dedupe tags preserving order."""
    seen = set()
    normalized: List[str] = []
    for tag in tags:
        slug = slugify(tag)
        if slug and slug not in seen:
            seen.add(slug)
            normalized.append(slug)
    return normalized


def extract_top_keywords(text: str, tags: List[str], limit: int = 5) -> List[str]:
    """Simple keyword extraction: combine tags + top non-stopword tokens by length/frequency."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())
    freq: Dict[str, int] = {}
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        freq[tok] = freq.get(tok, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    keywords = [t for t, _ in sorted_tokens[: limit - len(tags)]]
    combined = []
    seen = set()
    for candidate in tags + keywords:
        slug = slugify(candidate)
        if slug and slug not in seen:
            seen.add(slug)
            combined.append(slug)
        if len(combined) >= limit:
            break
    return combined


def infer_topic_label(text: str, tags: List[str]) -> str:
    """Heuristic topic labeling based on keywords."""
    haystack = (text + " " + " ".join(tags)).lower()
    topic_keywords = {
        "love": ["love", "heart"],
        "life": ["life", "living"],
        "success": ["success", "achievement"],
        "philosophy": ["truth", "mind", "thought"],
        "humor": ["funny", "humor", "laugh"],
        "friendship": ["friend", "friends", "friendship"],
        "inspiration": ["inspire", "inspiration", "dream", "hope"],
    }
    for topic, keys in topic_keywords.items():
        if any(key in haystack for key in keys):
            return topic
    return "general"


def infer_emotion_label(text: str) -> str:
    """Very light sentiment-like guess using keyword cues."""
    lower = text.lower()
    positive = ["love", "hope", "joy", "happy", "beauty", "dream"]
    negative = ["fear", "sad", "hate", "pain", "anger"]
    if any(word in lower for word in positive):
        return "positive"
    if any(word in lower for word in negative):
        return "negative"
    return "neutral"


def infer_structural_features(text: str, tags: List[str]) -> Dict[str, object]:
    """Infer simple structural/role labels from text."""
    lower = text.lower()
    quote_type = infer_quote_type(text, tags)
    perspective = infer_perspective(lower)
    tense = infer_tense(lower)
    named_entities = extract_named_entities(text)
    contains_named_entities = bool(named_entities)
    return {
        "quote_type": quote_type,
        "perspective": perspective,
        "tense": tense,
        "contains_named_entities": contains_named_entities,
        "named_entities": named_entities,
    }


def infer_quote_type(text: str, tags: List[str]) -> str:
    """Heuristic quote type."""
    lower = text.lower()
    haystack = lower + " " + " ".join(tags).lower()
    mapping = {
        "humorous": ["funny", "humor", "laugh", "ridiculous"],
        "inspirational": ["inspire", "dream", "hope", "believe"],
        "philosophical": ["truth", "thought", "mind", "reality"],
        "motivational": ["success", "achieve", "do it", "keep", "never give up"],
    }
    for label, cues in mapping.items():
        if any(cue in haystack for cue in cues):
            return label
    if "love" in haystack:
        return "inspirational"
    return "general"


def infer_perspective(lower_text: str) -> str:
    """Heuristic person perspective."""
    first = {" i ", " my ", " mine ", " we ", " our ", " us "}
    second = {" you ", " your ", " yours "}
    third = {" he ", " she ", " they ", " him ", " her ", " them ", " his ", " hers ", " theirs "}
    padded = f" {lower_text} "
    if any(tok in padded for tok in first):
        return "first_person"
    if any(tok in padded for tok in second):
        return "second_person"
    if any(tok in padded for tok in third):
        return "third_person"
    return "unspecified"


def infer_tense(lower_text: str) -> str:
    """Rough tense guess."""
    if re.search(r"\bwill\b|\bgoing to\b", lower_text):
        return "future"
    if re.search(r"\b(ed|was|were)\b", lower_text):
        return "past"
    return "present"


def extract_named_entities(text: str) -> List[Dict[str, str]]:
    """Simple capitalized-word entity grabber as a placeholder."""
    candidates = re.findall(r"\b[A-Z][a-zA-Z']+\b", text)
    entities = []
    seen = set()
    for cand in candidates:
        key = cand.lower()
        if key not in seen:
            seen.add(key)
            entities.append({"type": "UNKNOWN", "text": cand})
    return entities


def determine_parse_status(text_features: Dict[str, object]) -> str:
    """Classify parse outcome."""
    if text_features["word_count"] == 0:
        return "failed"
    if text_features["has_punctuation_issues"]:
        return "partial"
    return "ok"


def make_document_id(source_url: str) -> str:
    """Stable document ID from source URL."""
    return hashlib.sha1(source_url.encode("utf-8")).hexdigest()


def apply_sequence_links(payloads: List[dict]) -> None:
    """Add prev/next quote ids within the same document ordered by page/position."""
    groups: Dict[str, List[dict]] = {}
    for payload in payloads:
        doc_id = payload.get("document_id", "")
        groups.setdefault(doc_id, []).append(payload)

    for doc_id, items in groups.items():
        items.sort(
            key=lambda p: (
                p.get("page_number", 0),
                p.get("position_on_page", 0),
            )
        )
        for idx, item in enumerate(items):
            item["chunk_index"] = idx + 1
            item["chunk_id"] = f"{doc_id}::{item['chunk_index']}"
            item["prev_quote_id"] = items[idx - 1]["id"] if idx > 0 else None
            item["next_quote_id"] = items[idx + 1]["id"] if idx + 1 < len(items) else None


def build_author_lookup(authors: List[AuthorRecord], author_path: Path) -> Dict[str, dict]:
    """Combine current and existing authors to enrich quote records."""
    lookup: Dict[str, dict] = {}

    def add_author(author_name: str, born_date: str, born_location: str) -> None:
        birth_year = extract_year(born_date)
        country = extract_country(born_location)
        era = derive_era(birth_year)
        lookup[author_name] = {
            "birth_year": birth_year,
            "death_year": None,
            "country": country,
            "era": era,
        }

    existing = load_existing_records(author_path)
    for row in existing:
        add_author(row.get("author_name_raw", "") or row.get("name", ""), row.get("born_date", ""), row.get("born_location", ""))

    for author in authors:
        add_author(author.name, author.born_date, author.born_location)

    return lookup


def extract_year(date_text: str) -> int | None:
    """Extract first 4-digit year."""
    match = re.search(r"(1[0-9]{3}|20[0-9]{2})", date_text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def extract_country(location_text: str) -> str | None:
    """Heuristic country extraction from location string."""
    if not location_text:
        return None
    parts = [p.strip() for p in location_text.split(",") if p.strip()]
    return parts[-1] if parts else None


def derive_era(year: int | None) -> str | None:
    """Map birth year to coarse era."""
    if year is None:
        return None
    century = (year // 100) + 1
    suffix = _ordinal_suffix(century)
    return f"{century}{suffix}_century"


def resolve_collection_name(configured: str, auto_increment: bool, existing: List[dict]) -> str:
    """Resolve collection name, optionally auto-incrementing based on existing data."""
    base, initial_version = parse_collection_name(configured)
    if not auto_increment:
        return f"{base}_v{initial_version}"

    versions: List[int] = []
    for row in existing:
        name = row.get("collection_name") or ""
        base_match, version = parse_collection_name(name)
        if base_match == base and version:
            versions.append(version)
    next_version = max(versions) + 1 if versions else initial_version
    return f"{base}_v{next_version}"


def parse_collection_name(name: str) -> tuple[str, int]:
    """Split collection name into base and version, defaulting sensibly."""
    if not name:
        return "collection", 1
    match = re.match(r"^(.*)_v(\\d+)$", name)
    if match:
        base = match.group(1) or "collection"
        return base, int(match.group(2))
    return name, 1


def write_quotes_jsonl(
    quotes: List[QuoteRecord],
    path: Path,
    start_url: str,
    scraped_at: str,
    author_lookup: Dict[str, dict],
    collection_name: str,
    auto_increment: bool,
) -> None:
    """Write quote records to a JSONL file, merging with any existing entries (idempotent)."""
    existing = load_existing_records(path)
    merged: Dict[Tuple[str, str], dict] = {}
    dedupe_counts: Dict[str, int] = {}
    for row in existing:
        key = (row.get("quote", ""), row.get("author") or row.get("author_name_raw", ""))
        merged[key] = row
        dedupe_key_existing = row.get("dedupe_key") or build_dedupe_key(row.get("normalized_text") or row.get("quote", ""))
        if dedupe_key_existing:
            dedupe_counts[dedupe_key_existing] = dedupe_counts.get(dedupe_key_existing, 0) + 1

    pipeline = QuoteMetadataPipeline.default()
    resolved_collection = resolve_collection_name(collection_name, auto_increment, existing)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in quotes:
            text_features = compute_text_features(record.quote)
            if record.language_hint and record.language_hint != "unknown":
                text_features["language"] = record.language_hint
                text_features["language_confidence"] = 1.0
            tags_normalized = normalize_tags(record.tags)
            source_site = urlparse(record.source_url).netloc
            embedding = build_mini_embedding(text_features["normalized_text"])
            top_keywords = extract_top_keywords(text_features["normalized_text"], record.tags)
            topic_label = infer_topic_label(text_features["normalized_text"], record.tags)
            emotion_label = infer_emotion_label(text_features["normalized_text"])
            structural = infer_structural_features(record.quote, record.tags)
            dedupe_key = build_dedupe_key(text_features["normalized_text"])
            is_duplicate = dedupe_counts.get(dedupe_key, 0) > 0
            dedupe_counts[dedupe_key] = dedupe_counts.get(dedupe_key, 0) + 1
            source_confidence = 0.6 if text_features["has_punctuation_issues"] else 1.0
            parse_status = determine_parse_status(text_features)
            quality_score = compute_quality_score(text_features)
            safety_flags = detect_safety_flags(text_features["normalized_text"])
            document_id = make_document_id(record.source_url)
            author_meta = author_lookup.get(record.author, {})
            primary_tag = record.tags[0] if record.tags else None
            secondary_tags = record.tags[1:] if len(record.tags) > 1 else []
            ctx = QuoteContext(
                record=record,
                author_meta=author_meta,
                text_features=text_features,
                structural=structural,
                dedupe_key=dedupe_key,
                is_duplicate=is_duplicate,
                source_confidence=source_confidence,
                parse_status=parse_status,
                quality_score=quality_score,
                safety_flags=safety_flags,
                embedding=embedding,
                top_keywords=top_keywords,
                topic_label=topic_label,
                emotion_label=emotion_label,
                document_id=document_id,
                collection_name=resolved_collection,
                scraped_at=scraped_at,
                start_url=start_url,
                source_site=source_site,
                tags_normalized=tags_normalized,
                tags_count=len(record.tags),
                primary_tag=primary_tag,
                secondary_tags=secondary_tags,
            )
            merged[(record.quote, record.author)] = pipeline.build(ctx)

        payloads = list(merged.values())
        apply_sequence_links(payloads)
        for payload in payloads:
            handle.write(json.dumps(payload))
            handle.write("\n")


def write_authors_jsonl(authors: List[AuthorRecord], path: Path, start_url: str, scraped_at: str) -> None:
    """Write author records to a JSONL file, merging with any existing entries (idempotent)."""
    existing = load_existing_records(path)
    merged: Dict[str, dict] = {}
    for row in existing:
        key = row.get("source_url")
        if key:
            merged[key] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in authors:
            word_count = len(record.description.split())
            char_count = len(record.description)
            language = fallback_detect_language(record.description)
            read_time_seconds = int(math.ceil((word_count / 200) * 60)) if word_count > 0 else 0
            author_norm = slugify(record.name)
            source_site = urlparse(record.source_url).netloc
            author_id = f"author-{hashlib.sha1(f'{record.name}::{record.source_url}'.encode('utf-8')).hexdigest()}"
            payload = {
                "id": author_id,
                "type": "author",
                "source_url": record.source_url,
                "start_url": start_url,
                "source_site": source_site,
                "author_name_raw": record.name,
                "author_normalized_id": author_norm,
                "born_date": record.born_date,
                "born_location": record.born_location,
                "description": record.description,
                "depth": record.depth,
                "scraped_at": scraped_at,
                "fetched_at": scraped_at,
                "crawl_timestamp": scraped_at,
                "word_count": word_count,
                "char_count": char_count,
                "language": language,
                "content_type": "author_bio",
                "estimated_read_time_seconds": read_time_seconds,
            }
            merged[record.source_url] = payload

        for payload in merged.values():
            handle.write(json.dumps(payload))
            handle.write("\n")


def scrape(url: str, config: Config) -> tuple[List[QuoteRecord], List[AuthorRecord]]:
    """Fetch pages breadth-first, extract quotes and authors, respecting depth/filters."""
    start_url = normalize_url(url)
    if not start_url:
        raise ValueError("Start URL must be http(s) and valid")
    start_netloc = urlparse(start_url).netloc.lower()
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    visited = {start_url}
    quotes: List[QuoteRecord] = []
    authors: List[AuthorRecord] = []
    author_pages_seen: set[str] = set()
    is_first_request = True
    pages_fetched = 0
    fetched_pages: set[str] = set()
    pbar = tqdm(total=config.max_pages, desc="Crawling", unit="page")
    retry_counts: Dict[str, int] = {}

    while queue:
        current_url, depth = queue.popleft()
        if pages_fetched >= config.max_pages:
            logging.info("Reached max_pages limit (%s); stopping crawl.", config.max_pages)
            break

        if not matches_include_patterns(current_url, config.include_patterns):
            logging.debug("Skipping %s; does not match include_patterns", current_url)
            continue

        if should_skip_url(current_url, config.skip_keywords):
            logging.debug("Skipping URL due to skip_keywords: %s", current_url)
            continue

        if not is_first_request and config.request_delay > 0:
            time.sleep(config.request_delay)

        if current_url in fetched_pages:
            logging.debug("Already fetched %s; skipping.", current_url)
            continue

        try:
            html = fetch_html(current_url, config)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else "unknown"
            retries = retry_counts.get(current_url, 0)
            if retries < config.max_retries:
                retry_counts[current_url] = retries + 1
                queue.append((current_url, depth))
                logging.warning(
                    "HTTP %s for %s, retrying (%s/%s)",
                    status,
                    current_url,
                    retries + 1,
                    config.max_retries,
                )
            else:
                logging.error("HTTP %s for %s, giving up after %s attempts", status, current_url, retries)
            continue
        except requests.RequestException as exc:
            retries = retry_counts.get(current_url, 0)
            if retries < config.max_retries:
                retry_counts[current_url] = retries + 1
                queue.append((current_url, depth))
                logging.warning("Request error for %s, retrying (%s/%s): %s", current_url, retries + 1, config.max_retries, exc)
            else:
                logging.error("Request error for %s, giving up after %s attempts: %s", current_url, retries, exc)
            continue

        is_first_request = False
        pages_fetched += 1
        pbar.update(1)

        fetched_pages.add(current_url)

        links = extract_links(html, current_url, start_netloc)
        doc_lang = detect_language_from_html(html) or fallback_detect_language(html)
        page_quotes = extract_quotes(html, current_url, depth, doc_lang)
        author_details = extract_author_details(html, current_url, depth)
        quotes.extend(page_quotes)
        if author_details and author_details.source_url not in author_pages_seen:
            authors.append(author_details)
            author_pages_seen.add(author_details.source_url)

        if depth >= config.crawl_depth:
            continue

        next_depth = depth + 1
        for link in links:
            link_netloc = urlparse(link).netloc.lower()
            if link_netloc != start_netloc:
                continue
            if not matches_include_patterns(link, config.include_patterns):
                continue
            if should_skip_url(link, config.skip_keywords):
                continue
            if link not in visited:
                visited.add(link)
                queue.append((link, next_depth))

    pbar.close()
    return quotes, authors


def main() -> None:
    """CLI entrypoint retained for convenience; main.py is preferred."""
    parser = argparse.ArgumentParser(description="Minimal web scraper that extracts links.")
    parser.add_argument("url", nargs="?", default="https://quotes.toscrape.com", help="URL to scrape")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config file (default: config/config.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO), format="%(levelname)s: %(message)s")
    quotes, authors = scrape(args.url, config)
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
            args.url,
            scraped_at,
            author_lookup,
            config.collection_name,
            config.auto_increment_collection,
        )
        write_authors_jsonl([], config.author_output_path, args.url, scraped_at)
        return

    write_quotes_jsonl(
        quotes,
        config.quotes_output_path,
        args.url,
        scraped_at,
        author_lookup,
        config.collection_name,
        config.auto_increment_collection,
    )
    write_authors_jsonl(authors, config.author_output_path, args.url, scraped_at)
    logging.info(
        "Wrote %s quotes to %s; %s authors to %s",
        len(quotes),
        config.quotes_output_path,
        len(authors),
        config.author_output_path,
    )


if __name__ == "__main__":
    main()
