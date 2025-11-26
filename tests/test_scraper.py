import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from webscraper_krew import scraper as s


def test_normalize_url_and_page_number():
    assert s.normalize_url("https://example.com/path/#fragment") == "https://example.com/path"
    assert s.normalize_url("https://example.com/path/") == "https://example.com/path"
    assert s.derive_page_number("https://example.com/tag/test/page/3/") == 3
    assert s.derive_page_number("https://example.com/?page=2") == 2
    assert s.derive_page_number("https://example.com/") == 1


def test_compute_text_features_and_embedding():
    text = "“Hello world!”  Hello"
    feats = s.compute_text_features(text)
    assert feats["word_count"] >= 4
    assert feats["char_count"] == len("“Hello world!” Hello")
    assert feats["has_exclamation_mark"] is True
    assert feats["normalized_text"] == "“hello world!” hello"
    emb1 = s.build_mini_embedding(feats["normalized_text"])
    emb2 = s.build_mini_embedding(feats["normalized_text"])
    assert len(emb1) == 16
    assert emb1 == emb2  # deterministic


def test_keywords_topic_emotion_structural():
    text = "I love success and dreams!"
    tags = ["Inspiration", "success"]
    keywords = s.extract_top_keywords(text, tags, limit=4)
    assert "inspiration" in keywords and "success" in keywords
    assert s.infer_topic_label(text, tags) in {"inspiration", "success", "general", "love"}
    assert s.infer_emotion_label(text) == "positive"
    structural = s.infer_structural_features(text, tags)
    assert structural["quote_type"] in {"inspirational", "general"}
    assert structural["perspective"] in {"first_person", "second_person", "third_person", "unspecified"}
    assert structural["contains_named_entities"] in {True, False}


def test_extract_quotes_and_authors():
    html = """
    <div class="quote">
      <span class="text">“Quote one.”</span>
      <small class="author">Author One</small>
      <div class="tags">
        <a class="tag">Life</a>
        <a class="tag">Happy</a>
      </div>
    </div>
    <div class="quote">
      <span class="text">“Quote two.”</span>
      <small class="author">Author Two</small>
      <div class="tags">
        <a class="tag">Humor</a>
      </div>
    </div>
    """
    quotes = s.extract_quotes(html, "https://example.com/?page=2", depth=0)
    assert len(quotes) == 2
    assert quotes[0].page_number == 2
    assert quotes[0].position_on_page == 1
    # include pattern blocking
    cfg = s.Config(
        user_agent="ua",
        http_timeout=5,
        quotes_output_path=Path("out"),
        author_output_path=Path("out"),
        collection_name="col",
        crawl_depth=0,
        request_delay=0,
        max_pages=10,
        skip_keywords=[],
        max_retries=0,
        include_patterns=[r"/tag/"],
    )
    assert s.matches_include_patterns("https://example.com/tag/test", cfg.include_patterns)
    assert not s.matches_include_patterns("https://example.com/author/test", cfg.include_patterns)

    author_html = """
    <div class="author-details">
      <h3 class="author-title">Jane Doe</h3>
      <span class="author-born-date">January 1, 1900</span>
      <span class="author-born-location">in City, Country</span>
      <div class="author-description">Bio text here.</div>
    </div>
    """
    author = s.extract_author_details(author_html, "https://example.com/author/Jane-Doe", depth=1)
    assert author is not None
    assert author.name == "Jane Doe"
    assert author.born_location.endswith("Country")


def test_extract_quotes_fallback_selectors():
    html = """
    <div class="quote" lang="en">
      <blockquote>“Alt quote.”</blockquote>
      <span class="author">Alt Author</span>
      <a rel="tag">AltTag</a>
    </div>
    """
    quotes = s.extract_quotes(html, "https://example.com/page/2", depth=0)
    assert len(quotes) == 1
    q = quotes[0]
    assert q.quote == "“Alt quote.”"
    assert q.author == "Alt Author"
    assert q.tags == ["AltTag"]
    assert q.source_html_lang is None or q.source_html_lang == "en"


def test_extract_author_fallback():
    author_html = """
    <article>
      <h1>Fallback Author</h1>
      <p>Born 1901</p>
      <p>in City, Wonderland</p>
      <p>Some description.</p>
    </article>
    """
    author = s.extract_author_details(author_html, "https://example.com/author/fallback", depth=1)
    assert author is not None
    assert author.name == "Fallback Author"


def test_write_quotes_jsonl_idempotent(tmp_path: Path):
    # Seed existing file with one record to trigger dedupe flag
    existing = {
        "quote": "alpha quote",
        "author": "Author",
        "dedupe_key": s.build_dedupe_key("alpha quote"),
        "id": "existing",
    }
    quotes_path = tmp_path / "quotes.jsonl"
    quotes_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    qr1 = s.QuoteRecord(
        source_url="https://example.com/page/1",
        quote="Alpha quote",
        author="Author",
        tags=["tag1", "tag2"],
        depth=0,
        page_number=1,
        position_on_page=2,
    )
    qr2 = s.QuoteRecord(
        source_url="https://example.com/page/1",
        quote="Beta quote",
        author="Author",
        tags=["tag3"],
        depth=0,
        page_number=1,
        position_on_page=1,
    )
    scraped_at = "2024-01-01T00:00:00Z"
    author_lookup = {
        "Author": {
            "country": "Country",
            "birth_year": 1900,
            "death_year": None,
            "era": "20th_century",
        }
    }

    s.write_quotes_jsonl(
        [qr1, qr2],
        quotes_path,
        "https://example.com",
        scraped_at,
        author_lookup,
        "collection_v1",
        False,
    )

    lines = [json.loads(l) for l in quotes_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 3  # existing + two new
    alpha = next(l for l in lines if l.get("quote", "").startswith("Alpha"))
    beta = next(l for l in lines if l.get("quote", "").startswith("Beta"))
    for entry in (alpha, beta):
        assert "chunk_id" in entry and "collection_name" in entry
        assert entry["document_id"]
        assert entry["quality_score"] >= 0
    # Sequence links ordered by position_on_page within the document
    assert beta["chunk_index"] == 1
    assert beta["next_quote_id"] == alpha["id"]
    assert alpha["prev_quote_id"] == beta["id"]
    # Dedupe flag should be true because of seeded record
    assert alpha["is_duplicate"] is True


def test_build_author_lookup_and_era(tmp_path: Path):
    author = s.AuthorRecord(
        source_url="https://example.com/author/jane",
        name="Jane Doe",
        born_date="December 12, 1899",
        born_location="City, Wonderland",
        description="Bio",
        depth=0,
    )
    author_future = s.AuthorRecord(
        source_url="https://example.com/author/future",
        name="Future Person",
        born_date="January 1, 2001",
        born_location="City, Futureland",
        description="Bio",
        depth=0,
    )
    author_path = tmp_path / "authors.jsonl"
    author_lookup = s.build_author_lookup([author, author_future], author_path)
    meta = author_lookup["Jane Doe"]
    assert meta["birth_year"] == 1899
    assert meta["era"] == "19th_century"
    assert meta["country"] == "Wonderland"
    meta_future = author_lookup["Future Person"]
    assert meta_future["birth_year"] == 2001
    assert meta_future["era"] == "21st_century"
    assert meta_future["country"] == "Futureland"
