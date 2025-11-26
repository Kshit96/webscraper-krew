# Webscraper Krew

Minimal Python starter for crawling a site, collecting quotes and author details.

## Quickstart
- Python 3.10+ recommended.
- Create a virtual environment and install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Run the scraper:
  ```bash
  PYTHONPATH=src python -m src.webscraper_krew.scraper "https://example.com"
  # or with a custom config
  PYTHONPATH=src python -m src.webscraper_krew.scraper "https://example.com" --config path/to/config.json
  ```
- Outputs: JSONL files at `quotes_output_path` and `author_output_path` (configurable). Runs are idempotent (merge on quote+author or source_url).

## Configuration
Defaults in `config/config.json`:
```json
{
  "user_agent": "webscraper-krew/0.1 (+https://example.com)",
  "http_timeout": 10,
  "log_level": "INFO",
  "quotes_output_path": "output/quotes.jsonl",
  "author_output_path": "output/authors.jsonl",
  "crawl_depth": 0,
  "request_delay": 1.0,
  "max_pages": 50,
  "skip_keywords": ["login", "signin", "signup", "search", "admin"],
  "max_retries": 1,
  "include_patterns": [],
  "follow_include_patterns": [],
  "auto_increment_collection": true
}
```
- `crawl_depth`, `request_delay`, `max_pages`, `skip_keywords`: control crawl scope and politeness.
- `include_patterns`: optional glob/regex filters applied to discovered links (start URL always fetched).
- `follow_include_patterns`: optional filters for discovered links; if set, overrides `include_patterns`.
- Only same-domain links are followed.

## Project Structure
- `src/webscraper_krew/main.py` — CLI entry; loads config, runs scrape, writes outputs.
- `src/webscraper_krew/scraper.py` — crawl, extraction, enrichment, writes JSONL.
- `src/webscraper_krew/models.py` — dataclasses for config, records, context.
- `src/webscraper_krew/settings.py` — config loading/validation.
- `src/webscraper_krew/pipeline.py` — `QuoteMetadataPipeline` (transforms context → payload).
- `src/webscraper_krew/context_factory.py` — builds `QuoteContext` from raw quotes + author lookup.
- `src/webscraper_krew/utils.py` — shared helpers (quality, dedupe, slugify, etc.).
- `analytics_dashboard.py` — Streamlit UI (control panel + analytics).
- `requirements*.txt` — core and optional deps (NER, embeddings).
- `config/config.json` — default runtime configuration.

### Optional extras
- NER model (dslim/bert-base-NER): `pip install -r requirements-ner.txt` (fallback heuristic if absent).
- Embeddings (MiniLM-L6 projected to 10D): `pip install -r requirements-embeddings.txt` (fallback hash embedding if absent).
- Output writes use a streamed merge-by-key (quote+author) to reduce memory usage; we rewrite via a temp file instead of holding the entire dataset in memory.
- Logging: detailed logs go to `logs/scrape.log` (major steps, warnings/errors); console stays quiet except for errors so the progress bar remains readable.

## Docker (dashboard + in-UI scrape)
- Build lean: `docker build -t webscraper-krew .`
- Build with models: `docker build --build-arg EXTRAS="ner,embeddings" -t webscraper-krew:full .`
- Run dashboard: `docker run -p 8501:8501 -v "$(pwd)/output:/app/output" webscraper-krew`
- Run scraper via UI control panel inside the dashboard (container entrypoint is Streamlit); mount `output/` to persist.
### Build/run matrix
- Lean (no NER/embeddings):  
  `docker build -t webscraper-krew .`
- NER only:  
  `docker build --build-arg EXTRAS="ner" -t webscraper-krew:ner .`
- Embeddings only:  
  `docker build --build-arg EXTRAS="embeddings" -t webscraper-krew:emb .`
- NER + Embeddings:  
  `docker build --build-arg EXTRAS="ner,embeddings" -t webscraper-krew:full .`
- Run dashboard for any image (example for full):  
  `docker run --rm -p 8501:8501 -v "$(pwd)/output:/app/output" webscraper-krew:full`

## Documentation (Short Write-Up)
### Site chosen and why
- `quotes.toscrape.com` — public demo site, predictable structure, safe for scraping and quick iteration.

### How to run (deps, commands, config)
- Dependencies: `pip install -r requirements.txt` (optional: `requirements-ner.txt` for NER, `requirements-embeddings.txt` for MiniLM).
- Commands: `PYTHONPATH=src python -m src.webscraper_krew.scraper "http://quotes.toscrape.com" --config config/config.json`
- Config: adjust depth/delay/max_pages/skip/include/follow patterns in `config/config.json` (or via the Streamlit control panel). Start URL is always fetched; include/follow patterns apply to discovered links.
- Docker: build lean or with models (see Docker section) and trigger scrapes from the dashboard UI.
- Pattern examples (applied to discovered links):
  - Allow all: `"include_patterns": [], "follow_include_patterns": []`
  - Only tag pages (glob): `"include_patterns": ["*/tag/*"]`
  - Only tag pages (regex): `"include_patterns": [".*/tag/.*"]`
  - Only tag pages via follow-only knob (leave include open): `"include_patterns": [], "follow_include_patterns": ["*/tag/*"]`

### Data schema (field names & meanings)
- Collection/IDs: `collection_name`, `id`, `document_id`, `chunk_id`, `chunk_index` — stable IDs/versioning; quote ordering within a page.
- Source: `source_url`, `start_url`, `source_site`, `page_title`, `scraped_at`, `last_updated_at`, `fetched_at`, `crawl_timestamp` — provenance and freshness.
- Quote content: `quote`, `author_name_raw`, `author_normalized_id`, `tags`, `tags_normalized`, `tags_count`, `primary_tag`, `secondary_tags`, `page_number`, `position_on_page`, `depth` — raw text, normalized author/tags, placement on site.
- Language: `resolved_lang`, `source_html_lang`, `source_node_lang`, `detected_lang`, `detected_lang_confidence`, `language`, `language_confidence` — language hints and final choice for routing/filters.
- Text features: `word_count`, `char_count`, `token_count`, `avg_word_length`, `normalized_text`, `quote_normalized`, `estimated_read_time_seconds`, `has_punctuation_issues`, `has_ellipsis`, `has_question_mark`, `has_exclamation_mark` — basic stats for QA/preprocessing.
- Structural/semantic: `quote_type`, `perspective`, `tense`, `contains_named_entities`, `named_entities`, `topic_label`, `emotion_label`, `top_keywords` — facets for filtering/analytics and retrieval cues.
- Embeddings: `embedding_vector`, `embedding_model` — similarity/RAG; hash by default, MiniLM-10D if installed.
- Quality/safety: `quality_score`, `safety_flags`, `parse_status`, `source_confidence` — filter low-quality or sensitive content.
- Duplicates: `dedupe_key`, `is_duplicate` — stable hash + flag to avoid double ingestion.
- Author meta: `author_country`, `author_birth_year`, `author_death_year`, `author_era` — enrich quotes with author context.
- Graph links: `prev_quote_id`, `next_quote_id` — navigate adjacent quotes within a document.
- Content type: `content_type` — classify payload (quote vs author_bio).

Use cases:
- RAG/search: embeddings + keywords/topics + language filters; chunk/document IDs for chunking; quality/safety gating.
- Analytics: tag/author/topic/emotion/quote_type distributions; author era/country; quality vs length; duplicate ratio.
- QA/debugging: parse_status, source_confidence, dedupe_key, positional fields (`page_number`, `position_on_page`, `depth`).

### Design decisions
- Pages to keep: same-domain only; skip non-content via `skip_keywords`; `include_patterns`/`follow_include_patterns` whitelist discovered links (start URL always fetched).
- Main content extraction: target `.quote` blocks (demo structure), capturing text/author/tags/page/position; page titles from `<title>` (or heading/meta fallback).
- AI collections workflow: normalization, dedupe, language/safety/topic/emotion/structural labels, embeddings, chunk/document IDs, collection versioning. Author country/era for richer filters.
- Pipeline + Factory relationship:
  - `context_factory.py` builds `QuoteContext` (raw quote, author metadata, text/structural features, safety/dedupe signals).
  - `pipeline.py` projects that context to the final JSON via small builder functions.
  - Extensibility: add a builder `def my_fields(ctx: QuoteContext) -> dict` and register it in `QuoteMetadataPipeline.default()` (or swap pipelines); extend `QuoteContextFactory` to compute new signals before the pipeline runs.

### Future Work
- Scheduling + monitoring: cron/worker with retries, metrics, alerts, and per-run audit logs.
- Cross-source dedupe: global dedupe keys across sites/domains.
- Smarter extraction: model-based content extraction, better NER, improved topic/emotion classifiers.
- Storage/indexing: push to a vector store and/or search index; versioned collections for A/B retrieval.
- Politeness/compliance: robots.txt checks, per-domain rate limits, backoff, and user-agent rotation if needed.
