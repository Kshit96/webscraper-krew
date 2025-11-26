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
  "auto_increment_collection": true
}
```
- `crawl_depth`, `request_delay`, `max_pages`, `skip_keywords`, `include_patterns`: control crawl scope and politeness.
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

### How to run
- Dependencies: `pip install -r requirements.txt` (optional: `requirements-ner.txt`, `requirements-embeddings.txt`).
- Commands: `PYTHONPATH=src python -m src.webscraper_krew.scraper "http://quotes.toscrape.com" --config config/config.json`
- Config: tweak crawl depth, delays, max_pages, include/skip patterns in `config/config.json` (or via dashboard control panel).
- Docker: see commands above (lean vs. model-enabled builds).

### Data schema (key fields)
- Common: `id`, `type`, `source_url`, `start_url`, `source_site`, `collection_name`, `scraped_at`, `last_updated_at`, `dedupe_key`, `is_duplicate`, `parse_status`, `source_confidence`.
- Quote-specific: `quote`, `author_name_raw`, `author_normalized_id`, `tags`, `tags_normalized`, `tags_count`, `primary_tag`, `secondary_tags`, `page_number`, `position_on_page`, `depth`, `document_id`, `chunk_id`, `chunk_index`.
- Text/AI signals: `word_count`, `char_count`, `token_count`, `language`/`resolved_lang`, `language_confidence`, `normalized_text`, `quote_normalized`, `quality_score`, `safety_flags`, `topic_label`, `emotion_label`, `quote_type`, `perspective`, `tense`, `contains_named_entities`, `named_entities`, `top_keywords`, `embedding_vector` (hash or MiniLM PCA 10D), `embedding_model`.
- Author-specific: `author_country`, `author_birth_year`, `author_death_year`, `author_era`, plus bios in author JSONL.

### Design decisions
- Pages to keep: same-domain links only; skip obvious non-content (login/search/admin) via `skip_keywords`; optional include_patterns for whitelisting.
- Main content extraction: look for `.quote` blocks on the page (demo site structure), grabbing text, author, tags, page/position.
- AI workflow support: normalize text/tags, dedupe keys, language signals, safety flags, topic/emotion/structural labels, embedding vector, chunk/document IDs, collection versioning for incremental updates. Authors carry country/era for richer filters.
- Pipeline + Factory relationship:
  - `context_factory.py` builds a `QuoteContext` (raw quote, author metadata, text/structural features, safety/dedupe signals).
  - `pipeline.py` takes that context and emits the final JSON payload via small “builder” functions.
  - Extensibility: add a builder `def my_fields(ctx: QuoteContext) -> dict` and register it in `QuoteMetadataPipeline.default()` (or create a custom pipeline). You can also extend `QuoteContextFactory` if you need to compute new signals from raw HTML before the pipeline runs.

### Future Work
- Scheduling + monitoring: cron/worker with retries, metrics, alerts, and per-run audit logs.
- Cross-source dedupe: global dedupe keys across sites/domains.
- Smarter extraction: model-based content extraction, better NER, improved topic/emotion classifiers.
- Storage/indexing: push to a vector store and/or search index; versioned collections for A/B retrieval.
- Politeness/compliance: robots.txt checks, per-domain rate limits, backoff, and user-agent rotation if needed.
