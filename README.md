# Webscraper Krew

Minimal Python starter for crawling a site, collecting quotes and author details.

## Setup
- Python 3.10+ recommended.
- Create a virtual environment and install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

## Usage
Run the scraper against a target URL:
```bash
PYTHONPATH=src python -m src.webscraper_krew.scraper "https://example.com"
# optionally point to another config file
PYTHONPATH=src python -m src.webscraper_krew.scraper "https://example.com" --config path/to/config.json
```
The scraper writes structured results as JSON Lines (one JSON object per line). Quotes include text, author, tags plus metadata (`start_url`, `scraped_at`, `fetched_at`, `depth`, `word_count`, `char_count`, `tags_count`, `language`, `content_type`, `estimated_read_time_seconds`); authors are written separately to `author_output_path` with bio details and the same style of metadata.

Runs are idempotent: existing quote/author JSONL files are merged by key (quote+author for quotes, source_url for authors), so reruns update/replace records rather than appending duplicates.

## Configuration
All runtime settings live in a JSON config file (default: `config/config.json`):
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
- `user_agent`: string sent with each request.
- `http_timeout`: request timeout in seconds.
- `log_level`: one of DEBUG, INFO, WARNING, ERROR.
- `quotes_output_path`: where scraped quotes are written as JSONL.
- `author_output_path`: where author records are written as JSONL.
- `crawl_depth`: breadth-first crawl depth (0 = only the start URL, 1 = follow its links once, etc.).
- `request_delay`: seconds to sleep between requests (global, applies across depths).
- `max_pages`: maximum pages to fetch per run (hard cap across depths).
- `skip_keywords`: skip URLs containing these keywords in path/query (e.g., login or search pages).
- `max_retries`: number of retries per URL on request/HTTP errors.
- `include_patterns`: only enqueue/fetch URLs matching these regex patterns (empty = allow all).
- `auto_increment_collection`: when true, bump collection version automatically across runs.
- The crawler only follows links on the same domain as the start URL.

## Extending Metadata (QuoteMetadataPipeline)
- Metadata for quotes is assembled by `QuoteMetadataPipeline` in `src/webscraper_krew/scraper.py`. To add your own fields:
  1. Write a processor function with signature `def my_processor(ctx: QuoteContext) -> dict` that returns a dict of new fields. `QuoteContext` exposes the parsed quote record plus derived features (text_features, structural info, author metadata, etc.).
  2. Register it in `QuoteMetadataPipeline.default()` (append to the processors list), or create a custom pipeline and replace the default call in `write_quotes_jsonl`.
  3. Ensure keys you emit do not collide with existing ones unless you intend to overwrite.
  Example:
  ```python
  def add_uppercase_quote(ctx: QuoteContext) -> dict:
      return {"quote_upper": ctx.record.quote.upper()}

  # inside QuoteMetadataPipeline.default():
  return cls([...existing..., add_uppercase_quote])
  ```

## Project Structure
- `src/webscraper_krew/main.py` — preferred CLI entry point: loads config, runs scrape, writes outputs.
- `src/webscraper_krew/scraper.py` — crawl, extraction, enrichment, and writing logic.
- `src/webscraper_krew/models.py` — dataclasses for config, records, and context.
- `src/webscraper_krew/settings.py` — config loading/validation.
- `src/webscraper_krew/pipeline.py` — `QuoteMetadataPipeline` and default processors.
- `src/webscraper_krew/utils.py` — shared helpers (quality, dedupe, slugify, etc.).
- `config/config.json` — default runtime configuration.
- `requirements.txt` — dependencies: `requests`, `beautifulsoup4`.
- `.gitignore` — common Python ignores and virtualenv.

## Docker (dashboard + in-UI scrape)
- Build: `docker build -t webscraper-krew .`
- Run the dashboard (mounting local output for live data):  
  `docker run -p 8501:8501 -v "$(pwd)/output:/app/output" webscraper-krew`
- Open http://localhost:8501 to view analytics and trigger scrapes via the UI control panel.

## Next Steps
- Add parsing tailored to your target sites (e.g., article titles, metadata).
- Layer in polite scraping practices: rate limiting, retries, and robots.txt checks.
- Add tests (pytest) and CI once parsing logic grows.
