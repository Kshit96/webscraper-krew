# Webscraper Krew

Minimal Python starter for scraping web pages and extracting links.

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
python -m webscraper_krew.scraper "https://example.com"
# optionally point to another config file
python -m webscraper_krew.scraper "https://example.com" --config path/to/config.json
```
The scraper writes structured results as JSON Lines (one JSON object per line) to the configured output path, including `source_url`, `text`, `href`, and `depth`.
Quotes include quote text, author, tags; authors are written separately to `author_output_path` with bio details when available.

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
  "max_pages": 50
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
- The crawler only records and follows links on the same domain as the start URL.

## Project Structure
- `src/webscraper_krew/scraper.py` — CLI entry point and scraping logic.
- `config/config.json` — default runtime configuration.
- `requirements.txt` — dependencies: `requests`, `beautifulsoup4`.
- `.gitignore` — common Python ignores and virtualenv.

## Next Steps
- Add parsing tailored to your target sites (e.g., article titles, metadata).
- Layer in polite scraping practices: rate limiting, retries, and robots.txt checks.
- Add tests (pytest) and CI once parsing logic grows.
