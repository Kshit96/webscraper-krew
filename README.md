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

## Configuration
All runtime settings live in a JSON config file (default: `config/config.json`):
```json
{
  "user_agent": "webscraper-krew/0.1 (+https://example.com)",
  "http_timeout": 10,
  "log_level": "INFO",
  "output_path": "output/links.jsonl",
  "crawl_depth": 0,
  "request_delay": 1.0
}
```
- `user_agent`: string sent with each request.
- `http_timeout`: request timeout in seconds.
- `log_level`: one of DEBUG, INFO, WARNING, ERROR.
- `output_path`: where scraped links are written as JSONL.
- `crawl_depth`: breadth-first crawl depth (0 = only the start URL, 1 = follow its links once, etc.).
- `request_delay`: seconds to sleep between requests (global, applies across depths).

## Project Structure
- `src/webscraper_krew/scraper.py` — CLI entry point and scraping logic.
- `config/config.json` — default runtime configuration.
- `requirements.txt` — dependencies: `requests`, `beautifulsoup4`.
- `.gitignore` — common Python ignores and virtualenv.

## Next Steps
- Add parsing tailored to your target sites (e.g., article titles, metadata).
- Layer in polite scraping practices: rate limiting, retries, and robots.txt checks.
- Add tests (pytest) and CI once parsing logic grows.
