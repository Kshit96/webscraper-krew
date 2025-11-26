# Build a self-contained image to run the scraper or analytics dashboard.
# Base build (lean, hash embeddings, heuristic NER):
#   docker build -t webscraper-krew .
# Build with optional NER and/or MiniLM embeddings:
#   docker build --build-arg EXTRAS="ner,embeddings" -t webscraper-krew:full .
# Run scraper (mount output to persist):
#   docker run --rm -e MODE=scrape -e TARGET_URL="https://quotes.toscrape.com" \
#     -v "$(pwd)/output:/app/output" webscraper-krew
# Run dashboard:
#   docker run --rm -p 8501:8501 -v "$(pwd)/output:/app/output" webscraper-krew
#
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG EXTRAS=""

COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    if echo "$EXTRAS" | grep -qi "ner"; then pip install --no-cache-dir -r requirements-ner.txt; fi && \
    if echo "$EXTRAS" | grep -qi "embeddings"; then pip install --no-cache-dir -r requirements-embeddings.txt; fi

COPY src ./src
COPY analytics_dashboard.py .
COPY config ./config
COPY output ./output
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "analytics_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
