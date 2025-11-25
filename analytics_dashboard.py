"""
Local analytics dashboard for scraped quotes/authors.

Run with:
  streamlit run analytics_dashboard.py
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_config(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_scrape(target_url: str, config_path: Path) -> tuple[bool, str]:
    """Invoke scraper as a subprocess to keep behavior consistent with CLI."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent / "src")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "webscraper_krew.scraper", target_url, "--config", str(config_path)],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        return True, completed.stdout + completed.stderr
    except subprocess.CalledProcessError as exc:
        return False, exc.stdout + exc.stderr


def ensure_dataframe(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Quotes Analytics", layout="wide")
    st.title("Quotes Analytics Dashboard")

    base_dir = Path.cwd()
    quotes_path = st.sidebar.text_input(
        "Quotes JSONL path", value=str(base_dir / "output/quotes.jsonl")
    )
    authors_path = st.sidebar.text_input(
        "Authors JSONL path", value=str(base_dir / "output/authors.jsonl")
    )
    config_path = st.sidebar.text_input(
        "Config path", value=str(base_dir / "config/config.json")
    )
    target_url = st.sidebar.text_input("Target URL", value="https://quotes.toscrape.com")

    quotes_rows = load_jsonl(Path(quotes_path))
    authors_rows = load_jsonl(Path(authors_path))

    quotes_df = ensure_dataframe(quotes_rows)
    authors_df = ensure_dataframe(authors_rows)

    if quotes_df.empty:
        st.warning("No quotes data found. Scrape first, then reload.")
    config_data = load_config(Path(config_path))

    with st.sidebar.expander("Scraper Controls", expanded=False):
        crawl_depth = st.number_input("crawl_depth", min_value=0, max_value=10, value=int(config_data.get("crawl_depth", 1)))
        request_delay = st.number_input("request_delay", min_value=0.0, value=float(config_data.get("request_delay", 0.5)), step=0.1)
        max_pages = st.number_input("max_pages", min_value=1, value=int(config_data.get("max_pages", 200)))
        include_patterns = st.text_input(
            "include_patterns (comma-separated regex)",
            value=",".join(config_data.get("include_patterns", []) or []),
        )
        skip_keywords = st.text_input(
            "skip_keywords (comma-separated)",
            value=",".join(config_data.get("skip_keywords", []) or []),
        )
        run_button = st.button("Run scrape now")

    if run_button:
        new_config = config_data.copy()
        new_config.update(
            {
                "crawl_depth": int(crawl_depth),
                "request_delay": float(request_delay),
                "max_pages": int(max_pages),
                "include_patterns": [p.strip() for p in include_patterns.split(",") if p.strip()],
                "skip_keywords": [p.strip() for p in skip_keywords.split(",") if p.strip()],
            }
        )
        save_config(Path(config_path), new_config)
        status_placeholder = st.empty()
        progress = st.progress(0, text="Starting scraper...")
        # We cannot stream progress from the subprocess easily; show indeterminate style by bouncing.
        for i in range(0, 100, 10):
            progress.progress(i, text="Running scraper...")
        ok, logs = run_scrape(target_url, Path(config_path))
        progress.progress(100, text="Scrape finished")
        if ok:
            status_placeholder.success("Scrape completed")
        else:
            status_placeholder.error("Scrape failed")
        if logs:
            st.text_area("Scrape logs", logs, height=200)
        quotes_rows = load_jsonl(Path(quotes_path))
        authors_rows = load_jsonl(Path(authors_path))
        quotes_df = ensure_dataframe(quotes_rows)
        authors_df = ensure_dataframe(authors_rows)

    st.sidebar.subheader("Filters")
    collection_filter = st.sidebar.text_input("Collection name", value="")
    tag_filter = st.sidebar.text_input("Tag contains", value="")

    df = quotes_df.copy()
    if collection_filter:
        df = df[df.get("collection_name", "") == collection_filter]
    if tag_filter:
        df = df[df["tags"].apply(lambda t: any(tag_filter.lower() in str(tag).lower() for tag in (t or [])))]

    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quotes", f"{len(df):,}")
    col2.metric("Authors", f"{len(authors_df):,}")
    col3.metric("Avg quality", f"{df['quality_score'].mean():.2f}" if "quality_score" in df else "n/a")
    col4.metric("Avg word count", f"{df['word_count'].mean():.1f}" if "word_count" in df else "n/a")

    st.subheader("Top Tags")
    tag_counts = Counter([tag for tags in df["tags"].dropna() for tag in tags])
    st.bar_chart(pd.DataFrame(tag_counts.most_common(15), columns=["tag", "count"]).set_index("tag"))

    st.subheader("Top Authors")
    st.bar_chart(df["author"].value_counts().head(15))

    if "topic_label" in df:
        st.subheader("Topics")
        st.bar_chart(df["topic_label"].value_counts())

    if "emotion_label" in df:
        st.subheader("Emotions")
        st.bar_chart(df["emotion_label"].value_counts())

    if "quote_type" in df:
        st.subheader("Quote Type")
        st.bar_chart(df["quote_type"].value_counts())

    st.subheader("Quality vs. Length")
    if "quality_score" in df and "word_count" in df:
        st.scatter_chart(df[["word_count", "quality_score"]])

    st.subheader("Raw Data")
    tab1, tab2 = st.tabs(["Quotes", "Authors"])
    with tab1:
        st.dataframe(df.head(200))
    with tab2:
        st.dataframe(authors_df.head(200))


if __name__ == "__main__":
    main()
