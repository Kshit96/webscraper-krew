from typing import Dict
from urllib.parse import urlparse

from .features import (
    build_mini_embedding,
    compute_text_features,
    determine_parse_status,
    extract_top_keywords,
    infer_emotion_label,
    infer_structural_features,
    infer_topic_label,
    normalize_tags,
)
from .models import QuoteContext, QuoteRecord
from .utils import build_dedupe_key, compute_quality_score, detect_safety_flags, make_document_id


class QuoteContextFactory:
    """Build QuoteContext objects from raw quotes and auxiliary data."""

    def __init__(
        self,
        start_url: str,
        scraped_at: str,
        collection_name: str,
        author_lookup: Dict[str, dict],
        dedupe_counts: Dict[str, int],
    ) -> None:
        self.start_url = start_url
        self.scraped_at = scraped_at
        self.collection_name = collection_name
        self.author_lookup = author_lookup
        self.dedupe_counts = dedupe_counts

    def build(self, record: QuoteRecord) -> QuoteContext:
        """Assemble a QuoteContext from a raw QuoteRecord.

        Args:
            record: Raw quote with source URL, text, author, tags, depth, page/position, and language hints.
        Returns:
            QuoteContext containing derived text/structural features, author metadata, dedupe info, quality/safety
            signals, embeddings/keywords/topics, and collection/run metadata ready for the pipeline to emit.
        """
        text_features = compute_text_features(record.quote)
        # Language resolution: prefer node-level lang, then document lang, else detected heuristic
        source_html_lang = record.source_html_lang
        source_node_lang = record.source_node_lang
        detected_lang = text_features["detected_lang"]
        detected_conf = text_features["detected_lang_confidence"]
        resolved_lang = source_node_lang or source_html_lang or detected_lang
        resolved_conf = 1.0 if (source_node_lang or source_html_lang) else detected_conf
        text_features["source_html_lang"] = source_html_lang
        text_features["source_node_lang"] = source_node_lang
        text_features["language"] = resolved_lang
        text_features["language_confidence"] = resolved_conf

        tags_normalized = normalize_tags(record.tags)
        source_site = urlparse(record.source_url).netloc
        embedding = build_mini_embedding(text_features["normalized_text"])
        top_keywords = extract_top_keywords(text_features["normalized_text"], record.tags)
        topic_label = infer_topic_label(text_features["normalized_text"], record.tags)
        emotion_label = infer_emotion_label(text_features["normalized_text"])
        structural = infer_structural_features(record.quote, record.tags)
        dedupe_key = build_dedupe_key(text_features["normalized_text"])
        is_duplicate = self.dedupe_counts.get(dedupe_key, 0) > 0
        self.dedupe_counts[dedupe_key] = self.dedupe_counts.get(dedupe_key, 0) + 1
        source_confidence = 0.6 if text_features["has_punctuation_issues"] else 1.0
        parse_status = determine_parse_status(text_features)
        quality_score = compute_quality_score(text_features)
        safety_flags = detect_safety_flags(text_features["normalized_text"])
        document_id = make_document_id(record.source_url)
        author_meta = self.author_lookup.get(record.author, {})
        primary_tag = record.tags[0] if record.tags else None
        secondary_tags = record.tags[1:] if len(record.tags) > 1 else []

        return QuoteContext(
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
            collection_name=self.collection_name,
            scraped_at=self.scraped_at,
            start_url=self.start_url,
            source_site=source_site,
            tags_normalized=tags_normalized,
            tags_count=len(record.tags),
            primary_tag=primary_tag,
            secondary_tags=secondary_tags,
            source_html_lang=source_html_lang,
            source_node_lang=source_node_lang,
            detected_lang=detected_lang,
            detected_lang_confidence=detected_conf,
            resolved_lang=resolved_lang,
            page_title=record.page_title,
        )
