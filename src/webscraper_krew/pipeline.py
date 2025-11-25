from typing import List

from .models import QuoteContext
from .utils import estimate_read_time_seconds, slugify


class QuoteMetadataPipeline:
    """Composable metadata pipeline for quotes."""

    def __init__(self, processors: List) -> None:
        self.processors = processors

    def register(self, fn) -> None:
        self.processors.append(fn)

    def build(self, ctx: QuoteContext) -> dict:
        payload: dict = {}
        for fn in self.processors:
            payload.update(fn(ctx))
        return payload

    @classmethod
    def default(cls) -> "QuoteMetadataPipeline":
        """Default pipeline with all enrichment steps."""
        return cls(
            [
                cls._base_fields,
                cls._text_fields,
                cls._author_fields,
                cls._tag_fields,
                cls._structural_fields,
                cls._embedding_fields,
                cls._quality_fields,
                cls._rag_fields,
            ]
        )

    @staticmethod
    def _base_fields(ctx: QuoteContext) -> dict:
        r = ctx.record
        return {
            "id": f"quote-{ctx.dedupe_key}" if ctx.dedupe_key else f"quote-{r.author}",
            "type": "quote",
            "source_url": r.source_url,
            "start_url": ctx.start_url,
            "source_site": ctx.source_site,
            "quote": r.quote,
            "author_name_raw": r.author,
            "author_normalized_id": slugify(r.author),
            "depth": r.depth,
            "scraped_at": ctx.scraped_at,
            "fetched_at": ctx.scraped_at,
            "crawl_timestamp": ctx.scraped_at,
            "page_number": r.page_number,
            "position_on_page": r.position_on_page,
            "document_id": ctx.document_id,
            "chunk_id": f"{ctx.document_id}::{r.position_on_page}",
            "chunk_index": r.position_on_page,
            "collection_name": ctx.collection_name,
            "last_updated_at": ctx.scraped_at,
            "dedupe_key": ctx.dedupe_key,
            "is_duplicate": ctx.is_duplicate,
            "source_confidence": ctx.source_confidence,
            "parse_status": ctx.parse_status,
        }

    @staticmethod
    def _text_fields(ctx: QuoteContext) -> dict:
        tf = ctx.text_features
        return {
            "word_count": tf["word_count"],
            "char_count": tf["char_count"],
            "token_count": tf["token_count"],
            "language": tf["language"],
            "language_confidence": tf["language_confidence"],
            "estimated_read_time_seconds": estimate_read_time_seconds(tf["word_count"]),
            "avg_word_length": tf["avg_word_length"],
            "has_punctuation_issues": tf["has_punctuation_issues"],
            "has_ellipsis": tf["has_ellipsis"],
            "has_question_mark": tf["has_question_mark"],
            "has_exclamation_mark": tf["has_exclamation_mark"],
            "normalized_text": tf["normalized_text"],
            "quote_normalized": tf["quote_normalized"],
        }

    @staticmethod
    def _author_fields(ctx: QuoteContext) -> dict:
        meta = ctx.author_meta
        return {
            "author_country": meta.get("country"),
            "author_birth_year": meta.get("birth_year"),
            "author_death_year": meta.get("death_year"),
            "author_era": meta.get("era"),
        }

    @staticmethod
    def _tag_fields(ctx: QuoteContext) -> dict:
        return {
            "tags": ctx.record.tags,
            "tags_normalized": ctx.tags_normalized,
            "tag_count": ctx.tags_count,
            "tags_count": ctx.tags_count,
            "primary_tag": ctx.primary_tag,
            "secondary_tags": ctx.secondary_tags,
        }

    @staticmethod
    def _structural_fields(ctx: QuoteContext) -> dict:
        st = ctx.structural
        return {
            "quote_type": st["quote_type"],
            "perspective": st["perspective"],
            "tense": st["tense"],
            "contains_named_entities": st["contains_named_entities"],
            "named_entities": st["named_entities"],
            "topic_label": ctx.topic_label,
            "emotion_label": ctx.emotion_label,
        }

    @staticmethod
    def _embedding_fields(ctx: QuoteContext) -> dict:
        return {
            "embedding_vector": ctx.embedding,
            "embedding_model": "hash-mini-16d",
            "top_keywords": ctx.top_keywords,
        }

    @staticmethod
    def _quality_fields(ctx: QuoteContext) -> dict:
        return {
            "quality_score": ctx.quality_score,
            "safety_flags": ctx.safety_flags,
        }

    @staticmethod
    def _rag_fields(ctx: QuoteContext) -> dict:
        return {
            "content_type": "quote",
            "document_id": ctx.document_id,
            "chunk_id": f"{ctx.document_id}::{ctx.record.position_on_page}",
            "chunk_index": ctx.record.position_on_page,
            "collection_name": ctx.collection_name,
            "last_updated_at": ctx.scraped_at,
        }
