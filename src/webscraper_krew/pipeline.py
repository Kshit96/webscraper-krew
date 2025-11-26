from typing import Callable, List

from .models import QuoteContext
from .utils import estimate_read_time_seconds, slugify


class QuoteMetadataPipeline:
    """Extensible metadata pipeline: register field builders that consume QuoteContext and return dicts."""

    def __init__(self, builders: List[Callable[[QuoteContext], dict]] | None = None) -> None:
        self.builders = builders or []

    @classmethod
    def default(cls) -> "QuoteMetadataPipeline":
        """Default pipeline with all enrichment steps."""
        pipeline = cls()
        pipeline.register(pipeline._base_fields)
        pipeline.register(pipeline._text_fields)
        pipeline.register(pipeline._author_fields)
        pipeline.register(pipeline._tag_fields)
        pipeline.register(pipeline._structural_fields)
        pipeline.register(pipeline._embedding_fields)
        pipeline.register(pipeline._quality_fields)
        pipeline.register(pipeline._rag_fields)
        return pipeline

    def register(self, builder: Callable[[QuoteContext], dict]) -> None:
        """Register an additional builder to extend metadata."""
        self.builders.append(builder)

    def build(self, ctx: QuoteContext) -> dict:
        """Run all registered builders over the given context."""
        payload: dict = {}
        for builder in self.builders:
            payload.update(builder(ctx))
        return payload

    # Builders below:
    def _base_fields(self, ctx: QuoteContext) -> dict:
        r = ctx.record
        return {
            "id": f"quote-{ctx.dedupe_key}" if ctx.dedupe_key else f"quote-{slugify(r.author)}",
            "type": "quote",
            "source_url": r.source_url,
            "start_url": ctx.start_url,
            "source_site": ctx.source_site,
            "page_title": r.page_title,
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

    def _text_fields(self, ctx: QuoteContext) -> dict:
        tf = ctx.text_features
        return {
            "word_count": tf["word_count"],
            "char_count": tf["char_count"],
            "token_count": tf["token_count"],
            "source_html_lang": tf.get("source_html_lang"),
            "source_node_lang": tf.get("source_node_lang"),
            "detected_lang": tf.get("detected_lang"),
            "detected_lang_confidence": tf.get("detected_lang_confidence"),
            "resolved_lang": tf.get("language"),
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

    def _author_fields(self, ctx: QuoteContext) -> dict:
        meta = ctx.author_meta
        return {
            "author_country": meta.get("country"),
            "author_birth_year": meta.get("birth_year"),
            "author_death_year": meta.get("death_year"),
            "author_era": meta.get("era"),
        }

    def _tag_fields(self, ctx: QuoteContext) -> dict:
        return {
            "tags": ctx.record.tags,
            "tags_normalized": ctx.tags_normalized,
            "tags_count": ctx.tags_count,
            "primary_tag": ctx.primary_tag,
            "secondary_tags": ctx.secondary_tags,
        }

    def _structural_fields(self, ctx: QuoteContext) -> dict:
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

    def _embedding_fields(self, ctx: QuoteContext) -> dict:
        return {
            "embedding_vector": ctx.embedding,
            "embedding_model": "hash-mini-16d",
            "top_keywords": ctx.top_keywords,
        }

    def _quality_fields(self, ctx: QuoteContext) -> dict:
        return {
            "quality_score": ctx.quality_score,
            "safety_flags": ctx.safety_flags,
        }

    def _rag_fields(self, ctx: QuoteContext) -> dict:
        return {
            "content_type": "quote",
            "document_id": ctx.document_id,
            "chunk_id": f"{ctx.document_id}::{ctx.record.position_on_page}",
            "chunk_index": ctx.record.position_on_page,
            "collection_name": ctx.collection_name,
            "last_updated_at": ctx.scraped_at,
        }
