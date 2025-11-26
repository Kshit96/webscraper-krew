import hashlib
import re
from typing import Dict, List, Optional

try:
    from transformers import BasicTokenizer, pipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BasicTokenizer = None  # type: ignore
    pipeline = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

import numpy as np

from .utils import slugify

# Stopwords for basic keyword filtering
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "to",
    "for",
    "with",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "it",
    "this",
    "that",
    "as",
    "but",
    "if",
    "into",
    "about",
    "than",
    "then",
    "so",
    "very",
    "too",
    "can",
    "will",
    "would",
    "could",
    "should",
}

# Shared HF tokenizer to get a more realistic token count than simple whitespace split
_basic_tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=True) if BasicTokenizer else None
_ner_pipeline: Optional[object] = None
_ner_pipeline_tried = False
_sentence_model: Optional[object] = None
_sentence_model_tried = False
_pca_matrix: Optional[np.ndarray] = None


def compute_text_features(text: str) -> Dict[str, object]:
    """Compute lightweight text features for AI enrichment (counts, flags, language guess)."""
    normalized_text = " ".join(text.split())
    quote_normalized = normalized_text.replace("“", '"').replace("”", '"').replace("’", "'")
    if _basic_tokenizer:
        tokens = _basic_tokenizer.tokenize(quote_normalized)
    else:
        tokens = re.findall(r"\w+", quote_normalized.lower())
    word_count = len(tokens)
    char_count = len(normalized_text)
    token_count = len(tokens)
    avg_word_length = (sum(len(w) for w in tokens) / word_count) if word_count else 0
    ascii_ratio = sum(1 for c in normalized_text if c.isascii()) / max(len(normalized_text), 1)
    detected_lang = "en" if ascii_ratio > 0.8 else "unknown"
    detected_lang_confidence = round(ascii_ratio, 3)
    has_punctuation_issues = ascii_ratio < 0.7 or any(ord(c) < 32 for c in normalized_text if c not in "\n\t\r")
    has_ellipsis = "..." in normalized_text or "… " in normalized_text
    has_question_mark = "?" in normalized_text
    has_exclamation_mark = "!" in normalized_text

    return {
        "normalized_text": normalized_text.lower(),
        "quote_normalized": quote_normalized,
        "word_count": word_count,
        "char_count": char_count,
        "token_count": token_count,
        "avg_word_length": avg_word_length,
        "detected_lang": detected_lang,
        "detected_lang_confidence": detected_lang_confidence,
        "has_punctuation_issues": has_punctuation_issues,
        "has_ellipsis": has_ellipsis,
        "has_question_mark": has_question_mark,
        "has_exclamation_mark": has_exclamation_mark,
    }


def build_mini_embedding(text: str, dims: int = 16) -> List[float]:
    """
    Embedding with optional MiniLM + PCA; falls back to deterministic hash.

    - If sentence-transformers is installed and MiniLM can load, generate a 384D embedding and
      project to 10D via a fixed, deterministic projection matrix.
    - Otherwise, return a 16D hash-based embedding (or `dims` length if provided).
    """
    global _sentence_model, _sentence_model_tried, _pca_matrix

    def _hash_embedding(n: int) -> List[float]:
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        while len(digest) < n * 2:
            digest += hashlib.sha1(digest).digest()
        vec: List[float] = []
        for i in range(n):
            segment = digest[i * 2 : i * 2 + 2]
            val = int.from_bytes(segment, "big", signed=False) / 65535.0
            vec.append(round(val, 6))
        return vec

    # Try to load MiniLM once
    if not _sentence_model_tried and SentenceTransformer is not None:
        try:
            _sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        except Exception:
            _sentence_model = None
        _sentence_model_tried = True
        if _sentence_model is not None:
            rng = np.random.default_rng(42)
            mat = rng.standard_normal((384, 10))
            # Orthonormalize columns for a stable projection
            q, _ = np.linalg.qr(mat)
            _pca_matrix = q[:, :10]

    if _sentence_model is None or _pca_matrix is None:
        return _hash_embedding(dims)

    try:
        vec_384 = _sentence_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        projected = vec_384 @ _pca_matrix
        return [float(round(v, 6)) for v in projected]
    except Exception:
        return _hash_embedding(dims)


def normalize_tags(tags: List[str]) -> List[str]:
    """Lowercase, slugify, and dedupe tags preserving order."""
    seen = set()
    normalized: List[str] = []
    for tag in tags:
        slug = slugify(tag)
        if slug and slug not in seen:
            seen.add(slug)
            normalized.append(slug)
    return normalized


def extract_top_keywords(text: str, tags: List[str], limit: int = 5) -> List[str]:
    """Simple keyword extraction: combine tags + top non-stopword tokens by length/frequency."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())
    freq: Dict[str, int] = {}
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        freq[tok] = freq.get(tok, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    keywords = [t for t, _ in sorted_tokens[: limit - len(tags)]]
    combined = []
    seen = set()
    for candidate in tags + keywords:
        slug = slugify(candidate)
        if slug and slug not in seen:
            seen.add(slug)
            combined.append(slug)
        if len(combined) >= limit:
            break
    return combined


def infer_topic_label(text: str, tags: List[str]) -> str:
    """Heuristic topic labeling based on keywords."""
    haystack = (text + " " + " ".join(tags)).lower()
    topic_keywords = {
        "love": ["love", "heart"],
        "life": ["life", "living"],
        "success": ["success", "achievement"],
        "philosophy": ["truth", "mind", "thought"],
        "humor": ["funny", "humor", "laugh"],
        "friendship": ["friend", "friends", "friendship"],
        "inspiration": ["inspire", "inspiration", "dream", "hope"],
    }
    for topic, keys in topic_keywords.items():
        if any(key in haystack for key in keys):
            return topic
    return "general"


def infer_emotion_label(text: str) -> str:
    """Very light sentiment-like guess using keyword cues."""
    lower = text.lower()
    positive = ["love", "hope", "joy", "happy", "beauty", "dream"]
    negative = ["fear", "sad", "hate", "pain", "anger"]
    if any(word in lower for word in positive):
        return "positive"
    if any(word in lower for word in negative):
        return "negative"
    return "neutral"


def infer_structural_features(text: str, tags: List[str]) -> Dict[str, object]:
    """Infer simple structural/role labels from text."""
    lower = text.lower()
    quote_type = infer_quote_type(text, tags)
    perspective = infer_perspective(lower)
    tense = infer_tense(lower)
    named_entities = extract_named_entities(text)
    contains_named_entities = bool(named_entities)
    return {
        "quote_type": quote_type,
        "perspective": perspective,
        "tense": tense,
        "contains_named_entities": contains_named_entities,
        "named_entities": named_entities,
    }


def infer_quote_type(text: str, tags: List[str]) -> str:
    """Heuristic quote type."""
    lower = text.lower()
    haystack = lower + " " + " ".join(tags).lower()
    mapping = {
        "humorous": ["funny", "humor", "laugh", "ridiculous"],
        "inspirational": ["inspire", "dream", "hope", "believe"],
        "philosophical": ["truth", "thought", "mind", "reality"],
        "motivational": ["success", "achieve", "do it", "keep", "never give up"],
    }
    for label, cues in mapping.items():
        if any(cue in haystack for cue in cues):
            return label
    if "love" in haystack:
        return "inspirational"
    return "general"


def infer_perspective(lower_text: str) -> str:
    """Heuristic person perspective."""
    first = {" i ", " my ", " mine ", " we ", " our ", " us "}
    second = {" you ", " your ", " yours "}
    third = {" he ", " she ", " they ", " him ", " her ", " them ", " his ", " hers ", " theirs "}
    padded = f" {lower_text} "
    if any(tok in padded for tok in first):
        return "first_person"
    if any(tok in padded for tok in second):
        return "second_person"
    if any(tok in padded for tok in third):
        return "third_person"
    return "unspecified"


def infer_tense(lower_text: str) -> str:
    """Rough tense guess."""
    if re.search(r"\bwill\b|\bgoing to\b", lower_text):
        return "future"
    if re.search(r"\b(ed|was|were)\b", lower_text):
        return "past"
    return "present"


def extract_named_entities(text: str) -> List[Dict[str, str]]:
    """Named entity extraction with a small NER model; falls back to heuristic on failure."""
    global _ner_pipeline, _ner_pipeline_tried

    def _heuristic_entities() -> List[Dict[str, str]]:
        candidates = re.findall(r"\b[A-Z][a-zA-Z']+\b", text)
        stop = {"the", "a", "an", "and", "but", "if", "or", "nor", "yet", "so"}
        entities: List[Dict[str, str]] = []
        seen = set()
        for cand in candidates:
            key = cand.lower()
            if key in stop:
                continue
            if key not in seen:
                seen.add(key)
                entities.append({"type": "UNKNOWN", "text": cand})
        return entities

    if not _ner_pipeline_tried:
        try:
            _ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        except Exception:
            _ner_pipeline = None
        _ner_pipeline_tried = True

    if _ner_pipeline is None:
        return _heuristic_entities()

    try:
        spans = _ner_pipeline(text)
        entities = []
        seen = set()
        for span in spans:
            label = span.get("entity_group") or span.get("entity") or "ENT"
            word = span.get("word") or span.get("text") or ""
            key = (label, word.lower())
            if key in seen:
                continue
            seen.add(key)
            entities.append({"type": label, "text": word})
        return entities or _heuristic_entities()
    except Exception:
        return _heuristic_entities()


def determine_parse_status(text_features: Dict[str, object]) -> str:
    """Classify parse outcome."""
    if text_features["word_count"] == 0:
        return "failed"
    if text_features["has_punctuation_issues"]:
        return "partial"
    return "ok"
