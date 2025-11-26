import hashlib
import math
import re
from typing import Dict, List


def slugify(text: str) -> str:
    """Create a simple slug/id.

    Args:
        text: Input string to normalize.
    Returns:
        Lowercased string containing only alphanumerics and dashes.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return cleaned


def detect_safety_flags(normalized_text: str) -> List[str]:
    """Flag basic safety concerns.

    Args:
        normalized_text: Lowercased text to scan.
    Returns:
        List of safety flags such as profanity or sensitive topics.
    """
    PROFANITY = {"damn", "hell"}
    SENSITIVE = {"religion", "god", "politics", "suicide", "self-harm"}
    flags: List[str] = []
    lower = normalized_text.lower()
    if any(word in lower.split() for word in PROFANITY):
        flags.append("contains_profanity")
    if any(term in lower for term in SENSITIVE):
        flags.append("contains_sensitive_topic")
    return flags


def compute_quality_score(text_features: Dict[str, object]) -> float:
    """Heuristic quality score between 0 and 1.

    Args:
        text_features: Dict containing word_count, char_count, has_punctuation_issues.
    Returns:
        A bounded float quality score in [0,1].
    """
    word_count = text_features["word_count"]
    char_count = text_features["char_count"]
    has_punct = text_features["has_punctuation_issues"]
    score = 1.0
    if word_count < 5 or word_count > 60:
        score -= 0.2
    if char_count < 20 or char_count > 300:
        score -= 0.1
    if has_punct:
        score -= 0.2
    if 8 <= word_count <= 40:
        score += 0.1
    return max(0.0, min(1.0, round(score, 3)))


def build_dedupe_key(normalized_text: str) -> str:
    """Stable dedupe key from normalized text.

    Args:
        normalized_text: Lowercased, normalized text string.
    Returns:
        SHA1 hex digest to use for deduplication.
    """
    return hashlib.sha1(normalized_text.encode("utf-8")).hexdigest() if normalized_text else ""


def estimate_read_time_seconds(word_count: int, wpm: int = 200) -> int:
    """Estimate read time in seconds.

    Args:
        word_count: Number of words.
        wpm: Words per minute reading speed (default 200).
    Returns:
        Estimated read time in seconds.
    """
    if word_count <= 0:
        return 0
    return int(math.ceil((word_count / wpm) * 60))


def make_document_id(source_url: str) -> str:
    """Stable document ID from source URL.

    Args:
        source_url: URL string.
    Returns:
        SHA1 hex digest for document grouping.
    """
    return hashlib.sha1(source_url.encode("utf-8")).hexdigest()


def _ordinal_suffix(n: int) -> str:
    """Return ordinal suffix for a positive integer.

    Args:
        n: Positive integer.
    Returns:
        Suffix such as st/nd/rd/th.
    """
    if 10 <= n % 100 <= 20:
        return "th"
    last = n % 10
    if last == 1:
        return "st"
    if last == 2:
        return "nd"
    if last == 3:
        return "rd"
    return "th"
