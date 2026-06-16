"""
Text-extraction quality gate.

A PDF/text extractor that silently drops inter-word whitespace produces
"token-soup" (``"Spinozaisoneofthose..."``) that is unreadable in citations
*and* tokenises into garbage subwords, so the resulting embeddings cannot
distinguish queries (every query retrieves the same glued chunk).  PR #21
audited exactly this failure shipping silently.

This module turns that silent failure into a loud one.  ``assert_text_quality``
RAISES ``ExtractionQualityError`` when extracted text looks like token-soup,
so a bad extraction can never be chunked, embedded and shipped again.

Two cheap, portable signals (no heavy NLP dependency):

* **whitespace_ratio** = ``count(" ") / len(text)``.  Healthy English prose is
  ~0.13-0.18; spaceless soup is ~0.00-0.02.  This is the primary, language- and
  corpus-independent gate.
* **dictionary_hit_rate** = fraction of whitespace-split tokens that are real
  English words, scored against a small bundled common-word list
  (``data/words_common_en.txt.gz``).  Glued runs collapse into long non-words,
  so soup scores ~0.03 while clean prose scores ~0.85.

Measured on the Spinoza *Ethics* source PDF (see PR body):

    extractor                whitespace_ratio   dictionary_hit_rate
    pypdf default (broken)        0.001                0.036
    pymupdf (fitz)                0.144                0.863
"""
from __future__ import annotations

import gzip
import logging
import re
from importlib import resources
from typing import FrozenSet, Optional, Tuple

logger = logging.getLogger(__name__)

#: Default thresholds for the fail-loud gate.  Chosen to sit far from both the
#: broken baseline (ws 0.001 / dict 0.04) and healthy prose (ws 0.14 / dict 0.86)
#: so neither false-positives on clean text nor passes token-soup.
DEFAULT_MIN_WHITESPACE_RATIO: float = 0.08
DEFAULT_MIN_DICTIONARY_HIT_RATE: float = 0.6

#: UAT-level quality policy for ``dictionary_hit_rate`` (a *stricter* band than the
#: catastrophic ``DEFAULT_MIN_DICTIONARY_HIT_RATE`` floor the pipeline guard uses).
#: A clean, well-OCR'd scan scores ~0.90+; a noisier-but-valid scan (e.g. the
#: Spinoza *Ethics* at 0.86, with OCR artefacts like "Bfess"/"hj"/"truJy") sits in
#: the warn band; only genuinely garbled text (token-soup ~0.11) falls below the
#: hard floor.  Lowered from 0.85 -> 0.80 so legitimate OCR noise is tolerated
#: while catastrophic breakage is still caught with full margin.
DICT_HIT_FAIL_FLOOR: float = 0.80
DICT_HIT_WARN_BELOW: float = 0.90

#: The dictionary hit-rate is only meaningful with enough tokens to score; below
#: this count we skip it (a tiny snippet is judged on whitespace alone).
_MIN_TOKENS_FOR_DICT_CHECK: int = 50
#: Whitespace ratio is only enforced once the text is long enough to be prose.
_MIN_CHARS_FOR_WS_CHECK: int = 200

_NON_ALPHA = re.compile(r"[^A-Za-z]")
_WORDLIST_CACHE: Optional[FrozenSet[str]] = None


class ExtractionQualityError(RuntimeError):
    """Raised when extracted text fails the quality gate (looks like soup)."""


def _load_wordlist() -> FrozenSet[str]:
    """Load and cache the bundled common-English word list."""
    global _WORDLIST_CACHE
    if _WORDLIST_CACHE is None:
        raw = (resources.files("extraction.data") / "words_common_en.txt.gz").read_bytes()
        words = gzip.decompress(raw).decode("utf-8").split("\n")
        _WORDLIST_CACHE = frozenset(w for w in words if w)
    return _WORDLIST_CACHE


def whitespace_ratio(text: str) -> float:
    """Fraction of characters that are the space character ``" "``."""
    if not text:
        return 0.0
    return text.count(" ") / len(text)


def dictionary_hit_rate(text: str, wordlist: Optional[FrozenSet[str]] = None) -> float:
    """
    Fraction of whitespace-split tokens that are real English words.

    Tokens are lower-cased and stripped of non-alphabetic characters before
    lookup.  Glued runs (``"isoneofthose"``) become single non-words and so
    drive the rate down, which is exactly the signal we want.
    """
    wl = wordlist if wordlist is not None else _load_wordlist()
    tokens = [_NON_ALPHA.sub("", t).lower() for t in text.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in wl)
    return hits / len(tokens)


def classify_dictionary_hit_rate(
    value: float,
    *,
    source: str = "",
    fail_floor: float = DICT_HIT_FAIL_FLOOR,
    warn_below: float = DICT_HIT_WARN_BELOW,
) -> Tuple[str, str]:
    """
    Classify a ``dictionary_hit_rate`` against the three-band quality policy.

    Returns ``(status, message)`` where ``status`` is one of:

    * ``"fail"`` — ``value < fail_floor`` (default 0.80).  The text looks garbled
      (e.g. whitespace-stripped token-soup, which scores ~0.11); callers should
      treat this as a hard failure and raise.
    * ``"warn"`` — ``fail_floor <= value < warn_below`` (default [0.80, 0.90)).
      Acceptable but the source likely carries OCR noise.  **Not** a failure; a
      ``logging.warning`` is emitted so it is never silent.
    * ``"ok"``   — ``value >= warn_below`` (default 0.90).  Clean; no warning.

    This is the UAT/reporting policy and is intentionally separate from the
    pipeline's catastrophic :data:`DEFAULT_MIN_DICTIONARY_HIT_RATE` (0.6) floor
    enforced by :func:`assert_text_quality`.
    """
    where = f" for '{source}'" if source else ""
    if value < fail_floor:
        return "fail", (
            f"dictionary_hit_rate={value:.3f}{where} is below the hard floor "
            f"{fail_floor:.2f}; the text looks garbled (e.g. whitespace-stripped "
            f"token-soup), not merely noisy."
        )
    if value < warn_below:
        message = (
            f"dictionary_hit_rate={value:.3f}{where} is in the warn band "
            f"[{fail_floor:.2f}, {warn_below:.2f}); extraction looks correct but "
            f"the source likely carries OCR noise. Not failing the build."
        )
        logger.warning(message)
        return "warn", message
    return "ok", (
        f"dictionary_hit_rate={value:.3f}{where} is clean (>= {warn_below:.2f})."
    )


def assess(text: str) -> dict:
    """Return the quality metrics for ``text`` without raising (for reporting)."""
    tokens = [_NON_ALPHA.sub("", t).lower() for t in text.split()]
    tokens = [t for t in tokens if t]
    lengths = [len(t) for t in tokens]
    return {
        "chars": len(text),
        "whitespace_ratio": whitespace_ratio(text),
        "dictionary_hit_rate": dictionary_hit_rate(text),
        "n_tokens": len(tokens),
        "mean_token_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "max_token_length": max(lengths) if lengths else 0,
    }


def assert_text_quality(
    text: str,
    *,
    source: str = "",
    min_whitespace_ratio: float = DEFAULT_MIN_WHITESPACE_RATIO,
    min_dictionary_hit_rate: float = DEFAULT_MIN_DICTIONARY_HIT_RATE,
) -> dict:
    """
    Raise ``ExtractionQualityError`` if ``text`` looks like whitespace-stripped
    token-soup; otherwise return the computed metrics.

    Args:
        text:                     Extracted document text to validate.
        source:                   Human-readable source name for the error message.
        min_whitespace_ratio:     Floor for ``whitespace_ratio`` (enforced once
                                  the text is at least 200 chars).
        min_dictionary_hit_rate:  Floor for ``dictionary_hit_rate`` (enforced
                                  once there are at least 50 tokens).

    Returns:
        The metrics dict from :func:`assess`.

    Raises:
        ExtractionQualityError: if either enforced metric is below its floor.
    """
    metrics = assess(text)
    where = f" for '{source}'" if source else ""

    if metrics["chars"] >= _MIN_CHARS_FOR_WS_CHECK:
        ws = metrics["whitespace_ratio"]
        if ws < min_whitespace_ratio:
            raise ExtractionQualityError(
                f"extracted text{where} has whitespace_ratio={ws:.3f} "
                f"(< {min_whitespace_ratio}); inter-word spaces appear to have "
                f"been stripped (token-soup). Refusing to build a pack from it. "
                f"Re-extract with a layout-aware extractor (pymupdf)."
            )

    if metrics["n_tokens"] >= _MIN_TOKENS_FOR_DICT_CHECK:
        dh = metrics["dictionary_hit_rate"]
        if dh < min_dictionary_hit_rate:
            raise ExtractionQualityError(
                f"extracted text{where} has dictionary_hit_rate={dh:.3f} "
                f"(< {min_dictionary_hit_rate}); most tokens are not real words "
                f"(mean_token_length={metrics['mean_token_length']:.1f}, "
                f"max_token_length={metrics['max_token_length']}). Refusing to "
                f"build a pack from likely-garbled text."
            )

    return metrics
