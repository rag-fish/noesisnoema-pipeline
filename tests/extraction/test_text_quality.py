"""Tests for the extraction-quality fail-loud gate."""
import pytest

from extraction.text_quality import (
    ExtractionQualityError,
    assert_text_quality,
    assess,
    dictionary_hit_rate,
    whitespace_ratio,
)

# A healthy paragraph of real English prose (well above all thresholds).
HEALTHY = (
    "Spinoza is one of those great men whose eminence grows the more we study "
    "his work. The nature of substance and of God is the foundation of his "
    "system, and from it he derives the whole of his ethics by geometric "
    "demonstration. Human freedom, for Spinoza, consists in understanding the "
    "necessity of nature rather than in escaping it. " * 6
)

# The actual failure mode: pypdf default glued every word together.
SOUP = (
    "Spinozaisoneofthosegreatmenwhoseeminencegrowsthemorewestudyhiswork"
    "ThenatureofsubstanceandofGodisthefoundationofhissystemandfromithederives"
    "thewholeofhisethicsbygeometricdemonstration" * 4
)


def test_whitespace_ratio_separates_healthy_from_soup():
    assert whitespace_ratio(HEALTHY) > 0.10
    assert whitespace_ratio(SOUP) < 0.02


def test_dictionary_hit_rate_separates_healthy_from_soup():
    assert dictionary_hit_rate(HEALTHY) > 0.7
    assert dictionary_hit_rate(SOUP) < 0.3


def test_assert_passes_healthy_text():
    metrics = assert_text_quality(HEALTHY, source="healthy.txt")
    assert metrics["whitespace_ratio"] > 0.10
    assert metrics["dictionary_hit_rate"] > 0.7


def test_assert_rejects_token_soup():
    with pytest.raises(ExtractionQualityError) as exc:
        assert_text_quality(SOUP, source="broken.pdf")
    # whitespace is the first/primary gate to trip on glued text
    assert "whitespace_ratio" in str(exc.value)
    assert "broken.pdf" in str(exc.value)


def test_assert_rejects_spaced_gibberish_on_dictionary_rate():
    # Has spaces (passes whitespace gate) but tokens are not real words.
    gibberish = " ".join(["xqzptl", "wkrfmn", "bzzlqx", "vnrtps"] * 100)
    with pytest.raises(ExtractionQualityError) as exc:
        assert_text_quality(gibberish, source="ocr.pdf")
    assert "dictionary_hit_rate" in str(exc.value)


def test_short_snippet_not_falsely_rejected():
    # Below the size/token floors: judged leniently, must not raise.
    assert_text_quality("Hi there.", source="tiny.md")


def test_assess_reports_glued_run_signal():
    m = assess(SOUP)
    assert m["max_token_length"] > 50  # glued runs show up as huge tokens
