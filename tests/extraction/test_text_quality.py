"""Tests for the extraction-quality fail-loud gate."""
import pytest

from extraction.text_quality import (
    DICT_HIT_FAIL_FLOOR,
    DICT_HIT_WARN_BELOW,
    ExtractionQualityError,
    assert_text_quality,
    assess,
    classify_dictionary_hit_rate,
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


# --- dictionary_hit_rate three-band policy (fail 0.80 / warn 0.90 / ok) ---

def test_classify_thresholds_are_relaxed_to_080_080_090():
    assert DICT_HIT_FAIL_FLOOR == 0.80
    assert DICT_HIT_WARN_BELOW == 0.90


def test_classify_fail_below_floor():
    status, msg = classify_dictionary_hit_rate(0.79, source="x.pdf")
    assert status == "fail"
    assert "0.80" in msg and "x.pdf" in msg


def test_classify_broken_token_soup_still_fails_at_080():
    # The known-broken signature (token-soup) scores ~0.11 and must FAIL the
    # relaxed 0.80 floor with full margin.
    dh = dictionary_hit_rate(SOUP)
    assert dh < 0.30
    status, _ = classify_dictionary_hit_rate(dh)
    assert status == "fail"


def test_classify_warn_band_does_not_fail(caplog):
    # The real Ethics pack lands here (~0.86): acceptable OCR noise, warn not fail.
    import logging
    with caplog.at_level(logging.WARNING):
        status, msg = classify_dictionary_hit_rate(0.861, source="ethics")
    assert status == "warn"
    assert "warn band" in msg
    assert any("warn band" in r.message for r in caplog.records)  # never silent


@pytest.mark.parametrize("value", [0.80, 0.85, 0.899])
def test_classify_warn_band_boundaries(value):
    assert classify_dictionary_hit_rate(value)[0] == "warn"


def test_classify_ok_at_or_above_090():
    assert classify_dictionary_hit_rate(0.90)[0] == "ok"
    assert classify_dictionary_hit_rate(0.97)[0] == "ok"
