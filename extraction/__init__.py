"""
Text-extraction package for the RAGpack pipeline.

Exposes a layout-aware PDF extractor (pymupdf) and a fail-loud quality gate
that refuses whitespace-stripped "token-soup" before it can be chunked,
embedded and shipped.
"""
from .pdf_extractor import convert_pdf_to_txt, extract_pdf_text
from .text_quality import (
    ExtractionQualityError,
    assert_text_quality,
    assess,
    dictionary_hit_rate,
    whitespace_ratio,
)

__all__ = [
    "extract_pdf_text",
    "convert_pdf_to_txt",
    "ExtractionQualityError",
    "assert_text_quality",
    "assess",
    "whitespace_ratio",
    "dictionary_hit_rate",
]
