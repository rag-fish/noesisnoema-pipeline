"""
PDF → plain-text extraction for the RAGpack pipeline.

Uses **pymupdf** (``fitz``) ``get_text("text")``, the measured winner of a
bake-off on the actual Spinoza *Ethics* source PDF (see PR body / docs/audit).
pymupdf preserves inter-word whitespace where ``pypdf``'s default mode (the
previous notebook implementation) glued words together, and ``pypdf``'s
``extraction_mode="layout"`` over-padded with alignment whitespace.

    extractor                whitespace_ratio   dictionary_hit_rate   seconds
    pypdf default (broken)        0.001                0.036             3.6
    pypdf layout                  0.364                0.872             4.2   (rejected)
    pdfplumber                    0.132                0.796            14.8
    pymupdf (fitz)                0.144                0.887             0.9   (winner)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .text_quality import ExtractionQualityError, assert_text_quality


def _open(pdf_path: str):
    """Import pymupdf under either module name and open the document."""
    try:
        import pymupdf  # modern import name
        return pymupdf.open(pdf_path)
    except ImportError:
        pass
    try:
        import fitz  # legacy import name
        return fitz.open(pdf_path)
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "pymupdf is required for PDF extraction. Install it with: "
            "pip install pymupdf"
        ) from exc


def extract_pdf_text(pdf_path: Union[str, Path]) -> str:
    """
    Extract the full text of a PDF, preserving word boundaries.

    Pages are joined with a blank line (``"\\n\\n"``) so paragraph/page breaks
    survive into chunking.  No quality gate is applied here; call
    :func:`convert_pdf_to_txt` (or ``text_quality.assert_text_quality``) to
    refuse garbled output.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        The extracted text.

    Raises:
        ImportError: if pymupdf is not installed.
    """
    doc = _open(str(pdf_path))
    try:
        return "\n\n".join(doc[i].get_text("text") for i in range(doc.page_count))
    finally:
        doc.close()


def convert_pdf_to_txt(
    pdf_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    *,
    validate: bool = True,
) -> Path:
    """
    Extract ``pdf_path`` to a sibling ``.txt`` file, failing loud on garbled text.

    Args:
        pdf_path: Source PDF path.
        out_path: Destination ``.txt`` path; defaults to ``pdf_path`` with a
                  ``.txt`` suffix.
        validate: If True (default), run the extraction-quality gate and raise
                  ``ExtractionQualityError`` rather than write token-soup.

    Returns:
        Path to the written ``.txt`` file.

    Raises:
        ImportError:             if pymupdf is not installed.
        ExtractionQualityError:  if ``validate`` and the text fails the gate.
    """
    pdf_path = Path(pdf_path)
    text = extract_pdf_text(pdf_path)
    if validate:
        assert_text_quality(text, source=pdf_path.name)
    out_path = Path(out_path) if out_path else pdf_path.with_suffix(".txt")
    out_path.write_text(text, encoding="utf-8")
    return out_path


__all__ = ["extract_pdf_text", "convert_pdf_to_txt", "ExtractionQualityError"]
