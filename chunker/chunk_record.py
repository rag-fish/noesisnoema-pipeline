"""
Canonical chunk record schema and deterministic chunk_id generation.

This module defines ChunkRecord — the single source of truth for chunk
identity, lineage, and metadata in the RAGPack generation pipeline.

Design contract (EPIC3 Stage 1):
- chunk_id is always derived deterministically from content and config.
- No UUID4 or wall-clock timestamps are used as primary identifiers.
- Every ChunkRecord is fully traceable back to its source document.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Optional


_CHUNK_ID_VERSION = "1"
_SNIPPET_MAX_CHARS = 200


def _sha256_hex(*parts: str) -> str:
    """Return a hex SHA-256 digest of all parts joined by a null byte."""
    payload = "\x00".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_chunk_text_hash(chunk_text: str) -> str:
    """
    Return a stable SHA-256 hex digest of the chunk text.

    The text is UTF-8 encoded before hashing so the hash is
    unambiguous and portable across Python environments.
    """
    return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()


def compute_chunking_config_hash(
    chunk_size: int,
    overlap: int,
    tokenizer_name: str,
    preserve_sentences: bool,
) -> str:
    """
    Return a stable SHA-256 hex digest of the chunking configuration.

    The config is serialised as a canonically-sorted JSON string so
    that field ordering never changes the resulting hash.
    """
    config = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "preserve_sentences": preserve_sentences,
        "tokenizer_name": tokenizer_name,
        "version": _CHUNK_ID_VERSION,
    }
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_chunk_id(
    source_id: str,
    char_start: int,
    char_end: int,
    chunk_text_hash: str,
    chunking_config_hash: str,
) -> str:
    """
    Return a deterministic, stable chunk identifier.

    Rule (EPIC3 Stage 1):
        chunk_id = sha256(
            source_id +
            char_start +
            char_end +
            chunk_text_hash +
            chunking_config_hash
        )

    All inputs must be non-empty / non-negative; passing empty strings
    or negative positions is an explicit caller error.
    """
    if not source_id:
        raise ValueError("source_id must not be empty")
    if char_start < 0:
        raise ValueError("char_start must be >= 0")
    if char_end <= char_start:
        raise ValueError("char_end must be > char_start")
    if not chunk_text_hash:
        raise ValueError("chunk_text_hash must not be empty")
    if not chunking_config_hash:
        raise ValueError("chunking_config_hash must not be empty")

    return _sha256_hex(
        source_id,
        str(char_start),
        str(char_end),
        chunk_text_hash,
        chunking_config_hash,
    )


@dataclass
class ChunkRecord:
    """
    Canonical record for a single text chunk produced by the pipeline.

    All fields are required unless explicitly marked Optional.
    There are no mutable defaults; callers must supply every field.

    Identity / lineage fields
    -------------------------
    chunk_id            Deterministic SHA-256 derived from source + offsets +
                        chunk content + chunking config.  Never a UUID.
    source_id           Stable identifier for the originating source document
                        (e.g. SHA-256 of canonical file path + file content hash).
    source_path         Original file path or URI of the source document.
    source_hash         SHA-256 hex digest of the raw source file content.
    chunk_index         Zero-based ordinal of this chunk within the source.

    Position fields
    ---------------
    char_start          Start character offset in the stripped source text.
    char_end            End character offset (exclusive) in the stripped source text.
    token_count         Number of tokens in this chunk (model-specific; recorded for
                        information and reproducibility checks, not as a stable ID).

    Structure fields (optional, populated when source supports them)
    ----------------------------------------------------------------
    section             Heading or section title nearest to this chunk, if extractable.
    page                Page number(s) this chunk spans, if the source is paginated.

    Content fields
    --------------
    text_snippet        First 200 characters of the chunk text, for human inspection.
    chunk_text_hash     SHA-256 hex digest of the chunk text (UTF-8 encoded).

    Config lineage
    --------------
    chunking_config_hash  SHA-256 hex digest of the chunking configuration that
                          produced this chunk.  Allows later reproducibility checks.
    """

    chunk_id: str
    source_id: str
    source_path: str
    source_hash: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    text_snippet: str
    chunk_text_hash: str
    chunking_config_hash: str

    # Optional — populated when source type supports these
    section: Optional[str] = None
    page: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate field invariants after construction."""
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.source_path:
            raise ValueError("source_path must not be empty")
        if not self.source_hash:
            raise ValueError("source_hash must not be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        if self.char_start < 0:
            raise ValueError("char_start must be >= 0")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be > char_start")
        if self.token_count < 1:
            raise ValueError("token_count must be >= 1")
        if not self.chunk_text_hash:
            raise ValueError("chunk_text_hash must not be empty")
        if not self.chunking_config_hash:
            raise ValueError("chunking_config_hash must not be empty")
        if len(self.text_snippet) > _SNIPPET_MAX_CHARS + 3:
            # +3 for the trailing "..." that _build_snippet appends
            raise ValueError(
                f"text_snippet must not exceed {_SNIPPET_MAX_CHARS + 3} characters"
            )

    def to_dict(self) -> dict:
        """Return a plain dict representation suitable for JSON serialisation."""
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "section": self.section,
            "page": self.page,
            "text_snippet": self.text_snippet,
            "chunk_text_hash": self.chunk_text_hash,
            "chunking_config_hash": self.chunking_config_hash,
        }


def build_snippet(text: str) -> str:
    """
    Return a human-readable snippet of at most 200 characters.

    Appends '...' when the text is truncated so the consumer knows the
    snippet is not the full chunk text.
    """
    if len(text) <= _SNIPPET_MAX_CHARS:
        return text
    return text[:_SNIPPET_MAX_CHARS] + "..."


def build_chunk_record(
    *,
    chunk_text: str,
    chunk_index: int,
    char_start: int,
    char_end: int,
    token_count: int,
    source_id: str,
    source_path: str,
    source_hash: str,
    chunking_config_hash: str,
    section: Optional[str] = None,
    page: Optional[int] = None,
) -> ChunkRecord:
    """
    Construct a fully-populated ChunkRecord.

    This factory is the single authorised path for creating ChunkRecords.
    It derives chunk_text_hash and chunk_id deterministically from the
    supplied inputs so callers cannot accidentally supply stale values.

    Args:
        chunk_text:           Raw text of the chunk.
        chunk_index:          Zero-based ordinal within the source.
        char_start:           Start character offset in the stripped source.
        char_end:             End character offset (exclusive) in the stripped source.
        token_count:          Token count for this chunk.
        source_id:            Stable source document identifier.
        source_path:          Original file path or URI.
        source_hash:          SHA-256 hex of the raw source file content.
        chunking_config_hash: SHA-256 hex of the chunking configuration.
        section:              Optional nearest section/heading.
        page:                 Optional page number.

    Returns:
        A validated ChunkRecord.
    """
    chunk_text_hash = compute_chunk_text_hash(chunk_text)
    chunk_id = compute_chunk_id(
        source_id=source_id,
        char_start=char_start,
        char_end=char_end,
        chunk_text_hash=chunk_text_hash,
        chunking_config_hash=chunking_config_hash,
    )
    return ChunkRecord(
        chunk_id=chunk_id,
        source_id=source_id,
        source_path=source_path,
        source_hash=source_hash,
        chunk_index=chunk_index,
        char_start=char_start,
        char_end=char_end,
        token_count=token_count,
        text_snippet=build_snippet(chunk_text),
        chunk_text_hash=chunk_text_hash,
        chunking_config_hash=chunking_config_hash,
        section=section,
        page=page,
    )

