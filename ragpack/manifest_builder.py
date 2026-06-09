"""
ManifestBuilder — constructs the canonical RAGPack manifest dict.

Design contract (EPIC3 Stage 3)
--------------------------------
- All required fields are explicit constructor arguments; nothing is inferred
  silently or read from global state.
- creation_time is accepted as a caller-supplied string so that the manifest
  is fully reproducible in tests without patching datetime.
- The dict produced by build() is the single source of truth fed to
  manifest.json.  It carries all fields required by manifest_v1_1.json.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


# ---------------------------------------------------------------------------
# Required manifest field names (kept as module-level constants so callers
# can assert completeness without hard-coding string literals).
# ---------------------------------------------------------------------------

REQUIRED_MANIFEST_FIELDS = frozenset({
    "ragpack_version",
    "creation_time",
    "chunk_count",
    "embedding_model",
    "embedding_dimension",
    "model_hash",
    "chunking_config_hash",
    "dtype",
})


# ---------------------------------------------------------------------------
# RAGpack v1.2 — nested, app-facing manifest (ADR-0011 §5)
# ---------------------------------------------------------------------------
#
# The flat ManifestBuilder below is the internal EPIC3 (parquet) format.  The
# NoesisNoema app consumes a *nested* manifest (pack_version + chunker/embedder/
# indexer/files blocks) validated against schemas/manifest_v1_2.json.  That
# nested manifest is produced here (and by writer.pack_writer.PackWriter, which
# delegates to this builder) so there is a single source of truth for its shape.

#: Default file map for a v1.2 pack written by PackWriter.
DEFAULT_V1_2_FILES = {
    "chunks": "chunks.json",
    "embeddings": "embeddings.npy",
    "citations": "citations.jsonl",
}

#: Fields the v1.2 schema requires inside the embedder block.
REQUIRED_V1_2_EMBEDDER_FIELDS = frozenset({
    "embedding_model",
    "embedding_version",
    "embedding_dimension",
    "model_hash",
    "dtype",
    "pooling",
    "l2_normalized",
})


def build_manifest_v1_2(
    *,
    pack_id: str,
    created_at: str,
    chunker: dict[str, Any],
    embedder: dict[str, Any],
    indexer: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    source_documents: list | None = None,
) -> dict[str, Any]:
    """
    Build the canonical, nested RAGpack **v1.2** manifest dict.

    The result conforms to ``schemas/manifest_v1_2.json`` and is the manifest
    the NoesisNoema app validates on import.  The central identity fields live
    in the ``embedder`` block: ``model_hash`` (GGUF file-bytes SHA-256),
    ``pooling`` (``"mean"``), ``l2_normalized`` (``true``), and ``dtype``
    (``"float32"``) — see ADR-0011 §3/§5.

    Args:
        pack_id:          Stable unique identifier for the pack.  Caller-supplied
                          so the manifest is reproducible (no uuid4 here).
        created_at:       ISO-8601 timestamp string.
        chunker:          Chunker metadata block (e.g.
                          ``TokenChunker.get_chunker_metadata()``).
        embedder:         Embedder metadata block, typically
                          ``EmbedderMetadata.to_dict()``.  MUST contain the v1.2
                          required embedder fields.
        indexer:          Optional indexer block (document/chunk counts, ts).
        files:            Optional file map; defaults to DEFAULT_V1_2_FILES.
        source_documents: Optional list of source-document dicts.

    Returns:
        Nested manifest dict with ``pack_version == "1.2"``.

    Raises:
        ValueError: if pack_id/created_at are empty, or the embedder block is
                    missing a v1.2-required field, or pooling/l2_normalized/dtype
                    carry non-v1.2 values.
    """
    if not pack_id:
        raise ValueError("pack_id must not be empty")
    if not created_at:
        raise ValueError("created_at must not be empty")
    if not isinstance(chunker, dict) or not chunker:
        raise ValueError("chunker metadata block must be a non-empty dict")
    if not isinstance(embedder, dict) or not embedder:
        raise ValueError("embedder metadata block must be a non-empty dict")

    missing = REQUIRED_V1_2_EMBEDDER_FIELDS - set(embedder)
    if missing:
        raise ValueError(
            f"embedder block missing v1.2-required field(s): {sorted(missing)}"
        )
    if embedder["dtype"] != "float32":
        raise ValueError(
            f"v1.2 requires embedder.dtype == 'float32', got {embedder['dtype']!r}"
        )
    if embedder["pooling"] != "mean":
        raise ValueError(
            f"v1.2 requires embedder.pooling == 'mean', got {embedder['pooling']!r}"
        )
    if embedder["l2_normalized"] is not True:
        raise ValueError(
            "v1.2 requires embedder.l2_normalized == true, got "
            f"{embedder['l2_normalized']!r}"
        )

    manifest: dict[str, Any] = {
        "pack_version": "1.2",
        "pack_id": pack_id,
        "created_at": created_at,
        "chunker": dict(chunker),
        "embedder": dict(embedder),
        "indexer": dict(indexer) if indexer else {},
        "files": dict(files) if files else dict(DEFAULT_V1_2_FILES),
        "source_documents": list(source_documents) if source_documents else [],
    }
    return manifest


def _manifest_hash(manifest_body: dict) -> str:
    """
    Return a SHA-256 hex digest of the manifest body (excluding the hash
    field itself) serialised as canonically sorted JSON.

    This hash lets consumers verify that a manifest has not been tampered
    with after generation.
    """
    canonical = json.dumps(manifest_body, sort_keys=True, ensure_ascii=True,
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ManifestBuilder:
    """
    Constructs the canonical manifest.json content for a RAGPack.

    All fields that affect reproducibility are required at construction time.
    Optional fields (source_documents, extra_metadata) may be supplied but
    the manifest is valid without them.

    Usage
    -----
    ::

        builder = ManifestBuilder(
            chunk_count=42,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=384,
            model_hash="abc...",
            chunking_config_hash="def...",
            dtype="float32",
            creation_time="2026-03-06T00:00:00",
            embedding_version="3.4.1",
        )
        manifest = builder.build()
    """

    #: Bump this when the manifest schema changes in a breaking way.
    RAGPACK_VERSION: str = "1.2"

    def __init__(
        self,
        *,
        chunk_count: int,
        embedding_model: str,
        embedding_dimension: int,
        model_hash: str,
        chunking_config_hash: str,
        dtype: str,
        creation_time: str,
        embedding_version: str = "",
        source_documents: list | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            chunk_count:          Total number of chunks in the pack.
            embedding_model:      HuggingFace model identifier.
            embedding_dimension:  Number of dimensions per vector.
            model_hash:           SHA-256 of the model configuration dict.
            chunking_config_hash: SHA-256 of the chunking configuration dict.
            dtype:                NumPy dtype string (e.g. "float32").
            creation_time:        ISO-8601 timestamp string (caller-supplied
                                  so the manifest is reproducible in tests).
            embedding_version:    Library version string (optional).
            source_documents:     Optional list of source document dicts.
            extra_metadata:       Optional dict of additional top-level fields.
        """
        if chunk_count < 0:
            raise ValueError("chunk_count must be >= 0")
        if not embedding_model:
            raise ValueError("embedding_model must not be empty")
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be >= 1")
        if not model_hash:
            raise ValueError("model_hash must not be empty")
        if not chunking_config_hash:
            raise ValueError("chunking_config_hash must not be empty")
        if not dtype:
            raise ValueError("dtype must not be empty")
        if not creation_time:
            raise ValueError("creation_time must not be empty")

        self._chunk_count = chunk_count
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._model_hash = model_hash
        self._chunking_config_hash = chunking_config_hash
        self._dtype = dtype
        self._creation_time = creation_time
        self._embedding_version = embedding_version
        self._source_documents = source_documents or []
        self._extra_metadata = extra_metadata or {}

    def build(self) -> dict[str, Any]:
        """
        Return the complete manifest as a plain dict.

        The dict is deterministic: given the same constructor arguments it
        always produces the same JSON-serialisable output.  A
        ``manifest_hash`` field is appended last so it can be used by
        consumers to verify integrity.
        """
        body: dict[str, Any] = {
            "ragpack_version": self.RAGPACK_VERSION,
            "creation_time": self._creation_time,
            "chunk_count": self._chunk_count,
            "embedding_model": self._embedding_model,
            "embedding_version": self._embedding_version,
            "embedding_dimension": self._embedding_dimension,
            "model_hash": self._model_hash,
            "chunking_config_hash": self._chunking_config_hash,
            "dtype": self._dtype,
            "files": {
                "embeddings": "embeddings.npy",
                "chunks": "chunks.parquet",
                "manifest": "manifest.json",
            },
            "source_documents": self._source_documents,
        }

        # Merge caller-supplied extra fields (they must not override required keys)
        for key, value in self._extra_metadata.items():
            if key not in REQUIRED_MANIFEST_FIELDS and key != "files":
                body[key] = value

        # Append integrity hash computed over the body built so far
        body["manifest_hash"] = _manifest_hash(body)

        return body

