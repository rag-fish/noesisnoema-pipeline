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

