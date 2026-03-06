"""
RagpackBuilder — orchestrates chunk embedding and Ragpack assembly.

Design contract (EPIC3 Stage 3)
--------------------------------
- Single explicit entry point: RagpackBuilder.build().
- No background threads, no hidden side effects, no autonomous retries.
- Chunk order in the output is identical to the input order.
- All non-determinism in the pipeline is eliminated before this layer;
  this module only assembles what the embedder already produced.
- creation_time is an explicit parameter (ISO-8601 string) so callers
  control reproducibility in tests without patching datetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from chunker.chunk_record import ChunkRecord
from embedder.deterministic_embedder import DeterministicEmbedder, EmbeddingResult
from .manifest_builder import ManifestBuilder


# ---------------------------------------------------------------------------
# Ragpack — the output value object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ragpack:
    """
    Immutable output of a single RagpackBuilder.build() call.

    Fields
    ------
    chunks        Ordered list of ChunkRecord objects, one per row of embeddings.
    embeddings    np.ndarray of shape (N, D), dtype float32.  Row i matches
                  chunks[i].
    manifest      Plain dict ready for JSON serialisation as manifest.json.
    chunk_ids     Ordered list of chunk_id strings, aligned with chunks and
                  embedding rows, for fast lookup without iterating ChunkRecords.
    """

    chunks: List[ChunkRecord]
    embeddings: np.ndarray
    manifest: dict
    chunk_ids: List[str]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D, got shape {self.embeddings.shape}"
            )
        if self.embeddings.shape[0] != len(self.chunks):
            raise ValueError(
                f"embeddings row count ({self.embeddings.shape[0]}) must equal "
                f"len(chunks) ({len(self.chunks)})"
            )
        if len(self.chunk_ids) != len(self.chunks):
            raise ValueError(
                f"len(chunk_ids) ({len(self.chunk_ids)}) must equal "
                f"len(chunks) ({len(self.chunks)})"
            )
        # Verify chunk_ids alignment
        for idx, (cid, chunk) in enumerate(zip(self.chunk_ids, self.chunks)):
            if cid != chunk.chunk_id:
                raise ValueError(
                    f"chunk_ids[{idx}] ('{cid}') does not match "
                    f"chunks[{idx}].chunk_id ('{chunk.chunk_id}')"
                )


# ---------------------------------------------------------------------------
# RagpackBuilder
# ---------------------------------------------------------------------------

class RagpackBuilder:
    """
    Orchestrates embedding of ChunkRecord objects and assembles a Ragpack.

    The builder is stateless between calls to build(); construct a new
    instance or reuse one — both are safe.

    Usage
    -----
    ::

        embedder = DeterministicEmbedder()
        builder  = RagpackBuilder(embedder)
        ragpack  = builder.build(
            chunks=chunk_records,
            creation_time="2026-03-06T00:00:00",
        )
    """

    def __init__(self, embedder: DeterministicEmbedder) -> None:
        """
        Args:
            embedder: A fully initialised DeterministicEmbedder.  The same
                      embedder instance may be reused across multiple build()
                      calls; it carries no mutable per-build state.
        """
        if not isinstance(embedder, DeterministicEmbedder):
            raise TypeError(
                f"embedder must be a DeterministicEmbedder, "
                f"got {type(embedder).__name__}"
            )
        self._embedder = embedder

    def build(
        self,
        chunks: Sequence[ChunkRecord],
        creation_time: str,
        source_documents: list | None = None,
    ) -> Ragpack:
        """
        Embed all chunks and assemble the complete Ragpack.

        Args:
            chunks:           Ordered sequence of ChunkRecord objects.
            creation_time:    ISO-8601 timestamp string, supplied by the caller
                              so the output is reproducible in tests without
                              patching datetime.
            source_documents: Optional list of source document metadata dicts
                              to embed in the manifest.

        Returns:
            A fully populated, immutable Ragpack.

        Raises:
            TypeError:  if any element of chunks is not a ChunkRecord.
            ValueError: if creation_time is empty.
        """
        if not creation_time:
            raise ValueError("creation_time must not be empty")

        # Validate input types explicitly — fail fast, no silent coercion.
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, ChunkRecord):
                raise TypeError(
                    f"chunks[{idx}] must be a ChunkRecord, "
                    f"got {type(chunk).__name__}"
                )

        chunk_list = list(chunks)

        # --- Embed in input order; EmbeddingResult.chunk_ids is aligned ---
        embedding_result: EmbeddingResult = self._embedder.embed_chunks(chunk_list)

        # Derive chunking_config_hash from the first chunk (all chunks from
        # the same chunker share the same config hash; the manifest needs one
        # representative value).  Use the sentinel "empty" when there are no
        # chunks so the manifest field remains non-empty and self-documenting.
        chunking_config_hash = (
            chunk_list[0].chunking_config_hash if chunk_list else "empty"
        )

        meta = self._embedder.metadata
        manifest_builder = ManifestBuilder(
            chunk_count=len(chunk_list),
            embedding_model=meta.embedding_model,
            embedding_dimension=meta.embedding_dimension,
            model_hash=meta.model_hash,
            chunking_config_hash=chunking_config_hash,
            dtype=meta.dtype,
            creation_time=creation_time,
            embedding_version=meta.embedding_version,
            source_documents=source_documents,
        )
        manifest = manifest_builder.build()

        return Ragpack(
            chunks=chunk_list,
            embeddings=embedding_result.embeddings,
            manifest=manifest,
            chunk_ids=embedding_result.chunk_ids,
        )

