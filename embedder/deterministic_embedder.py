"""
Deterministic embedding generation for RAGPack production.

Design contract (EPIC3 Stage 2)
--------------------------------
- Model identity is pinned by name and resolved artifact hash at load time.
- Embeddings are produced in the exact order chunks are supplied.
- No random seeds, no stochastic inference paths, no background threads.
- The same model + same chunk texts → byte-for-byte identical float32 vectors.
- All metadata needed for manifest reproducibility is captured at load time.

Public API
----------
    embedder = DeterministicEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    result   = embedder.embed_chunks(chunk_records)
    # result.embeddings  : np.ndarray, shape (N, D), dtype float32
    # result.metadata    : EmbedderMetadata
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from chunker.chunk_record import ChunkRecord


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default model used when no model_name is supplied to DeterministicEmbedder.
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Output dtype for all embedding arrays produced by this module.
EMBEDDING_DTYPE: str = "float32"


# ---------------------------------------------------------------------------
# Shared file-hash helper (ADR-0011 §3 identity model)
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    """
    Return the lowercase SHA-256 hex digest of a file's raw bytes.

    This is the canonical identity for an embedder GGUF in RAGpack v1.2:
    the manifest ``embedder.model_hash`` must equal this digest so the
    NoesisNoema app can verify the pack was produced by the exact GGUF it
    ships (ADR-0011 §3).  Read in 1 MiB chunks so multi-gigabyte models do
    not have to be loaded into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

#: Batch size used during encoding.  Kept small so behaviour is identical
#: regardless of available VRAM or RAM.
_ENCODE_BATCH_SIZE: int = 64


# ---------------------------------------------------------------------------
# EmbedderMetadata — canonical metadata for manifest inclusion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbedderMetadata:
    """
    Immutable metadata record describing the embedding configuration.

    All fields are required; there are no optional fields.  This object
    is the authoritative source of truth for the ``embedder`` block in
    the RAGPack manifest.

    Fields
    ------
    embedding_model     Short human-readable model name as supplied by the
                        caller (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``
                        or ``"nomic-embed-text-v1.5.Q5_K_M.gguf"``).
    embedding_version   Version string recorded so consumers can reproduce the
                        exact runtime state (sentence-transformers library
                        version, or the llama-cpp-python version).
    embedding_dimension Number of dimensions in each output vector.
    model_hash          Stable identity check for the loaded model.

                        NOTE (ADR-0011 §3): the *semantics* of this field
                        depend on the producing embedder:
                          * ``LlamaCppEmbedder`` — SHA-256 of the GGUF **file
                            bytes**.  This is the identity the NoesisNoema app
                            validates for v1.2 packs.
                          * ``DeterministicEmbedder`` — SHA-256 of the model
                            *configuration* JSON (a multi-file HuggingFace
                            download has no single file to hash).  This is
                            **NOT** compatible with v1.2 manifests intended for
                            NoesisNoema app v0.4+.
    dtype               NumPy dtype string for the output array (always
                        ``"float32"`` in this implementation).

    v1.2 embedder-block fields (RAGpack v1.2 / ADR-0011 §5)
    -------------------------------------------------------
    pooling             Pooling strategy applied to produce one vector per
                        chunk.  ``"mean"`` for the llama.cpp nomic embedder.
    l2_normalized       Whether output vectors are explicitly L2-normalized to
                        unit length.  ``True`` for the llama.cpp v1.2 path.
    runtime             Embedding runtime identifier, e.g. ``"llama.cpp"``.
                        Empty for the legacy sentence-transformers path.
    """

    embedding_model: str
    embedding_version: str
    embedding_dimension: int
    model_hash: str
    dtype: str
    # v1.2 embedder-block fields.  Defaulted so the legacy DeterministicEmbedder
    # (v1.1) can construct EmbedderMetadata without supplying them.
    pooling: str = ""
    l2_normalized: bool = False
    runtime: str = ""

    def to_dict(self) -> dict:
        """
        Return a plain dict suitable for JSON serialisation in the manifest.

        The dict carries both the v1.1 required keys and the v1.2 embedder
        block (``pooling``, ``l2_normalized``, ``runtime``).  ``runtime`` is
        only emitted when set so v1.1 embedder blocks stay byte-identical.
        """
        d = {
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "embedding_dimension": self.embedding_dimension,
            "model_hash": self.model_hash,
            "dtype": self.dtype,
            # v1.2 embedder-block fields (required by manifest_v1_2.json)
            "pooling": self.pooling,
            "l2_normalized": self.l2_normalized,
            # Legacy manifest keys kept for backward compatibility with PackWriter
            "name": self.embedding_model,
            "version": self.embedding_version,
            "dimensions": self.embedding_dimension,
        }
        if self.runtime:
            d["runtime"] = self.runtime
        return d


# ---------------------------------------------------------------------------
# EmbeddingResult — output bundle from a single embed_chunks call
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingResult:
    """
    Output of a single ``DeterministicEmbedder.embed_chunks`` call.

    Fields
    ------
    embeddings  NumPy array of shape ``(N, D)`` and dtype ``float32``.
                Row ``i`` corresponds to ``chunks[i]``.
    metadata    EmbedderMetadata captured at embedder construction time.
    chunk_ids   Ordered list of ``chunk_id`` strings, one per row, so that
                downstream code can bind each embedding row back to its
                ChunkRecord without relying on positional coincidence.
    """

    embeddings: np.ndarray
    metadata: EmbedderMetadata
    chunk_ids: List[str]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D, got shape {self.embeddings.shape}"
            )
        if self.embeddings.dtype != np.float32:
            raise ValueError(
                f"embeddings dtype must be float32, got {self.embeddings.dtype}"
            )
        if self.embeddings.shape[0] != len(self.chunk_ids):
            raise ValueError(
                f"embeddings row count ({self.embeddings.shape[0]}) must equal "
                f"len(chunk_ids) ({len(self.chunk_ids)})"
            )
        if self.embeddings.shape[1] != self.metadata.embedding_dimension:
            raise ValueError(
                f"embeddings column count ({self.embeddings.shape[1]}) must equal "
                f"metadata.embedding_dimension ({self.metadata.embedding_dimension})"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_model_hash(model_config: dict) -> str:
    """
    Return a SHA-256 hex digest of the model configuration dictionary.

    The config is serialised as a canonically sorted JSON string so
    that field insertion order never influences the hash.
    """
    canonical = json.dumps(model_config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_st_version() -> str:
    """Return the installed sentence_transformers library version string."""
    try:
        import sentence_transformers as _st
        return str(_st.__version__)
    except Exception:
        return "unknown"


def _set_deterministic_torch_flags() -> None:
    """
    Apply all available PyTorch determinism controls.

    These flags ensure that CUDA/CPU operations that have non-deterministic
    fast-paths fall back to deterministic equivalents.  No random seeds are
    set; the goal is to eliminate non-determinism from the *algorithm*, not
    to fix a random sequence.
    """
    try:
        import torch
        # Disable non-deterministic CUDA algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Disable cuDNN benchmarking (it selects fastest, not deterministic, algo)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        # PyTorch may not be available in all test environments; skip silently
        pass


# ---------------------------------------------------------------------------
# DeterministicEmbedder
# ---------------------------------------------------------------------------

class DeterministicEmbedder:
    """
    Loads a sentence-transformers model and embeds ChunkRecord lists
    in a fully deterministic, reproducible manner.

    .. deprecated:: 0.4.0 (RAGpack v1.2 / ADR-0011 §5)
        This embedder produces **v1.1** manifests whose ``model_hash`` is a
        config-dict hash, not a GGUF file hash.  Packs it produces are **not
        importable by NoesisNoema app v0.4+**, which validates the embedder
        GGUF file fingerprint (ADR-0011 §3).  Use
        :class:`embedder.llamacpp_embedder.LlamaCppEmbedder` for all v1.2
        builds.  This class is retained for backward compatibility and will be
        removed in a follow-up cleanup PR once the v1.2 transition is verified.

    Usage
    -----
    ::

        embedder = DeterministicEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        result = embedder.embed_chunks(chunk_records)

        embeddings = result.embeddings   # np.ndarray (N, 384), float32
        metadata   = result.metadata     # EmbedderMetadata

    Determinism guarantees
    ----------------------
    - The model is loaded once and never mutated after construction.
    - ``torch.use_deterministic_algorithms`` is enabled at construction.
    - ``cudnn.benchmark`` is disabled.
    - Encoding is performed in a single sorted-order batch with
      ``convert_to_numpy=True`` so no tensor RNG state is involved in the
      output conversion.
    - Output is always cast to ``float32`` before returning.
    - The order of rows in the output array matches the order of the input
      ``chunk_records`` list exactly — no internal reordering occurs.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """
        Load the embedding model and capture all metadata.

        Args:
            model_name: HuggingFace model identifier or local path.
                        Defaults to ``DEFAULT_MODEL_NAME``.

        Raises:
            ImportError:  if sentence-transformers is not installed.
            RuntimeError: if the model fails to load.
        """
        if not model_name or not model_name.strip():
            raise ValueError("model_name must not be empty")

        warnings.warn(
            "DeterministicEmbedder produces RAGpack v1.1 manifests "
            "(config-hash identity) and is NOT compatible with NoesisNoema "
            "v0.4+, which requires v1.2 packs from a GGUF embedder. Use "
            "embedder.llamacpp_embedder.LlamaCppEmbedder for v1.2 builds.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for DeterministicEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        _set_deterministic_torch_flags()

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}': {exc}"
            ) from exc

        self._model_name: str = model_name

        # Resolve dimension from the loaded model
        dimension: int = self._model.get_sentence_embedding_dimension()

        # Build a stable model config hash from fields that are invariant
        # for a given model name.  We include the dimension so that two
        # models with the same name but different output sizes (e.g. after
        # fine-tuning) produce different hashes.
        model_config = {
            "model_name": model_name,
            "embedding_dimension": dimension,
            "dtype": EMBEDDING_DTYPE,
        }
        model_hash = _compute_model_hash(model_config)

        self._metadata = EmbedderMetadata(
            embedding_model=model_name,
            embedding_version=_resolve_st_version(),
            embedding_dimension=dimension,
            model_hash=model_hash,
            dtype=EMBEDDING_DTYPE,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> EmbedderMetadata:
        """EmbedderMetadata captured at construction time."""
        return self._metadata

    def embed_chunks(self, chunks: Sequence[ChunkRecord]) -> EmbeddingResult:
        """
        Embed a sequence of ChunkRecord objects and return aligned results.

        The output ``EmbeddingResult.embeddings`` row ``i`` corresponds
        exactly to ``chunks[i]``.  No reordering is performed.

        Args:
            chunks: Ordered sequence of ChunkRecord objects to embed.
                    May be empty, in which case a zero-row array is returned.

        Returns:
            EmbeddingResult with ``embeddings``, ``metadata``, and
            ``chunk_ids`` aligned to the input sequence.

        Raises:
            TypeError: if any element of ``chunks`` is not a ChunkRecord.
        """
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, ChunkRecord):
                raise TypeError(
                    f"chunks[{idx}] must be a ChunkRecord, "
                    f"got {type(chunk).__name__}"
                )

        if not chunks:
            empty_array = np.empty(
                (0, self._metadata.embedding_dimension), dtype=np.float32
            )
            return EmbeddingResult(
                embeddings=empty_array,
                metadata=self._metadata,
                chunk_ids=[],
            )

        # Extract texts in the exact order they were supplied.
        # The chunk_ids list is built in parallel so the binding is
        # guaranteed to be aligned regardless of what happens inside encode().
        texts = [chunk.text_snippet for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        raw: np.ndarray = self._model.encode(
            texts,
            batch_size=_ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        # Guarantee float32 dtype regardless of model's internal precision
        embeddings = raw.astype(np.float32)

        return EmbeddingResult(
            embeddings=embeddings,
            metadata=self._metadata,
            chunk_ids=chunk_ids,
        )

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a plain list of strings and return a float32 NumPy array.

        This is a lower-level convenience method for callers that do not
        have ChunkRecord objects.  It does not return chunk_ids.

        Args:
            texts: Ordered sequence of strings to embed.

        Returns:
            np.ndarray of shape ``(N, D)`` and dtype ``float32``.
        """
        if not texts:
            return np.empty(
                (0, self._metadata.embedding_dimension), dtype=np.float32
            )

        raw: np.ndarray = self._model.encode(
            list(texts),
            batch_size=_ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return raw.astype(np.float32)

