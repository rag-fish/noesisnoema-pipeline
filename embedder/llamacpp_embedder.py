"""
llama.cpp-based embedding generation for RAGpack v1.2 production.

Design contract (ADR-0011 §5 — RAGpack v1.2)
--------------------------------------------
- Model identity is the SHA-256 of the GGUF **file bytes**, computed at load
  time.  This is the fingerprint the NoesisNoema app validates on import
  (ADR-0011 §3); a pack is only importable if its ``embedder.model_hash``
  equals the hash of the GGUF the app ships.
- Embeddings are mean-pooled (one vector per chunk), explicitly L2-normalized
  to unit length, and always emitted as ``float32``.
- The model is loaded once with mean pooling enabled and never mutated after
  construction.  Given the same input + same GGUF + same flags, llama.cpp is
  deterministic in embedding mode, so the same texts always produce the same
  vectors.
- Output order matches input order exactly — no internal reordering.

Public API (mirrors DeterministicEmbedder so callers need minimal changes)
--------------------------------------------------------------------------
    embedder = LlamaCppEmbedder("/path/to/nomic-embed-text-v1.5.Q5_K_M.gguf")
    result   = embedder.embed_chunks(chunk_records)   # -> EmbeddingResult
    vecs     = embedder.embed_texts(["hello world"])  # -> np.ndarray (N, 768)
    # result.embeddings : np.ndarray, shape (N, 768), dtype float32, L2-unit rows
    # result.metadata   : EmbedderMetadata (model_hash = GGUF file hash)

Task prefix
-----------
``nomic-embed-text-v1.5`` is a task-conditioned model: document/chunk texts
MUST be prefixed with ``"search_document: "`` (queries would use
``"search_query: "``, but this pipeline only embeds documents).  The prefix is
applied INSIDE ``embed_chunks`` / ``embed_texts`` — callers pass raw text and
must NOT add it themselves.  Stripping this prefix silently degrades retrieval
quality, so do not remove it.
"""

from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np

from chunker.chunk_record import ChunkRecord
from .deterministic_embedder import (
    EMBEDDING_DTYPE,
    EmbedderMetadata,
    EmbeddingResult,
    _sha256_file,
)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Runtime identifier recorded in the v1.2 manifest ``embedder.runtime`` field.
RUNTIME: str = "llama.cpp"

#: Pooling strategy.  v1.2 only supports mean pooling (see manifest_v1_2.json).
POOLING: str = "mean"

#: Task prefix required by nomic-embed-text-v1.5 for document/chunk texts.
DOCUMENT_TASK_PREFIX: str = "search_document: "

#: Expected output dimension for nomic-embed-text-v1.5.
NOMIC_EMBED_DIMENSION: int = 768

#: Context / batch sizing used at load time.  nomic-embed-text-v1.5 supports an
#: 8192-token context; matching n_batch / n_ubatch lets a full chunk be pooled
#: in a single ubatch so mean pooling sees the whole sequence.
_N_CTX: int = 8192
_N_BATCH: int = 8192
_N_UBATCH: int = 8192


class LlamaCppEmbedder:
    """
    Loads a GGUF embedding model via llama-cpp-python and embeds chunks in a
    deterministic, reproducible manner suitable for RAGpack v1.2.

    Usage
    -----
    ::

        embedder = LlamaCppEmbedder("nomic-embed-text-v1.5.Q5_K_M.gguf")
        result = embedder.embed_chunks(chunk_records)

        embeddings = result.embeddings   # np.ndarray (N, 768), float32, unit rows
        metadata   = result.metadata     # EmbedderMetadata (GGUF file hash)

    Determinism guarantees
    ----------------------
    - The GGUF is loaded once with ``embedding=True`` and mean pooling.
    - ``seed=0`` is pinned (embedding mode does not sample, but this removes
      any residual RNG dependence).
    - Output rows are explicitly L2-normalized (not relying on any internal
      llama.cpp normalization) so they byte-match the app's own normalization.
    - Output is always cast to ``float32``.
    """

    def __init__(
        self,
        gguf_path: str,
        *,
        n_ctx: int = _N_CTX,
        n_batch: int = _N_BATCH,
        n_ubatch: int = _N_UBATCH,
        seed: int = 0,
    ) -> None:
        """
        Load the GGUF embedding model and capture all metadata.

        Args:
            gguf_path: Path to the embedder GGUF file (e.g.
                       ``nomic-embed-text-v1.5.Q5_K_M.gguf``).
            n_ctx:     Context window passed to llama.cpp.
            n_batch:   Logical batch size.
            n_ubatch:  Physical (micro) batch size; sized to fit a full chunk.
            seed:      RNG seed, pinned to 0 for reproducibility.

        Raises:
            ValueError:    if gguf_path is empty or does not point to a file.
            ImportError:   if llama-cpp-python is not installed.
            RuntimeError:  if the model fails to load.
        """
        if not gguf_path or not str(gguf_path).strip():
            raise ValueError("gguf_path must not be empty")
        if not os.path.isfile(gguf_path):
            raise ValueError(f"gguf_path '{gguf_path}' is not a file")

        try:
            from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "llama-cpp-python is required for LlamaCppEmbedder. "
                "Install it with: pip install llama-cpp-python"
            ) from exc

        # Identity FIRST: the GGUF file-bytes hash is the v1.2 model identity
        # (ADR-0011 §3) and is computed independently of a successful load.
        model_hash = _sha256_file(gguf_path)

        try:
            # Mean pooling is requested via pooling_type.  If a future binding
            # drops pooling_type from the constructor, create_embedding() still
            # returns mean-pooled vectors for a model whose GGUF metadata
            # declares mean pooling (nomic-embed-text-v1.5 does); the manual
            # mean-pool fallback in _to_matrix() covers token-level output.
            self._model = Llama(
                model_path=gguf_path,
                embedding=True,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                pooling_type=LLAMA_POOLING_TYPE_MEAN,
                seed=seed,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load GGUF embedding model '{gguf_path}': {exc}"
            ) from exc

        self._gguf_path = gguf_path

        # Resolve the embedding dimension from the loaded model when possible,
        # falling back to a probe embedding.
        dimension = self._resolve_dimension()

        try:
            from llama_cpp import __version__ as _llama_version
        except Exception:  # pragma: no cover
            _llama_version = "unknown"

        self._metadata = EmbedderMetadata(
            embedding_model=os.path.basename(gguf_path),
            embedding_version=str(_llama_version),
            embedding_dimension=dimension,
            model_hash=model_hash,
            dtype=EMBEDDING_DTYPE,
            pooling=POOLING,
            l2_normalized=True,
            runtime=RUNTIME,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> EmbedderMetadata:
        """EmbedderMetadata captured at construction time (GGUF file hash)."""
        return self._metadata

    def embed_chunks(self, chunks: Sequence[ChunkRecord]) -> EmbeddingResult:
        """
        Embed a sequence of ChunkRecord objects and return aligned results.

        The ``"search_document: "`` task prefix is applied internally to each
        chunk's text before embedding (see module docstring).  Output row ``i``
        corresponds exactly to ``chunks[i]``; no reordering is performed.

        Args:
            chunks: Ordered sequence of ChunkRecord objects to embed.  May be
                    empty, in which case a zero-row array is returned.

        Returns:
            EmbeddingResult with float32, L2-normalized ``embeddings``,
            ``metadata``, and ``chunk_ids`` aligned to the input sequence.

        Raises:
            TypeError:  if any element of ``chunks`` is not a ChunkRecord.
            ValueError: if any produced vector has zero L2 norm.
        """
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, ChunkRecord):
                raise TypeError(
                    f"chunks[{idx}] must be a ChunkRecord, "
                    f"got {type(chunk).__name__}"
                )

        if not chunks:
            empty = np.empty((0, self._metadata.embedding_dimension), dtype=np.float32)
            return EmbeddingResult(embeddings=empty, metadata=self._metadata, chunk_ids=[])

        texts = [chunk.text_snippet for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        embeddings = self.embed_texts(texts)

        return EmbeddingResult(
            embeddings=embeddings,
            metadata=self._metadata,
            chunk_ids=chunk_ids,
        )

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a plain list of strings and return a float32 NumPy array.

        Applies the ``"search_document: "`` task prefix internally, mean-pools
        (handled by llama.cpp), and explicitly L2-normalizes every output row
        to unit length.

        Texts are embedded **one at a time** rather than as a single batched
        ``create_embedding`` call.  llama.cpp's batched mean pooling is
        position-dependent: an identical text yields slightly different vectors
        (~6e-3) depending on its slot in the batch, while a single-input call is
        stable.  Per-text embedding makes each chunk's vector independent of its
        neighbours and of pack ordering, which is required for reproducible
        packs and for parity with the app's single-input embedding path.

        Args:
            texts: Ordered sequence of raw chunk/document strings.  Do NOT
                   pre-apply the task prefix; it is added here.

        Returns:
            np.ndarray of shape ``(N, D)`` and dtype ``float32`` with unit-norm
            rows.

        Raises:
            ValueError: if any produced vector has zero L2 norm (mirrors the
                        app's ``EmbeddingError.zeroNorm``; ADR-0000 §4).
        """
        if not texts:
            return np.empty((0, self._metadata.embedding_dimension), dtype=np.float32)

        rows: List[np.ndarray] = []
        for text in texts:
            response = self._model.create_embedding([DOCUMENT_TASK_PREFIX + str(text)])
            rows.append(self._to_matrix(response, expected_count=1))
        matrix = np.vstack(rows).astype(np.float32, copy=False)
        return self._l2_normalize(matrix)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_dimension(self) -> int:
        """
        Return the embedding dimension, preferring the model's own n_embd().

        Falls back to a single probe embedding if the binding does not expose
        n_embd().  Raises if neither path yields a positive dimension.
        """
        n_embd = getattr(self._model, "n_embd", None)
        if callable(n_embd):
            try:
                dim = int(n_embd())
                if dim > 0:
                    return dim
            except Exception:  # pragma: no cover - defensive
                pass

        probe = self._model.create_embedding([DOCUMENT_TASK_PREFIX + "probe"])
        matrix = self._to_matrix(probe, expected_count=1)
        return int(matrix.shape[1])

    @staticmethod
    def _to_matrix(response: dict, expected_count: int) -> np.ndarray:
        """
        Convert a llama-cpp ``create_embedding`` response into an (N, D) float32
        matrix, preserving input order.

        Handles both pooled output (one vector per input) and the token-level
        case (a list of token vectors per input), mean-pooling the latter so
        the result is always one vector per input regardless of how the binding
        reports it.
        """
        data = response["data"]
        # Order by the 'index' field when present so rows match input order.
        data = sorted(data, key=lambda d: d.get("index", 0))
        if len(data) != expected_count:
            raise RuntimeError(
                f"embedding response returned {len(data)} rows, "
                f"expected {expected_count}"
            )

        vectors: List[np.ndarray] = []
        for entry in data:
            emb = np.asarray(entry["embedding"], dtype=np.float32)
            if emb.ndim == 2:
                # Token-level (no internal pooling) — mean-pool to one vector.
                emb = emb.mean(axis=0)
            elif emb.ndim != 1:
                raise RuntimeError(
                    f"unexpected embedding ndim {emb.ndim}; expected 1 or 2"
                )
            vectors.append(emb.astype(np.float32, copy=False))

        return np.vstack(vectors).astype(np.float32, copy=False)

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """
        Return a copy of ``matrix`` with every row scaled to unit L2 norm.

        Raises ValueError on any zero-norm row — a zero vector cannot be
        normalized and signals a degenerate embedding (mirrors the app's
        ``EmbeddingError.zeroNorm``; visible failure per ADR-0000 §4).
        """
        out = np.empty_like(matrix, dtype=np.float32)
        for i in range(matrix.shape[0]):
            norm = float(np.linalg.norm(matrix[i]))
            if not (norm > 0):
                raise ValueError(
                    f"embedding row {i} has zero L2 norm; cannot normalize"
                )
            out[i] = (matrix[i] / norm).astype(np.float32)
        return out
