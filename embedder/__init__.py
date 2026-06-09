"""
noesisnoema-pipeline embedder module.

Provides deterministic, reproducible embedding generation for RAGPack
artifact production (EPIC3 Stage 2).

All public types and helpers are exported from this single surface.
"""

from .deterministic_embedder import (
    DeterministicEmbedder,
    EmbedderMetadata,
    EmbeddingResult,
)
from .llamacpp_embedder import LlamaCppEmbedder

__all__ = [
    "LlamaCppEmbedder",
    "EmbedderMetadata",
    "EmbeddingResult",
    # Deprecated (RAGpack v1.1 only; not compatible with NoesisNoema v0.4+).
    "DeterministicEmbedder",
]

