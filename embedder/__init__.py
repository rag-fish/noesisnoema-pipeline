"""
noesisnoema-pipeline embedder module.

Provides deterministic, reproducible embedding generation for RAGPack
artifact production (EPIC3 Stage 2).

All public types and helpers are exported from this single surface.
"""

from .deterministic_embedder import DeterministicEmbedder, EmbedderMetadata

__all__ = [
    "DeterministicEmbedder",
    "EmbedderMetadata",
]

