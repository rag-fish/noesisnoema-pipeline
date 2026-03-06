"""
noesisnoema-pipeline chunker module.

Provides deterministic text chunking for RAGPack generation.
All public types and helpers are exported here so callers import
from a single stable surface.
"""

from .chunk_record import (
    ChunkRecord,
    build_chunk_record,
    build_snippet,
    compute_chunk_id,
    compute_chunk_text_hash,
    compute_chunking_config_hash,
)
from .token_chunker import TokenChunker

__all__ = [
    "ChunkRecord",
    "build_chunk_record",
    "build_snippet",
    "compute_chunk_id",
    "compute_chunk_text_hash",
    "compute_chunking_config_hash",
    "TokenChunker",
]
