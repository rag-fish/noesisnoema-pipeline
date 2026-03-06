"""
Token-based text chunking implementation.

This module provides the TokenChunker class, which splits text into chunks
based on token count with configurable overlap.

EPIC3 Stage 1 update
--------------------
TokenChunker now exposes ``chunk_document``, which returns a list of
``ChunkRecord`` objects with fully deterministic ``chunk_id`` values.

The existing ``chunk_text`` and ``chunk_text_with_offsets`` methods are
preserved for backward compatibility with existing code and tests.
"""

from typing import Dict, Any, List, Optional
import warnings

from .chunk_record import (
    ChunkRecord,
    build_chunk_record,
    compute_chunking_config_hash,
)

# Try to import transformers, fall back to basic splitting if not available
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn(
        "transformers not available, falling back to character-based estimation. "
        "Install transformers for accurate token counting: pip install transformers"
    )


class TokenChunker:
    """
    A chunker that splits text into fixed-size chunks based on token count.
    
    Supports configurable chunk size and overlap between chunks. Uses transformers
    tokenizer for accurate token counting when available, falls back to character
    estimation otherwise.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        tokenizer_name: str = "gpt2",
        preserve_sentences: bool = True
    ):
        """
        Initialize the TokenChunker.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            overlap: Number of tokens to overlap between chunks
            tokenizer_name: Name of the tokenizer to use for token counting
            preserve_sentences: Whether to try to break on sentence boundaries
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
            
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer_name = tokenizer_name
        self.preserve_sentences = preserve_sentences
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                warnings.warn(f"Could not load tokenizer {tokenizer_name}: {e}")
        
        # Character to token ratio estimation (rough approximation)
        self._char_to_token_ratio = 4.0  # Approximate chars per token

        # Pre-compute the config hash once so all chunks produced by this
        # instance share the same stable identifier.
        self._config_hash: str = compute_chunking_config_hash(
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            tokenizer_name=self.tokenizer_name,
            preserve_sentences=self.preserve_sentences,
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer or estimation."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: estimate tokens from character count
            return max(1, int(len(text) / self._char_to_token_ratio))
    
    def _find_sentence_boundary(self, text: str, start_pos: int, max_pos: int) -> int:
        """Find the nearest sentence boundary before max_pos."""
        if not self.preserve_sentences or start_pos >= max_pos:
            return max_pos
            
        # Look for sentence endings working backwards from max_pos
        sentence_endings = '.!?'
        for i in range(max_pos - 1, start_pos, -1):
            if text[i] in sentence_endings:
                # Check if this looks like a real sentence ending
                # (not an abbreviation or decimal)
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
        
        # If no sentence boundary found, return max_pos
        return max_pos
    
    def _estimate_char_position_for_tokens(self, text: str, target_tokens: int) -> int:
        """Estimate character position for a target token count."""
        if self.tokenizer is not None:
            # Binary search to find the right character position
            left, right = 0, len(text)
            while left < right:
                mid = (left + right + 1) // 2
                tokens = self._count_tokens(text[:mid])
                if tokens <= target_tokens:
                    left = mid
                else:
                    right = mid - 1
            return left
        else:
            # Use character estimation
            return min(len(text), int(target_tokens * self._char_to_token_ratio))
    
    def chunk_text_with_offsets(self, text: str, doc_id: str = "unknown") -> List[dict]:
        """
        Split text into chunks with detailed offset information.
        
        Args:
            text: Input text to chunk
            doc_id: Document identifier for tracking
            
        Returns:
            List of dictionaries with chunk data and offset information
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # If text is shorter than chunk size, return as single chunk
        if self._count_tokens(text) <= self.chunk_size:
            return [{
                'chunk_id': 0,
                'doc_id': doc_id,
                'text': text,
                'start_char': 0,
                'end_char': len(text),
                'token_count': self._count_tokens(text),
                'char_count': len(text),
                'paragraph_boundaries': self._find_paragraph_boundaries(text),
                'snippet': text[:200] + '...' if len(text) > 200 else text
            }]
        
        chunks = []
        start_pos = 0
        chunk_id = 0
        
        while start_pos < len(text):
            # Find end position for this chunk
            end_pos = self._estimate_char_position_for_tokens(
                text[start_pos:], 
                self.chunk_size
            ) + start_pos
            
            # Adjust for sentence boundaries if enabled
            if self.preserve_sentences and end_pos < len(text):
                end_pos = self._find_sentence_boundary(text, start_pos, end_pos)
            
            # Ensure we don't go past the end
            end_pos = min(end_pos, len(text))
            
            # Extract chunk
            chunk_text = text[start_pos:end_pos].strip()
            if chunk_text:  # Only add non-empty chunks
                # Find actual start/end positions of the stripped chunk in original text
                actual_start = text.find(chunk_text, start_pos)
                actual_end = actual_start + len(chunk_text)
                
                chunk_data = {
                    'chunk_id': chunk_id,
                    'doc_id': doc_id,
                    'text': chunk_text,
                    'start_char': actual_start,
                    'end_char': actual_end,
                    'token_count': self._count_tokens(chunk_text),
                    'char_count': len(chunk_text),
                    'paragraph_boundaries': self._find_paragraph_boundaries(chunk_text),
                    'snippet': chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text
                }
                chunks.append(chunk_data)
                chunk_id += 1
            
            # If we've reached the end, break
            if end_pos >= len(text):
                break
            
            # Calculate next start position with overlap
            if self.overlap > 0:
                overlap_pos = self._estimate_char_position_for_tokens(
                    text[start_pos:end_pos],
                    self.overlap
                )
                start_pos = max(start_pos + 1, end_pos - overlap_pos)
            else:
                start_pos = end_pos
        
        return chunks

    def _find_paragraph_boundaries(self, text: str) -> List[int]:
        """Find paragraph boundaries in text (positions of double newlines)."""
        boundaries = []
        pos = 0
        while True:
            pos = text.find('\n\n', pos)
            if pos == -1:
                break
            boundaries.append(pos)
            pos += 2
        return boundaries
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Use the new method and extract just the text
        chunks_with_offsets = self.chunk_text_with_offsets(text)
        return [chunk['text'] for chunk in chunks_with_offsets]
    
    def get_chunk_info(self, chunks: List[str]) -> List[dict]:
        """
        Get metadata information for each chunk.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of dictionaries with chunk metadata
        """
        info = []
        for i, chunk in enumerate(chunks):
            info.append({
                'chunk_id': i,
                'token_count': self._count_tokens(chunk),
                'char_count': len(chunk),
                'chunk_text_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
            })
        return info
    
    def get_chunker_metadata(self) -> Dict[str, Any]:
        """
        Return chunker configuration metadata for the manifest.

        Returns:
            Dictionary with chunker metadata including a stable config hash
            that can be stored in the RAGPack manifest for reproducibility.
        """
        return {
            "method": "token_based",
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "tokenizer_name": self.tokenizer_name,
            "preserve_sentences": self.preserve_sentences,
            "config_hash": self._config_hash,
        }

    def chunk_document(
        self,
        text: str,
        source_id: str,
        source_path: str,
        source_hash: str,
        section: Optional[str] = None,
        page: Optional[int] = None,
    ) -> List[ChunkRecord]:
        """
        Split text into ChunkRecord objects with deterministic chunk IDs.

        This is the primary method for EPIC3 production use.  Every returned
        ChunkRecord has a fully traceable, deterministic ``chunk_id`` derived
        from the source identity, character offsets, chunk text, and chunking
        configuration.

        Args:
            text:         Source text to split.
            source_id:    Stable identifier for the originating source document.
            source_path:  Original file path or URI.
            source_hash:  SHA-256 hex digest of the raw source file content.
            section:      Optional nearest section/heading for every chunk.
            page:         Optional page number for every chunk.

        Returns:
            Ordered list of ChunkRecord objects, empty for blank input.
        """
        raw_chunks = self.chunk_text_with_offsets(text, doc_id=source_id)
        records: List[ChunkRecord] = []

        for index, raw in enumerate(raw_chunks):
            record = build_chunk_record(
                chunk_text=raw["text"],
                chunk_index=index,
                char_start=raw["start_char"],
                char_end=raw["end_char"],
                token_count=raw["token_count"],
                source_id=source_id,
                source_path=source_path,
                source_hash=source_hash,
                chunking_config_hash=self._config_hash,
                section=section,
                page=page,
            )
            records.append(record)

        return records
