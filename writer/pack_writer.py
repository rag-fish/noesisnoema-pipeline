"""
Pack writer implementation for RAGpack v1.1 / v1.2 (app-facing) format.

This module provides the PackWriter class for generating the RAGpack the
NoesisNoema app consumes: a nested ``manifest.json`` (pack_version +
chunker/embedder/indexer/files blocks) plus ``citations.jsonl``, ``chunks.json``
and ``embeddings.npy``.

As of RAGpack v1.2 (ADR-0011 §5) the default output is **v1.2**: the embedder
block carries the GGUF file-hash identity, ``pooling="mean"`` and
``l2_normalized=true``, and citations are keyed to the app's RAGpackReader spec
(``chunk_index`` / ``char_start`` / ``char_end`` / ``page``).  The nested
manifest body is assembled by ``ragpack.manifest_builder.build_manifest_v1_2``
so there is one source of truth for its shape.
"""

import json
import zipfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import uuid
from io import BytesIO

from ragpack.manifest_builder import build_manifest_v1_2


#: Default RAGpack format version emitted by PackWriter.
DEFAULT_PACK_VERSION: str = "1.2"


class PackWriter:
    """
    Writer class for generating the app-facing RAGpack (v1.2 by default).

    Creates RAGpack files with a nested manifest and citations for precise
    preview and citation capabilities.  Pass ``pack_version="1.1"`` to emit the
    legacy v1.1 manifest shape (not consumable by NoesisNoema v0.4+).

    Determinism: supply ``pack_id`` and ``created_at`` explicitly to get a
    reproducible manifest.  When omitted they fall back to a random uuid4 and
    the wall clock respectively (legacy behaviour, non-reproducible).
    """

    def __init__(
        self,
        pack_id: Optional[str] = None,
        created_at: Optional[str] = None,
        pack_version: str = DEFAULT_PACK_VERSION,
    ):
        """
        Initialize pack writer.

        Args:
            pack_id:      Unique identifier for the pack (random uuid4 if None).
            created_at:   ISO-8601 timestamp string (wall clock if None).
                          Supply explicitly for reproducible packs.
            pack_version: Manifest format version, "1.2" (default) or "1.1".
        """
        if pack_version not in ("1.1", "1.2"):
            raise ValueError(
                f"pack_version must be '1.1' or '1.2', got {pack_version!r}"
            )
        self.pack_id = pack_id or str(uuid.uuid4())
        self.created_at = created_at or datetime.now().isoformat()
        self.pack_version = pack_version
    
    def write_pack(self, 
                   chunks_with_metadata: List[Dict[str, Any]],
                   embeddings: np.ndarray,
                   chunker_metadata: Dict[str, Any],
                   embedder_metadata: Dict[str, Any],
                   indexer_metadata: Dict[str, Any],
                   source_documents: List[Dict[str, Any]],
                   output_path: Union[str, Path],
                   compress: bool = True) -> Path:
        """
        Write complete RAGpack v1.1 to file or directory.
        
        Args:
            chunks_with_metadata: List of enriched chunk dictionaries
            embeddings: NumPy array of embeddings
            chunker_metadata: Metadata from chunker
            embedder_metadata: Metadata from embedder  
            indexer_metadata: Metadata from indexer
            source_documents: List of source document metadata
            output_path: Output file path (.zip) or directory
            compress: Whether to create compressed zip file
            
        Returns:
            Path to created pack
        """
        output_path = Path(output_path)
        
        if compress or output_path.suffix == '.zip':
            return self._write_zip_pack(
                chunks_with_metadata, embeddings, chunker_metadata,
                embedder_metadata, indexer_metadata, source_documents,
                output_path
            )
        else:
            return self._write_directory_pack(
                chunks_with_metadata, embeddings, chunker_metadata,
                embedder_metadata, indexer_metadata, source_documents,
                output_path
            )
    
    def _write_zip_pack(self, 
                        chunks_with_metadata: List[Dict[str, Any]],
                        embeddings: np.ndarray,
                        chunker_metadata: Dict[str, Any],
                        embedder_metadata: Dict[str, Any],
                        indexer_metadata: Dict[str, Any],
                        source_documents: List[Dict[str, Any]],
                        zip_path: Path) -> Path:
        """Write RAGpack to zip file."""
        
        # Prepare data
        chunks_json = [chunk['text'] for chunk in chunks_with_metadata]
        citations_data = self._generate_citations(chunks_with_metadata)
        manifest_data = self._generate_manifest(
            chunker_metadata, embedder_metadata, indexer_metadata, source_documents
        )
        
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # Write chunks.json
            chunks_bytes = json.dumps(chunks_json, ensure_ascii=False).encode('utf-8')
            zf.writestr("chunks.json", chunks_bytes)
            
            # Write embeddings.npy
            embeddings_bytes_io = BytesIO()
            np.save(embeddings_bytes_io, embeddings)
            zf.writestr("embeddings.npy", embeddings_bytes_io.getvalue())
            
            # Write embeddings.csv (backup format)
            embeddings_csv_io = BytesIO()
            np.savetxt(embeddings_csv_io, embeddings, delimiter=",")
            zf.writestr("embeddings.csv", embeddings_csv_io.getvalue())
            
            # Write citations.jsonl
            citations_lines = []
            for citation in citations_data:
                citations_lines.append(json.dumps(citation, ensure_ascii=False))
            citations_content = '\n'.join(citations_lines)
            zf.writestr("citations.jsonl", citations_content.encode('utf-8'))
            
            # Write manifest.json (v1.1)
            manifest_bytes = json.dumps(manifest_data, ensure_ascii=False, indent=2).encode('utf-8')
            zf.writestr("manifest.json", manifest_bytes)
        
        return zip_path
    
    def _write_directory_pack(self,
                             chunks_with_metadata: List[Dict[str, Any]],
                             embeddings: np.ndarray,
                             chunker_metadata: Dict[str, Any],
                             embedder_metadata: Dict[str, Any],
                             indexer_metadata: Dict[str, Any],
                             source_documents: List[Dict[str, Any]],
                             dir_path: Path) -> Path:
        """Write RAGpack to directory."""
        
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        chunks_json = [chunk['text'] for chunk in chunks_with_metadata]
        citations_data = self._generate_citations(chunks_with_metadata)
        manifest_data = self._generate_manifest(
            chunker_metadata, embedder_metadata, indexer_metadata, source_documents
        )
        
        # Write chunks.json
        with open(dir_path / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_json, f, ensure_ascii=False, indent=2)
        
        # Write embeddings.npy
        np.save(dir_path / "embeddings.npy", embeddings)
        
        # Write embeddings.csv (backup format)
        np.savetxt(dir_path / "embeddings.csv", embeddings, delimiter=",")
        
        # Write citations.jsonl
        with open(dir_path / "citations.jsonl", 'w', encoding='utf-8') as f:
            for citation in citations_data:
                f.write(json.dumps(citation, ensure_ascii=False) + '\n')
        
        # Write manifest.json (v1.1)
        with open(dir_path / "manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        
        return dir_path
    
    def _generate_citations(self, chunks_with_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate citations keyed to the app's RAGpackReader spec (PR #98).

        Each line of citations.jsonl is:
            {"chunk_index": N, "doc_id": ..., "page": ..., "char_start": ...,
             "char_end": ..., "paragraph_boundaries": [...], "snippet": ...}

        ``chunk_index`` is the row index that aligns with embeddings.npy.  The
        chunker's offset records use ``start_char``/``end_char``/``chunk_id``;
        those are normalized here to the app's ``char_start``/``char_end``/
        ``chunk_index`` names — the app reader is strict about field names.
        ``snippet`` is retained for preview (extra fields are ignored by the
        reader).
        """
        citations = []

        for row_index, chunk in enumerate(chunks_with_metadata):
            text = chunk.get('text', '')
            snippet = chunk.get('snippet')
            if snippet is None:
                snippet = text[:200] + '...' if len(text) > 200 else text

            citation = {
                # Row index into embeddings.npy. Prefer an explicit chunk_index;
                # fall back to the chunker's integer chunk_id, then enumeration.
                'chunk_index': chunk.get('chunk_index',
                                         chunk.get('chunk_id', row_index)),
                'doc_id': chunk.get('doc_id'),
                'page': chunk.get('page', chunk.get('page_number')),
                'char_start': chunk.get('char_start', chunk.get('start_char', 0)),
                'char_end': chunk.get('char_end', chunk.get('end_char', 0)),
                'paragraph_boundaries': chunk.get('paragraph_boundaries', []),
                'snippet': snippet,
            }
            citations.append(citation)

        return citations

    def _generate_manifest(self,
                          chunker_metadata: Dict[str, Any],
                          embedder_metadata: Dict[str, Any],
                          indexer_metadata: Dict[str, Any],
                          source_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate the nested manifest for the configured pack_version."""
        if self.pack_version == "1.2":
            return build_manifest_v1_2(
                pack_id=self.pack_id,
                created_at=self.created_at,
                chunker=chunker_metadata,
                embedder=embedder_metadata,
                indexer=indexer_metadata,
                files={
                    "chunks": "chunks.json",
                    "embeddings": "embeddings.npy",
                    "citations": "citations.jsonl",
                    "metadata": {
                        "embeddings_csv": "embeddings.csv",
                        "manifest": "manifest.json",
                    },
                },
                source_documents=source_documents,
            )

        # Legacy v1.1 shape (deprecated; not consumable by NoesisNoema v0.4+).
        return {
            "pack_version": "1.1",
            "pack_id": self.pack_id,
            "created_at": self.created_at,
            "chunker": chunker_metadata,
            "embedder": embedder_metadata,
            "indexer": indexer_metadata,
            "files": {
                "chunks": "chunks.json",
                "embeddings": "embeddings.npy",
                "citations": "citations.jsonl",
                "metadata": {
                    "embeddings_csv": "embeddings.csv",
                    "manifest": "manifest.json"
                }
            },
            "source_documents": source_documents
        }