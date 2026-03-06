"""
RagpackWriter — writes a Ragpack to a directory on disk.

Design contract (EPIC3 Stage 3)
--------------------------------
- Writes exactly three files: embeddings.npy, chunks.parquet, manifest.json.
- All three files are written deterministically: given the same Ragpack the
  byte content of each file is identical across runs.
- embeddings.npy uses np.save (standard NumPy binary format).
- chunks.parquet uses pyarrow with fixed schema, no dictionary encoding,
  sorted column order, and snappy compression (reproducible).
- manifest.json uses json.dumps with sort_keys=True and indent=2.
- The writer does NOT modify the Ragpack object or the existing PackWriter.
- output_dir is created (including parents) if it does not exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .ragpack_builder import Ragpack


# ---------------------------------------------------------------------------
# Parquet column schema (fixed order — column order must never change)
# ---------------------------------------------------------------------------

_CHUNKS_SCHEMA = pa.schema([
    pa.field("chunk_id",              pa.string(),  nullable=False),
    pa.field("source_id",             pa.string(),  nullable=False),
    pa.field("source_path",           pa.string(),  nullable=False),
    pa.field("source_hash",           pa.string(),  nullable=False),
    pa.field("chunk_index",           pa.int64(),   nullable=False),
    pa.field("char_start",            pa.int64(),   nullable=False),
    pa.field("char_end",              pa.int64(),   nullable=False),
    pa.field("token_count",           pa.int64(),   nullable=False),
    pa.field("text_snippet",          pa.string(),  nullable=False),
    pa.field("chunk_text_hash",       pa.string(),  nullable=False),
    pa.field("chunking_config_hash",  pa.string(),  nullable=False),
    pa.field("section",               pa.string(),  nullable=True),
    pa.field("page",                  pa.int64(),   nullable=True),
])


def _chunks_to_arrow_table(ragpack: Ragpack) -> pa.Table:
    """Convert the ChunkRecord list in a Ragpack to a pyarrow Table."""
    columns: dict = {field.name: [] for field in _CHUNKS_SCHEMA}

    for chunk in ragpack.chunks:
        columns["chunk_id"].append(chunk.chunk_id)
        columns["source_id"].append(chunk.source_id)
        columns["source_path"].append(chunk.source_path)
        columns["source_hash"].append(chunk.source_hash)
        columns["chunk_index"].append(chunk.chunk_index)
        columns["char_start"].append(chunk.char_start)
        columns["char_end"].append(chunk.char_end)
        columns["token_count"].append(chunk.token_count)
        columns["text_snippet"].append(chunk.text_snippet)
        columns["chunk_text_hash"].append(chunk.chunk_text_hash)
        columns["chunking_config_hash"].append(chunk.chunking_config_hash)
        columns["section"].append(chunk.section)
        columns["page"].append(chunk.page)

    arrays = [
        pa.array(columns[field.name], type=field.type)
        for field in _CHUNKS_SCHEMA
    ]
    return pa.table(arrays, schema=_CHUNKS_SCHEMA)


class RagpackWriter:
    """
    Writes a Ragpack to a target directory, producing exactly:

        <output_dir>/
            embeddings.npy
            chunks.parquet
            manifest.json

    All three files are written deterministically.

    Usage
    -----
    ::

        writer = RagpackWriter()
        paths  = writer.write(ragpack, output_dir="ragpack/")
    """

    def write(
        self,
        ragpack: Ragpack,
        output_dir: Union[str, Path],
    ) -> dict[str, Path]:
        """
        Write the Ragpack to output_dir.

        Args:
            ragpack:    A fully assembled Ragpack (from RagpackBuilder.build()).
            output_dir: Target directory path.  Created (including parents)
                        if it does not exist.

        Returns:
            Dict mapping artifact name to absolute Path:
            ``{"embeddings": ..., "chunks": ..., "manifest": ...}``

        Raises:
            TypeError:  if ragpack is not a Ragpack instance.
            ValueError: if output_dir resolves to an existing file (not a dir).
        """
        if not isinstance(ragpack, Ragpack):
            raise TypeError(
                f"ragpack must be a Ragpack, got {type(ragpack).__name__}"
            )

        target = Path(output_dir).resolve()
        if target.exists() and not target.is_dir():
            raise ValueError(
                f"output_dir '{target}' exists and is a file, not a directory"
            )
        target.mkdir(parents=True, exist_ok=True)

        embeddings_path = self._write_embeddings(ragpack, target)
        chunks_path     = self._write_chunks(ragpack, target)
        manifest_path   = self._write_manifest(ragpack, target)

        return {
            "embeddings": embeddings_path,
            "chunks":     chunks_path,
            "manifest":   manifest_path,
        }

    # ------------------------------------------------------------------
    # Private writers — each writes one file deterministically
    # ------------------------------------------------------------------

    def _write_embeddings(self, ragpack: Ragpack, target: Path) -> Path:
        """Write embeddings as a standard NumPy .npy file."""
        path = target / "embeddings.npy"
        np.save(str(path), ragpack.embeddings)
        return path

    def _write_chunks(self, ragpack: Ragpack, target: Path) -> Path:
        """
        Write chunks as a Parquet file.

        Determinism controls:
        - Fixed pyarrow schema with explicit column order.
        - No dictionary encoding (dictionary pages are order-sensitive).
        - data_page_version="1.0" for maximum reader compatibility.
        - write_statistics=False eliminates any non-deterministic stat blobs.
        - snappy compression is deterministic for the same input bytes.
        - Row group size fixed at 128 MB (effectively one row group for
          typical packs) to avoid split-point non-determinism.
        """
        path = target / "chunks.parquet"
        table = _chunks_to_arrow_table(ragpack)
        pq.write_table(
            table,
            str(path),
            compression="snappy",
            use_dictionary=False,
            data_page_version="1.0",
            write_statistics=False,
            row_group_size=128 * 1024 * 1024,
        )
        return path

    def _write_manifest(self, ragpack: Ragpack, target: Path) -> Path:
        """Write manifest as UTF-8 JSON with stable key ordering."""
        path = target / "manifest.json"
        content = json.dumps(ragpack.manifest, sort_keys=True, indent=2,
                             ensure_ascii=False, default=str)
        path.write_text(content, encoding="utf-8")
        return path

