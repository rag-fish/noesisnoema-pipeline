"""
nn-pipeline build — CLI entrypoint for deterministic RAGPack generation.

Design contract (EPIC3 Stage 3 — CLI component)
-------------------------------------------------
Doctrine alignment (RAGFish invocation-boundary.md, execution-flow.md):
- Every execution is explicitly triggered by a human action (CLI invocation).
- All inputs are explicit arguments; nothing is inferred from environment
  globals, wall clocks called implicitly, or auto-discovered at runtime.
- creation_time is a required argument so output is fully reproducible
  from the same inputs without patching datetime inside the pipeline.
- Files are enumerated and sorted deterministically before processing.
- No background threads, no autonomous retries, no hidden side effects.
- All errors surface as structured, human-readable messages; nothing is
  swallowed silently.
- The pipeline logic (run_pipeline) is separated from the CLI layer so
  it can be tested directly without subprocess invocation.

Usage
-----
    nn-pipeline build \\
        --input_dir  ./docs \\
        --output_dir ./ragpack \\
        --chunk_size 512 \\
        --overlap    50 \\
        --creation_time 2026-03-07T00:00:00
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List

import typer

from chunker import TokenChunker
from chunker.chunk_record import ChunkRecord
from embedder.deterministic_embedder import DEFAULT_MODEL_NAME, DeterministicEmbedder
from ragpack import RagpackBuilder, RagpackWriter


# ---------------------------------------------------------------------------
# Supported source file extensions (deterministic, fixed set)
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="nn-pipeline",
    help="Deterministic RAGPack generator — EPIC3 Stage 3.",
    add_completion=False,
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# PipelineResult — returned by run_pipeline for testability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineResult:
    """
    Outcome of a single run_pipeline() call.

    Fields
    ------
    output_dir      Resolved output directory path.
    chunk_count     Total number of chunks produced.
    file_count      Number of source files processed.
    source_files    Sorted list of source file paths that were processed.
    written_paths   Dict mapping artifact name → absolute Path on disk.
    """

    output_dir: Path
    chunk_count: int
    file_count: int
    source_files: List[Path]
    written_paths: dict


# ---------------------------------------------------------------------------
# Source file helpers
# ---------------------------------------------------------------------------

def _collect_source_files(input_dir: Path) -> List[Path]:
    """
    Return a sorted, deterministic list of supported source files
    found directly inside input_dir (non-recursive).

    Sorting is by filename (case-sensitive, lexicographic) so the order
    is identical across all operating systems and file systems.

    Only files with extensions in _SUPPORTED_EXTENSIONS are included.
    Hidden files (names starting with '.') are excluded.

    Raises:
        typer.BadParameter: if input_dir does not exist or is not a directory.
        typer.Exit:         if no supported files are found.
    """
    if not input_dir.exists():
        typer.echo(f"ERROR: input_dir '{input_dir}' does not exist.", err=True)
        raise typer.Exit(code=1)
    if not input_dir.is_dir():
        typer.echo(f"ERROR: input_dir '{input_dir}' is not a directory.", err=True)
        raise typer.Exit(code=1)

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if not files:
        typer.echo(
            f"ERROR: No supported files ({', '.join(sorted(_SUPPORTED_EXTENSIONS))}) "
            f"found in '{input_dir}'.",
            err=True,
        )
        raise typer.Exit(code=1)

    return files


def _compute_source_hash(file_path: Path) -> str:
    """Return a SHA-256 hex digest of the raw file content bytes."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _compute_source_id(file_path: Path, source_hash: str) -> str:
    """
    Return a stable source_id derived from the canonical file name and
    its content hash, joined by a null byte.

    Using only the file name (not the full path) keeps source_id stable
    when the pack is moved between machines.
    """
    payload = f"{file_path.name}\x00{source_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Core pipeline — separated from CLI so it is directly testable
# ---------------------------------------------------------------------------

def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    creation_time: str,
    model_name: str = DEFAULT_MODEL_NAME,
    verbose: bool = False,
) -> PipelineResult:
    """
    Execute the full deterministic RAGPack generation pipeline.

    This function is the single authorised pipeline entry point.  It is
    called by the ``build`` CLI command and can also be called directly
    in tests without subprocess overhead.

    Pipeline steps (in order):
        1. Collect and sort source files from input_dir.
        2. For each file (in sorted order):
           a. Read content.
           b. Compute source_hash and source_id.
           c. Chunk with TokenChunker using the supplied configuration.
        3. All ChunkRecords are accumulated in file-sort order.
        4. RagpackBuilder embeds and assembles the Ragpack.
        5. RagpackWriter writes the three artifacts to output_dir.

    Args:
        input_dir:     Directory containing source text/markdown files.
        output_dir:    Directory where artifacts will be written.
        chunk_size:    Maximum tokens per chunk.
        overlap:       Token overlap between consecutive chunks.
        creation_time: ISO-8601 timestamp string (caller-supplied for
                       reproducibility; never generated internally).
        model_name:    HuggingFace embedding model identifier.
        verbose:       Whether to emit progress lines to stdout.

    Returns:
        PipelineResult with outcome metadata.

    Raises:
        typer.Exit(code=1): on any recoverable error.
        Any unhandled exception propagates to the caller.
    """
    source_files = _collect_source_files(input_dir)

    chunker = TokenChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        preserve_sentences=False,
    )

    all_chunks: List[ChunkRecord] = []
    source_docs: list = []

    for file_path in source_files:
        if verbose:
            typer.echo(f"  Processing: {file_path.name}")

        raw_bytes = file_path.read_bytes()
        text = raw_bytes.decode("utf-8", errors="replace")
        source_hash = hashlib.sha256(raw_bytes).hexdigest()
        source_id = _compute_source_id(file_path, source_hash)

        file_chunks = chunker.chunk_document(
            text=text,
            source_id=source_id,
            source_path=str(file_path),
            source_hash=source_hash,
        )
        all_chunks.extend(file_chunks)

        source_docs.append({
            "source_id":   source_id,
            "source_path": str(file_path),
            "source_hash": source_hash,
            "file_name":   file_path.name,
            "chunk_count": len(file_chunks),
        })

    if verbose:
        typer.echo(f"  Total chunks: {len(all_chunks)}")
        typer.echo(f"  Loading embedder: {model_name}")

    embedder = DeterministicEmbedder(model_name)
    builder  = RagpackBuilder(embedder)
    ragpack  = builder.build(
        chunks=all_chunks,
        creation_time=creation_time,
        source_documents=source_docs,
    )

    writer = RagpackWriter()
    written_paths = writer.write(ragpack, output_dir)

    return PipelineResult(
        output_dir=output_dir.resolve(),
        chunk_count=len(all_chunks),
        file_count=len(source_files),
        source_files=source_files,
        written_paths=written_paths,
    )


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@app.command("build")
def build(
    input_dir: Path = typer.Option(
        ...,
        "--input_dir",
        help="Directory containing source .txt or .md files to chunk and embed.",
        exists=False,           # validated manually for better error messages
        file_okay=False,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output_dir",
        help="Directory where ragpack artifacts will be written.",
        resolve_path=True,
    ),
    chunk_size: int = typer.Option(
        512,
        "--chunk_size",
        help="Maximum number of tokens per chunk.",
        min=1,
    ),
    overlap: int = typer.Option(
        50,
        "--overlap",
        help="Number of tokens to overlap between consecutive chunks.",
        min=0,
    ),
    creation_time: str = typer.Option(
        ...,
        "--creation_time",
        help=(
            "ISO-8601 timestamp to embed in the manifest, e.g. "
            "2026-03-07T00:00:00.  Must be supplied explicitly so "
            "the output is reproducible."
        ),
    ),
    model: str = typer.Option(
        DEFAULT_MODEL_NAME,
        "--model",
        help="HuggingFace embedding model identifier.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Emit progress information to stdout.",
    ),
) -> None:
    """
    Build a deterministic RAGPack from documents in INPUT_DIR.

    Supported file types: .txt, .md

    Files are processed in sorted (lexicographic) order so output is
    identical across runs with the same inputs and creation_time.
    """
    # --- Validate overlap vs chunk_size before touching the filesystem ---
    if overlap >= chunk_size:
        typer.echo(
            f"ERROR: --overlap ({overlap}) must be less than --chunk_size ({chunk_size}).",
            err=True,
        )
        raise typer.Exit(code=1)

    if not creation_time.strip():
        typer.echo("ERROR: --creation_time must not be empty.", err=True)
        raise typer.Exit(code=1)

    if verbose:
        typer.echo(f"nn-pipeline build")
        typer.echo(f"  input_dir:     {input_dir}")
        typer.echo(f"  output_dir:    {output_dir}")
        typer.echo(f"  chunk_size:    {chunk_size}")
        typer.echo(f"  overlap:       {overlap}")
        typer.echo(f"  creation_time: {creation_time}")
        typer.echo(f"  model:         {model}")

    try:
        result = run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            creation_time=creation_time,
            model_name=model,
            verbose=verbose,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"ERROR: Pipeline failed — {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"RAGPack written to: {result.output_dir}\n"
        f"  Files processed:  {result.file_count}\n"
        f"  Chunks produced:  {result.chunk_count}\n"
        f"  Artifacts:\n"
        + "\n".join(f"    {k}: {v}" for k, v in result.written_paths.items())
    )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()

