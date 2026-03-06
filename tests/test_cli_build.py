"""
Stage 3 CLI tests: nn-pipeline build command.

Tests in this file validate:
- test_cli_build_smoke            — pipeline runs end-to-end, three artifacts created
- test_cli_deterministic_output   — identical inputs produce identical artifact bytes

All tests call run_pipeline() directly (no subprocess) so they are fast,
isolated, and independent of shell PATH configuration.
"""

import hashlib
import json
import sys
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cli.build_ragpack import (
    PipelineResult,
    _collect_source_files,
    _compute_source_hash,
    _compute_source_id,
    run_pipeline,
)
from ragpack.manifest_builder import REQUIRED_MANIFEST_FIELDS


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CREATION_TIME = "2026-03-07T00:00:00"

_DOC_A = """\
Artificial intelligence is transforming every industry.
Machine learning models learn from data to make predictions.
Natural language processing enables computers to understand human text.
Deep learning uses many-layered neural networks for complex tasks.
"""

_DOC_B = """\
The retrieval-augmented generation pattern combines search with generation.
A vector store indexes embeddings of document chunks for similarity lookup.
At query time the nearest chunks are retrieved and injected into the prompt.
This grounds the model response in verifiable source material.
"""


def _make_input_dir(tmp: str, docs: dict | None = None) -> Path:
    """
    Create a temp input directory with synthetic .txt files.

    Args:
        tmp:  tempfile.TemporaryDirectory path string.
        docs: Mapping of filename → content.  Defaults to two standard docs.
    """
    if docs is None:
        docs = {"doc_a.txt": _DOC_A, "doc_b.txt": _DOC_B}
    input_dir = Path(tmp) / "input"
    input_dir.mkdir()
    for name, content in docs.items():
        (input_dir / name).write_text(content, encoding="utf-8")
    return input_dir


def _run(input_dir: Path, output_dir: Path, **kwargs) -> PipelineResult:
    """Run pipeline with test defaults."""
    defaults = dict(
        chunk_size=40,
        overlap=5,
        creation_time=_CREATION_TIME,
    )
    defaults.update(kwargs)
    return run_pipeline(input_dir=input_dir, output_dir=output_dir, **defaults)


# ---------------------------------------------------------------------------
# _collect_source_files unit tests
# ---------------------------------------------------------------------------

class TestCollectSourceFiles(unittest.TestCase):
    """_collect_source_files must return a stable sorted list."""

    def test_returns_sorted_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ["z_last.txt", "a_first.txt", "m_middle.md"]:
                (d / name).write_text("x")
            files = _collect_source_files(d)
        names = [f.name for f in files]
        self.assertEqual(names, sorted(names))

    def test_filters_unsupported_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "keep.txt").write_text("x")
            (d / "keep.md").write_text("x")
            (d / "skip.py").write_text("x")
            (d / "skip.pdf").write_text("x")
            files = _collect_source_files(d)
        names = {f.name for f in files}
        self.assertIn("keep.txt",  names)
        self.assertIn("keep.md",   names)
        self.assertNotIn("skip.py",  names)
        self.assertNotIn("skip.pdf", names)

    def test_excludes_hidden_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / ".hidden.txt").write_text("x")
            (d / "visible.txt").write_text("x")
            files = _collect_source_files(d)
        names = {f.name for f in files}
        self.assertNotIn(".hidden.txt", names)
        self.assertIn("visible.txt",    names)

    def test_missing_dir_exits(self):
        import typer
        with self.assertRaises(typer.Exit):
            _collect_source_files(Path("/nonexistent_dir_abc_xyz"))

    def test_empty_dir_exits(self):
        import typer
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(typer.Exit):
                _collect_source_files(Path(tmp))

    def test_two_calls_return_same_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ["b.txt", "a.txt", "c.md"]:
                (d / name).write_text("x")
            first  = [f.name for f in _collect_source_files(d)]
            second = [f.name for f in _collect_source_files(d)]
        self.assertEqual(first, second)


# ---------------------------------------------------------------------------
# Source hash / id helpers
# ---------------------------------------------------------------------------

class TestSourceHelpers(unittest.TestCase):

    def test_source_hash_is_64_char_hex(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello world")
            p = Path(f.name)
        h = _compute_source_hash(p)
        p.unlink()
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_source_hash_matches_manual_sha256(self):
        content = b"deterministic content"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(content)
            p = Path(f.name)
        expected = hashlib.sha256(content).hexdigest()
        actual = _compute_source_hash(p)
        p.unlink()
        self.assertEqual(actual, expected)

    def test_source_id_stable_for_same_inputs(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"content")
            p = Path(f.name)
        h = _compute_source_hash(p)
        id_a = _compute_source_id(p, h)
        id_b = _compute_source_id(p, h)
        p.unlink()
        self.assertEqual(id_a, id_b)

    def test_source_id_changes_when_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "doc.txt"
            p.write_bytes(b"version one")
            h1 = _compute_source_hash(p)
            id1 = _compute_source_id(p, h1)
            p.write_bytes(b"version two")
            h2 = _compute_source_hash(p)
            id2 = _compute_source_id(p, h2)
        self.assertNotEqual(id1, id2)


# ---------------------------------------------------------------------------
# test_cli_build_smoke
# ---------------------------------------------------------------------------

class TestCliBuildSmoke(unittest.TestCase):
    """
    End-to-end smoke test: pipeline runs without error and produces the
    three required artifacts with correct structure.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._input_dir  = _make_input_dir(self._tmp.name)
        self._output_dir = Path(self._tmp.name) / "ragpack"

    def tearDown(self):
        self._tmp.cleanup()

    def _result(self) -> PipelineResult:
        return _run(self._input_dir, self._output_dir)

    def test_returns_pipeline_result(self):
        self.assertIsInstance(self._result(), PipelineResult)

    def test_three_artifact_files_exist(self):
        result = self._result()
        self.assertTrue(result.written_paths["embeddings"].exists())
        self.assertTrue(result.written_paths["chunks"].exists())
        self.assertTrue(result.written_paths["manifest"].exists())

    def test_artifact_filenames_correct(self):
        result = self._result()
        self.assertEqual(result.written_paths["embeddings"].name, "embeddings.npy")
        self.assertEqual(result.written_paths["chunks"].name,     "chunks.parquet")
        self.assertEqual(result.written_paths["manifest"].name,   "manifest.json")

    def test_manifest_contains_required_fields(self):
        result   = self._result()
        manifest = json.loads(result.written_paths["manifest"].read_text(encoding="utf-8"))
        for field in REQUIRED_MANIFEST_FIELDS:
            self.assertIn(field, manifest, f"Required field '{field}' missing")

    def test_manifest_chunk_count_positive(self):
        result   = self._result()
        manifest = json.loads(result.written_paths["manifest"].read_text(encoding="utf-8"))
        self.assertGreater(manifest["chunk_count"], 0)

    def test_manifest_chunk_count_matches_result(self):
        result   = self._result()
        manifest = json.loads(result.written_paths["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["chunk_count"], result.chunk_count)

    def test_manifest_creation_time_matches_input(self):
        result   = self._result()
        manifest = json.loads(result.written_paths["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["creation_time"], _CREATION_TIME)

    def test_embeddings_npy_loads_as_float32_array(self):
        result     = self._result()
        embeddings = np.load(str(result.written_paths["embeddings"]))
        self.assertEqual(embeddings.ndim, 2)
        self.assertEqual(embeddings.dtype, np.float32)

    def test_embeddings_row_count_matches_chunk_count(self):
        result     = self._result()
        embeddings = np.load(str(result.written_paths["embeddings"]))
        manifest   = json.loads(result.written_paths["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(embeddings.shape[0], manifest["chunk_count"])

    def test_chunks_parquet_row_count_matches_chunk_count(self):
        result = self._result()
        table  = pq.read_table(str(result.written_paths["chunks"]))
        self.assertEqual(table.num_rows, result.chunk_count)

    def test_chunks_parquet_required_columns_present(self):
        result = self._result()
        table  = pq.read_table(str(result.written_paths["chunks"]))
        for col in ["chunk_id", "source_id", "source_path", "source_hash",
                    "chunk_index", "char_start", "char_end", "token_count",
                    "text_snippet", "chunk_text_hash", "chunking_config_hash"]:
            self.assertIn(col, table.schema.names,
                          f"Required column '{col}' missing from chunks.parquet")

    def test_file_count_matches_input_files(self):
        result = self._result()
        self.assertEqual(result.file_count, 2)

    def test_source_files_list_is_sorted(self):
        result = self._result()
        names  = [f.name for f in result.source_files]
        self.assertEqual(names, sorted(names))

    def test_no_nan_or_inf_in_embeddings(self):
        result     = self._result()
        embeddings = np.load(str(result.written_paths["embeddings"]))
        self.assertTrue(np.all(np.isfinite(embeddings)),
                        "Embedding array contains NaN or Inf")

    def test_output_dir_created_automatically(self):
        deep_output = Path(self._tmp.name) / "nested" / "deep" / "ragpack"
        self.assertFalse(deep_output.exists())
        _run(self._input_dir, deep_output)
        self.assertTrue(deep_output.is_dir())

    def test_single_file_input_works(self):
        """A single-file input directory must produce a valid pack."""
        with tempfile.TemporaryDirectory() as tmp:
            single_dir = Path(tmp) / "single"
            single_dir.mkdir()
            (single_dir / "only.txt").write_text(_DOC_A, encoding="utf-8")
            out_dir = Path(tmp) / "out"
            result = _run(single_dir, out_dir)
        self.assertGreater(result.chunk_count, 0)
        self.assertEqual(result.file_count, 1)


# ---------------------------------------------------------------------------
# test_cli_deterministic_output
# ---------------------------------------------------------------------------

class TestCliDeterministicOutput(unittest.TestCase):
    """
    Two pipeline runs with identical inputs must produce byte-identical
    artifacts on disk.
    """

    def test_embeddings_npy_identical_across_two_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = _make_input_dir(tmp)
            out_a = Path(tmp) / "out_a"
            out_b = Path(tmp) / "out_b"
            _run(input_dir, out_a)
            _run(input_dir, out_b)
            bytes_a = (out_a / "embeddings.npy").read_bytes()
            bytes_b = (out_b / "embeddings.npy").read_bytes()
        self.assertEqual(bytes_a, bytes_b,
                         "embeddings.npy bytes differ between two runs")

    def test_manifest_json_identical_across_two_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = _make_input_dir(tmp)
            out_a = Path(tmp) / "out_a"
            out_b = Path(tmp) / "out_b"
            _run(input_dir, out_a)
            _run(input_dir, out_b)
            text_a = (out_a / "manifest.json").read_text(encoding="utf-8")
            text_b = (out_b / "manifest.json").read_text(encoding="utf-8")
        self.assertEqual(text_a, text_b,
                         "manifest.json content differs between two runs")

    def test_chunks_parquet_identical_across_two_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = _make_input_dir(tmp)
            out_a = Path(tmp) / "out_a"
            out_b = Path(tmp) / "out_b"
            _run(input_dir, out_a)
            _run(input_dir, out_b)
            bytes_a = (out_a / "chunks.parquet").read_bytes()
            bytes_b = (out_b / "chunks.parquet").read_bytes()
        self.assertEqual(bytes_a, bytes_b,
                         "chunks.parquet bytes differ between two runs")

    def test_different_creation_times_produce_different_manifests(self):
        """creation_time must propagate into the manifest; different times differ."""
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = _make_input_dir(tmp)
            out_a = Path(tmp) / "out_a"
            out_b = Path(tmp) / "out_b"
            _run(input_dir, out_a, creation_time="2026-01-01T00:00:00")
            _run(input_dir, out_b, creation_time="2026-06-01T00:00:00")
            m_a = json.loads((out_a / "manifest.json").read_text())
            m_b = json.loads((out_b / "manifest.json").read_text())
        self.assertNotEqual(m_a["manifest_hash"], m_b["manifest_hash"])

    def test_chunk_ids_identical_across_two_runs(self):
        """chunk_ids in the parquet must be identical between two runs."""
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = _make_input_dir(tmp)
            out_a = Path(tmp) / "out_a"
            out_b = Path(tmp) / "out_b"
            _run(input_dir, out_a)
            _run(input_dir, out_b)
            ids_a = pq.read_table(str(out_a / "chunks.parquet")).column("chunk_id").to_pylist()
            ids_b = pq.read_table(str(out_b / "chunks.parquet")).column("chunk_id").to_pylist()
        self.assertEqual(ids_a, ids_b,
                         "chunk_id columns differ between two runs")

    def test_adding_file_changes_output(self):
        """Adding a file must change the manifest hash — new content ≠ same pack."""
        with tempfile.TemporaryDirectory() as tmp:
            input_a = _make_input_dir(tmp, {"doc_a.txt": _DOC_A})

            input_ab = Path(tmp) / "input_ab"
            input_ab.mkdir()
            (input_ab / "doc_a.txt").write_text(_DOC_A, encoding="utf-8")
            (input_ab / "doc_b.txt").write_text(_DOC_B, encoding="utf-8")

            out_a  = Path(tmp) / "out_a"
            out_ab = Path(tmp) / "out_ab"
            _run(input_a,  out_a)
            _run(input_ab, out_ab)
            m_a  = json.loads((out_a  / "manifest.json").read_text())
            m_ab = json.loads((out_ab / "manifest.json").read_text())
        self.assertNotEqual(m_a["manifest_hash"], m_ab["manifest_hash"])

    def test_file_order_independent_of_creation_order(self):
        """
        Chunk IDs must be identical whether files are created in a-b or b-a
        order, because sorting is by name, not creation timestamp.
        """
        with tempfile.TemporaryDirectory() as tmp:
            # First run: create a then b
            in1 = Path(tmp) / "in1"
            in1.mkdir()
            (in1 / "alpha.txt").write_text(_DOC_A, encoding="utf-8")
            (in1 / "beta.txt").write_text(_DOC_B, encoding="utf-8")

            # Second run: create b then a
            in2 = Path(tmp) / "in2"
            in2.mkdir()
            (in2 / "beta.txt").write_text(_DOC_B, encoding="utf-8")
            (in2 / "alpha.txt").write_text(_DOC_A, encoding="utf-8")

            out1 = Path(tmp) / "out1"
            out2 = Path(tmp) / "out2"
            _run(in1, out1)
            _run(in2, out2)

            ids1 = pq.read_table(str(out1 / "chunks.parquet")).column("chunk_id").to_pylist()
            ids2 = pq.read_table(str(out2 / "chunks.parquet")).column("chunk_id").to_pylist()

        self.assertEqual(ids1, ids2,
                         "chunk_ids differ when files are created in different order")


# ---------------------------------------------------------------------------
# Pipeline guard tests
# ---------------------------------------------------------------------------

class TestPipelineGuards(unittest.TestCase):

    def test_missing_input_dir_exits(self):
        import typer
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(typer.Exit):
                run_pipeline(
                    input_dir=Path("/does_not_exist_xyz"),
                    output_dir=Path(tmp) / "out",
                    chunk_size=40,
                    overlap=5,
                    creation_time=_CREATION_TIME,
                )

    def test_overlap_gte_chunk_size_raises(self):
        """overlap >= chunk_size must be caught before the pipeline runs."""
        import typer
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = _make_input_dir(tmp)
            out_dir   = Path(tmp) / "out"
            raised = False
            try:
                run_pipeline(
                    input_dir=input_dir,
                    output_dir=out_dir,
                    chunk_size=10,
                    overlap=10,          # equal — must fail
                    creation_time=_CREATION_TIME,
                )
            except (ValueError, typer.Exit, SystemExit):
                raised = True
            self.assertTrue(raised, "Expected error when overlap >= chunk_size")


if __name__ == "__main__":
    unittest.main()

