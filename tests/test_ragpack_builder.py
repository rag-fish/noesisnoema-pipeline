"""
Stage 3 contract tests: RAGPack builder, writer, and manifest.

Tests in this file validate the EPIC3 Stage 3 deliverables:
- test_ragpack_repeatability       — two builds with same inputs → identical files
- test_manifest_complete           — manifest contains every required field
- test_embedding_chunk_alignment   — embeddings rows align with chunk order
- test_ragpack_roundtrip           — write to disk → read back → assert fidelity
"""

import json
import sys
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chunker import TokenChunker
from embedder import DeterministicEmbedder
from embedder.deterministic_embedder import DEFAULT_MODEL_NAME
from ragpack import ManifestBuilder, Ragpack, RagpackBuilder, RagpackWriter
from ragpack.manifest_builder import REQUIRED_MANIFEST_FIELDS


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SOURCE_ID   = "src-stage3-test"
_SOURCE_PATH = "tests/fixtures/sample.txt"
_SOURCE_HASH = "c" * 64
_CREATION_TIME = "2026-03-06T00:00:00"

_TEXT = (
    "Artificial intelligence is transforming every industry. "
    "Machine learning models learn from data to make predictions. "
    "Natural language processing enables computers to understand human text. "
    "Deep learning uses many-layered neural networks for complex tasks. "
) * 3


def _make_records(text: str = _TEXT, chunk_size: int = 40, overlap: int = 5):
    chunker = TokenChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        preserve_sentences=False,
    )
    return chunker.chunk_document(
        text=text,
        source_id=_SOURCE_ID,
        source_path=_SOURCE_PATH,
        source_hash=_SOURCE_HASH,
    )


# One shared embedder to avoid loading the model on every test.
_EMBEDDER: DeterministicEmbedder | None = None


def _get_embedder() -> DeterministicEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = DeterministicEmbedder(DEFAULT_MODEL_NAME)
    return _EMBEDDER


def _build_ragpack(records=None) -> Ragpack:
    if records is None:
        records = _make_records()
    builder = RagpackBuilder(_get_embedder())
    return builder.build(chunks=records, creation_time=_CREATION_TIME)


# ---------------------------------------------------------------------------
# ManifestBuilder unit tests
# ---------------------------------------------------------------------------

class TestManifestBuilder(unittest.TestCase):

    def _make_builder(self, **overrides) -> ManifestBuilder:
        kwargs = dict(
            chunk_count=10,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=384,
            model_hash="a" * 64,
            chunking_config_hash="b" * 64,
            dtype="float32",
            creation_time=_CREATION_TIME,
            embedding_version="3.4.1",
        )
        kwargs.update(overrides)
        return ManifestBuilder(**kwargs)

    def test_build_returns_dict(self):
        self.assertIsInstance(self._make_builder().build(), dict)

    def test_all_required_fields_present(self):
        manifest = self._make_builder().build()
        for field in REQUIRED_MANIFEST_FIELDS:
            self.assertIn(field, manifest, f"Required field '{field}' missing from manifest")

    def test_chunk_count_stored_correctly(self):
        manifest = self._make_builder(chunk_count=42).build()
        self.assertEqual(manifest["chunk_count"], 42)

    def test_embedding_model_stored_correctly(self):
        manifest = self._make_builder().build()
        self.assertEqual(manifest["embedding_model"],
                         "sentence-transformers/all-MiniLM-L6-v2")

    def test_embedding_dimension_stored_correctly(self):
        manifest = self._make_builder().build()
        self.assertEqual(manifest["embedding_dimension"], 384)

    def test_model_hash_stored_correctly(self):
        manifest = self._make_builder().build()
        self.assertEqual(manifest["model_hash"], "a" * 64)

    def test_chunking_config_hash_stored_correctly(self):
        manifest = self._make_builder().build()
        self.assertEqual(manifest["chunking_config_hash"], "b" * 64)

    def test_dtype_stored_correctly(self):
        manifest = self._make_builder().build()
        self.assertEqual(manifest["dtype"], "float32")

    def test_creation_time_stored_correctly(self):
        manifest = self._make_builder().build()
        self.assertEqual(manifest["creation_time"], _CREATION_TIME)

    def test_manifest_hash_is_present_and_64_char_hex(self):
        manifest = self._make_builder().build()
        self.assertIn("manifest_hash", manifest)
        h = manifest["manifest_hash"]
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_manifest_hash_changes_when_chunk_count_changes(self):
        m1 = self._make_builder(chunk_count=10).build()
        m2 = self._make_builder(chunk_count=11).build()
        self.assertNotEqual(m1["manifest_hash"], m2["manifest_hash"])

    def test_same_inputs_produce_same_manifest_hash(self):
        m1 = self._make_builder().build()
        m2 = self._make_builder().build()
        self.assertEqual(m1["manifest_hash"], m2["manifest_hash"])

    def test_negative_chunk_count_raises(self):
        with self.assertRaises(ValueError):
            self._make_builder(chunk_count=-1)

    def test_empty_embedding_model_raises(self):
        with self.assertRaises(ValueError):
            self._make_builder(embedding_model="")

    def test_empty_creation_time_raises(self):
        with self.assertRaises(ValueError):
            self._make_builder(creation_time="")

    def test_files_block_points_to_three_artifacts(self):
        manifest = self._make_builder().build()
        files = manifest["files"]
        self.assertIn("embeddings", files)
        self.assertIn("chunks", files)
        self.assertIn("manifest", files)
        self.assertEqual(files["embeddings"], "embeddings.npy")
        self.assertEqual(files["chunks"],     "chunks.parquet")
        self.assertEqual(files["manifest"],   "manifest.json")


# ---------------------------------------------------------------------------
# test_manifest_complete  (end-to-end, from RagpackBuilder output)
# ---------------------------------------------------------------------------

class TestManifestComplete(unittest.TestCase):
    """Manifest produced by RagpackBuilder must contain all required fields."""

    def test_manifest_contains_all_required_fields(self):
        ragpack = _build_ragpack()
        for field in REQUIRED_MANIFEST_FIELDS:
            self.assertIn(field, ragpack.manifest,
                          f"Required field '{field}' missing from ragpack.manifest")

    def test_manifest_chunk_count_matches_actual_chunks(self):
        records = _make_records()
        ragpack = _build_ragpack(records)
        self.assertEqual(ragpack.manifest["chunk_count"], len(records))

    def test_manifest_embedding_model_matches_embedder(self):
        ragpack = _build_ragpack()
        self.assertEqual(ragpack.manifest["embedding_model"],
                         _get_embedder().metadata.embedding_model)

    def test_manifest_embedding_dimension_matches_embedder(self):
        ragpack = _build_ragpack()
        self.assertEqual(ragpack.manifest["embedding_dimension"],
                         _get_embedder().metadata.embedding_dimension)

    def test_manifest_model_hash_matches_embedder(self):
        ragpack = _build_ragpack()
        self.assertEqual(ragpack.manifest["model_hash"],
                         _get_embedder().metadata.model_hash)

    def test_manifest_chunking_config_hash_matches_first_chunk(self):
        records = _make_records()
        ragpack = _build_ragpack(records)
        self.assertEqual(ragpack.manifest["chunking_config_hash"],
                         records[0].chunking_config_hash)

    def test_manifest_dtype_is_float32(self):
        ragpack = _build_ragpack()
        self.assertEqual(ragpack.manifest["dtype"], "float32")

    def test_manifest_creation_time_matches_input(self):
        ragpack = _build_ragpack()
        self.assertEqual(ragpack.manifest["creation_time"], _CREATION_TIME)

    def test_manifest_is_json_serialisable(self):
        ragpack = _build_ragpack()
        try:
            json.dumps(ragpack.manifest)
        except (TypeError, ValueError) as exc:
            self.fail(f"manifest is not JSON-serialisable: {exc}")


# ---------------------------------------------------------------------------
# test_embedding_chunk_alignment
# ---------------------------------------------------------------------------

class TestEmbeddingChunkAlignment(unittest.TestCase):
    """Embedding row i must correspond to chunk i in every produced Ragpack."""

    def test_row_count_equals_chunk_count(self):
        records = _make_records()
        ragpack = _build_ragpack(records)
        self.assertEqual(ragpack.embeddings.shape[0], len(records))

    def test_chunk_ids_list_aligned_with_chunks(self):
        records = _make_records()
        ragpack = _build_ragpack(records)
        for idx, (cid, chunk) in enumerate(zip(ragpack.chunk_ids, ragpack.chunks)):
            self.assertEqual(cid, chunk.chunk_id,
                             f"chunk_ids[{idx}] does not match chunks[{idx}].chunk_id")

    def test_embedding_shape_n_by_d(self):
        ragpack = _build_ragpack()
        d = _get_embedder().metadata.embedding_dimension
        self.assertEqual(ragpack.embeddings.ndim, 2)
        self.assertEqual(ragpack.embeddings.shape[1], d)

    def test_embedding_dtype_is_float32(self):
        ragpack = _build_ragpack()
        self.assertEqual(ragpack.embeddings.dtype, np.float32)

    def test_no_nan_or_inf_in_embeddings(self):
        ragpack = _build_ragpack()
        self.assertTrue(np.all(np.isfinite(ragpack.embeddings)),
                        "Embedding array contains NaN or Inf")

    def test_reversed_chunks_produce_reversed_embedding_rows(self):
        records = _make_records()
        self.assertGreater(len(records), 1)

        builder = RagpackBuilder(_get_embedder())
        pack_fwd = builder.build(chunks=records,              creation_time=_CREATION_TIME)
        pack_rev = builder.build(chunks=list(reversed(records)), creation_time=_CREATION_TIME)

        n = len(records)
        for i in range(n):
            np.testing.assert_array_equal(
                pack_rev.embeddings[i],
                pack_fwd.embeddings[n - 1 - i],
                err_msg=f"Row {i} of reversed pack does not match row {n-1-i} of forward pack",
            )

    def test_chunk_records_unchanged_after_build(self):
        """RagpackBuilder must not mutate the input ChunkRecord objects."""
        records = _make_records()
        original_ids = [r.chunk_id for r in records]
        _build_ragpack(records)
        after_ids = [r.chunk_id for r in records]
        self.assertEqual(original_ids, after_ids)


# ---------------------------------------------------------------------------
# test_ragpack_repeatability
# ---------------------------------------------------------------------------

class TestRagpackRepeatability(unittest.TestCase):
    """Two builds with identical inputs must produce byte-identical artifacts."""

    def test_two_builds_produce_identical_embeddings(self):
        records = _make_records()
        builder = RagpackBuilder(_get_embedder())

        pack_a = builder.build(chunks=records, creation_time=_CREATION_TIME)
        pack_b = builder.build(chunks=records, creation_time=_CREATION_TIME)

        np.testing.assert_array_equal(
            pack_a.embeddings, pack_b.embeddings,
            err_msg="Embeddings differ between two builds with identical input",
        )

    def test_two_builds_produce_identical_manifests(self):
        records = _make_records()
        builder = RagpackBuilder(_get_embedder())

        pack_a = builder.build(chunks=records, creation_time=_CREATION_TIME)
        pack_b = builder.build(chunks=records, creation_time=_CREATION_TIME)

        self.assertEqual(pack_a.manifest, pack_b.manifest)

    def test_two_builds_produce_identical_chunk_ids(self):
        records = _make_records()
        builder = RagpackBuilder(_get_embedder())

        pack_a = builder.build(chunks=records, creation_time=_CREATION_TIME)
        pack_b = builder.build(chunks=records, creation_time=_CREATION_TIME)

        self.assertEqual(pack_a.chunk_ids, pack_b.chunk_ids)

    def test_different_creation_times_produce_different_manifest_hashes(self):
        """creation_time must influence the manifest hash."""
        records = _make_records()
        builder = RagpackBuilder(_get_embedder())

        pack_a = builder.build(chunks=records, creation_time="2026-01-01T00:00:00")
        pack_b = builder.build(chunks=records, creation_time="2026-06-01T00:00:00")

        self.assertNotEqual(pack_a.manifest["manifest_hash"],
                            pack_b.manifest["manifest_hash"])

    def test_written_embeddings_npy_identical_across_two_writes(self):
        """The embeddings.npy bytes must be identical for two writes of the same pack."""
        ragpack = _build_ragpack()
        writer = RagpackWriter()

        with tempfile.TemporaryDirectory() as tmp:
            dir_a = Path(tmp) / "pack_a"
            dir_b = Path(tmp) / "pack_b"
            writer.write(ragpack, dir_a)
            writer.write(ragpack, dir_b)

            bytes_a = (dir_a / "embeddings.npy").read_bytes()
            bytes_b = (dir_b / "embeddings.npy").read_bytes()
            self.assertEqual(bytes_a, bytes_b,
                             "embeddings.npy bytes differ between two writes")

    def test_written_manifest_json_identical_across_two_writes(self):
        """manifest.json content must be identical for two writes of the same pack."""
        ragpack = _build_ragpack()
        writer = RagpackWriter()

        with tempfile.TemporaryDirectory() as tmp:
            dir_a = Path(tmp) / "pack_a"
            dir_b = Path(tmp) / "pack_b"
            writer.write(ragpack, dir_a)
            writer.write(ragpack, dir_b)

            text_a = (dir_a / "manifest.json").read_text(encoding="utf-8")
            text_b = (dir_b / "manifest.json").read_text(encoding="utf-8")
            self.assertEqual(text_a, text_b,
                             "manifest.json content differs between two writes")

    def test_written_chunks_parquet_identical_across_two_writes(self):
        """chunks.parquet bytes must be identical for two writes of the same pack."""
        ragpack = _build_ragpack()
        writer = RagpackWriter()

        with tempfile.TemporaryDirectory() as tmp:
            dir_a = Path(tmp) / "pack_a"
            dir_b = Path(tmp) / "pack_b"
            writer.write(ragpack, dir_a)
            writer.write(ragpack, dir_b)

            bytes_a = (dir_a / "chunks.parquet").read_bytes()
            bytes_b = (dir_b / "chunks.parquet").read_bytes()
            self.assertEqual(bytes_a, bytes_b,
                             "chunks.parquet bytes differ between two writes")


# ---------------------------------------------------------------------------
# test_ragpack_roundtrip
# ---------------------------------------------------------------------------

class TestRagpackRoundtrip(unittest.TestCase):
    """Write to disk → read back → assert fidelity."""

    def _write_and_read(self, ragpack: Ragpack) -> dict:
        """Write pack to a temp dir and read all three artifacts back."""
        writer = RagpackWriter()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "ragpack"
            paths = writer.write(ragpack, out_dir)

            embeddings_back  = np.load(str(paths["embeddings"]))
            manifest_back    = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            chunks_table     = pq.read_table(str(paths["chunks"]))

        return {
            "embeddings": embeddings_back,
            "manifest":   manifest_back,
            "chunks_table": chunks_table,
        }

    def test_embeddings_npy_roundtrip(self):
        ragpack = _build_ragpack()
        result  = self._write_and_read(ragpack)
        np.testing.assert_array_equal(ragpack.embeddings, result["embeddings"],
                                      err_msg="embeddings.npy roundtrip mismatch")

    def test_embeddings_dtype_preserved_after_roundtrip(self):
        ragpack = _build_ragpack()
        result  = self._write_and_read(ragpack)
        self.assertEqual(result["embeddings"].dtype, np.float32)

    def test_manifest_roundtrip(self):
        ragpack = _build_ragpack()
        result  = self._write_and_read(ragpack)
        self.assertEqual(ragpack.manifest, result["manifest"])

    def test_chunks_parquet_row_count(self):
        records = _make_records()
        ragpack = _build_ragpack(records)
        result  = self._write_and_read(ragpack)
        self.assertEqual(result["chunks_table"].num_rows, len(records))

    def test_chunks_parquet_chunk_id_column(self):
        records = _make_records()
        ragpack = _build_ragpack(records)
        result  = self._write_and_read(ragpack)
        table   = result["chunks_table"]

        expected_ids = [r.chunk_id for r in records]
        actual_ids   = table.column("chunk_id").to_pylist()
        self.assertEqual(actual_ids, expected_ids,
                         "chunk_id column order differs after roundtrip")

    def test_chunks_parquet_required_columns_present(self):
        ragpack = _build_ragpack()
        result  = self._write_and_read(ragpack)
        table   = result["chunks_table"]

        required_cols = [
            "chunk_id", "source_id", "source_path", "source_hash",
            "chunk_index", "char_start", "char_end", "token_count",
            "text_snippet", "chunk_text_hash", "chunking_config_hash",
        ]
        for col in required_cols:
            self.assertIn(col, table.schema.names,
                          f"Required column '{col}' missing from chunks.parquet")

    def test_chunks_parquet_source_path_preserved(self):
        records = _make_records()
        ragpack = _build_ragpack(records)
        result  = self._write_and_read(ragpack)
        table   = result["chunks_table"]

        actual_paths = table.column("source_path").to_pylist()
        for path_val in actual_paths:
            self.assertEqual(path_val, _SOURCE_PATH)

    def test_three_files_written(self):
        ragpack = _build_ragpack()
        writer  = RagpackWriter()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "ragpack"
            paths   = writer.write(ragpack, out_dir)

            self.assertIn("embeddings", paths)
            self.assertIn("chunks",     paths)
            self.assertIn("manifest",   paths)
            self.assertTrue(paths["embeddings"].exists())
            self.assertTrue(paths["chunks"].exists())
            self.assertTrue(paths["manifest"].exists())

    def test_output_filenames_match_manifest(self):
        ragpack = _build_ragpack()
        writer  = RagpackWriter()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "ragpack"
            paths   = writer.write(ragpack, out_dir)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

        self.assertEqual(paths["embeddings"].name, manifest["files"]["embeddings"])
        self.assertEqual(paths["chunks"].name,     manifest["files"]["chunks"])
        self.assertEqual(paths["manifest"].name,   manifest["files"]["manifest"])


# ---------------------------------------------------------------------------
# RagpackBuilder construction guard tests
# ---------------------------------------------------------------------------

class TestRagpackBuilderGuards(unittest.TestCase):

    def test_non_embedder_raises_type_error(self):
        with self.assertRaises(TypeError):
            RagpackBuilder("not an embedder")  # type: ignore[arg-type]

    def test_empty_creation_time_raises_value_error(self):
        builder = RagpackBuilder(_get_embedder())
        with self.assertRaises(ValueError):
            builder.build(chunks=_make_records(), creation_time="")

    def test_non_chunk_record_in_list_raises_type_error(self):
        builder = RagpackBuilder(_get_embedder())
        with self.assertRaises(TypeError):
            builder.build(chunks=["not a record"], creation_time=_CREATION_TIME)  # type: ignore[arg-type]

    def test_empty_chunks_builds_valid_ragpack(self):
        """Empty chunk list is valid; manifest chunk_count must be 0."""
        builder = RagpackBuilder(_get_embedder())
        pack = builder.build(chunks=[], creation_time=_CREATION_TIME)
        self.assertEqual(len(pack.chunks), 0)
        self.assertEqual(pack.embeddings.shape[0], 0)
        self.assertEqual(pack.manifest["chunk_count"], 0)


# ---------------------------------------------------------------------------
# RagpackWriter guard tests
# ---------------------------------------------------------------------------

class TestRagpackWriterGuards(unittest.TestCase):

    def test_non_ragpack_raises_type_error(self):
        writer = RagpackWriter()
        with self.assertRaises(TypeError):
            writer.write("not a ragpack", "/tmp/whatever")  # type: ignore[arg-type]

    def test_output_dir_created_if_absent(self):
        ragpack = _build_ragpack()
        writer  = RagpackWriter()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "nested" / "new_dir"
            self.assertFalse(out_dir.exists())
            writer.write(ragpack, out_dir)
            self.assertTrue(out_dir.is_dir())

    def test_existing_file_at_output_dir_raises_value_error(self):
        ragpack = _build_ragpack()
        writer  = RagpackWriter()

        with tempfile.TemporaryDirectory() as tmp:
            conflict = Path(tmp) / "file_not_dir"
            conflict.write_text("oops")
            with self.assertRaises(ValueError):
                writer.write(ragpack, conflict)


if __name__ == "__main__":
    unittest.main()

