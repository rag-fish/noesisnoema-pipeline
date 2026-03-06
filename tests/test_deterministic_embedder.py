"""
Stage 2 contract tests: deterministic embedding generation.

Tests in this file validate the EPIC3 Stage 2 deliverables:
- test_embedding_repeatability        — same inputs → byte-identical float32 arrays
- test_embedding_dimension_consistent — every row has the declared dimension
- test_embedding_order_matches_chunks — output rows align with input chunk order
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chunker import TokenChunker
from embedder import DeterministicEmbedder, EmbedderMetadata
from embedder.deterministic_embedder import EmbeddingResult, DEFAULT_MODEL_NAME


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SOURCE_ID   = "src-stage2-test"
_SOURCE_PATH = "tests/fixtures/sample.txt"
_SOURCE_HASH = "b" * 64

_TEXT = (
    "Artificial intelligence is transforming every industry. "
    "Machine learning models learn from data to make predictions. "
    "Natural language processing enables computers to understand human text. "
    "Deep learning uses many-layered neural networks for complex tasks. "
) * 3


def _make_records(text: str = _TEXT, chunk_size: int = 40, overlap: int = 5):
    """Return a list of ChunkRecord objects using standard test config."""
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


# One shared embedder instance reused across tests to avoid loading the model
# multiple times per test run.  It is created lazily so import-time errors
# are surfaced clearly.
_SHARED_EMBEDDER: DeterministicEmbedder | None = None


def _get_embedder() -> DeterministicEmbedder:
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = DeterministicEmbedder(DEFAULT_MODEL_NAME)
    return _SHARED_EMBEDDER


# ---------------------------------------------------------------------------
# EmbedderMetadata unit tests
# ---------------------------------------------------------------------------

class TestEmbedderMetadata(unittest.TestCase):
    """EmbedderMetadata must carry all required fields and serialise correctly."""

    def _meta(self) -> EmbedderMetadata:
        return _get_embedder().metadata

    def test_embedding_model_is_non_empty_string(self):
        self.assertIsInstance(self._meta().embedding_model, str)
        self.assertGreater(len(self._meta().embedding_model), 0)

    def test_embedding_version_is_non_empty_string(self):
        self.assertIsInstance(self._meta().embedding_version, str)
        self.assertGreater(len(self._meta().embedding_version), 0)

    def test_embedding_dimension_is_positive_int(self):
        dim = self._meta().embedding_dimension
        self.assertIsInstance(dim, int)
        self.assertGreater(dim, 0)

    def test_model_hash_is_64_char_hex(self):
        h = self._meta().model_hash
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_dtype_is_float32(self):
        self.assertEqual(self._meta().dtype, "float32")

    def test_to_dict_contains_required_manifest_keys(self):
        d = self._meta().to_dict()
        required = {
            "embedding_model",
            "embedding_version",
            "embedding_dimension",
            "model_hash",
            "dtype",
        }
        for key in required:
            self.assertIn(key, d, f"Missing key '{key}' in EmbedderMetadata.to_dict()")

    def test_to_dict_contains_legacy_keys_for_pack_writer(self):
        """Legacy keys must be present so existing PackWriter stays compatible."""
        d = self._meta().to_dict()
        for legacy_key in ("name", "version", "dimensions"):
            self.assertIn(legacy_key, d, f"Missing legacy key '{legacy_key}'")

    def test_metadata_is_immutable(self):
        """EmbedderMetadata is a frozen dataclass; normal attribute assignment must raise."""
        meta = self._meta()
        raised = False
        try:
            # Use the normal assignment path that frozen dataclasses block.
            # object.__setattr__ is an internal bypass and must not be used here.
            setattr(meta, "embedding_model", "should-fail")
        except (AttributeError, TypeError):
            raised = True
        self.assertTrue(raised, "Expected frozen dataclass to reject attribute mutation")

    def test_model_hash_stable_across_instances(self):
        """Two embedders with the same model must produce the same model_hash."""
        embedder_a = DeterministicEmbedder(DEFAULT_MODEL_NAME)
        embedder_b = DeterministicEmbedder(DEFAULT_MODEL_NAME)
        self.assertEqual(embedder_a.metadata.model_hash, embedder_b.metadata.model_hash)


# ---------------------------------------------------------------------------
# test_embedding_repeatability
# ---------------------------------------------------------------------------

class TestEmbeddingRepeatability(unittest.TestCase):
    """
    Embedding the same chunks twice must produce byte-identical float32 arrays.
    """

    def test_two_calls_produce_identical_arrays(self):
        """Core repeatability requirement: same input → same output."""
        embedder = _get_embedder()
        records = _make_records()

        result_a = embedder.embed_chunks(records)
        result_b = embedder.embed_chunks(records)

        np.testing.assert_array_equal(
            result_a.embeddings,
            result_b.embeddings,
            err_msg="Embeddings differ between two calls with identical input",
        )

    def test_two_independent_embedder_instances_produce_identical_arrays(self):
        """
        A freshly constructed DeterministicEmbedder must produce the same
        embeddings as a previously constructed one with the same model.
        """
        records = _make_records()

        embedder_1 = DeterministicEmbedder(DEFAULT_MODEL_NAME)
        embedder_2 = DeterministicEmbedder(DEFAULT_MODEL_NAME)

        result_1 = embedder_1.embed_chunks(records)
        result_2 = embedder_2.embed_chunks(records)

        np.testing.assert_array_equal(
            result_1.embeddings,
            result_2.embeddings,
            err_msg="Embeddings differ between two independent embedder instances",
        )

    def test_dtype_is_always_float32(self):
        """Output array must always be float32 regardless of model internals."""
        result = _get_embedder().embed_chunks(_make_records())
        self.assertEqual(result.embeddings.dtype, np.float32)

    def test_empty_chunk_list_returns_zero_row_array(self):
        """embed_chunks([]) must return a valid (0, D) float32 array."""
        result = _get_embedder().embed_chunks([])
        self.assertEqual(result.embeddings.ndim, 2)
        self.assertEqual(result.embeddings.shape[0], 0)
        self.assertEqual(result.embeddings.shape[1], _get_embedder().metadata.embedding_dimension)
        self.assertEqual(result.embeddings.dtype, np.float32)
        self.assertEqual(result.chunk_ids, [])

    def test_single_chunk_repeatable(self):
        """A single-chunk input must be repeatable."""
        records = _make_records(text="A single sentence for repeatability.")
        self.assertGreaterEqual(len(records), 1)
        single = records[:1]

        result_a = _get_embedder().embed_chunks(single)
        result_b = _get_embedder().embed_chunks(single)

        np.testing.assert_array_equal(result_a.embeddings, result_b.embeddings)

    def test_embed_texts_matches_embed_chunks_for_same_text(self):
        """
        embed_texts() called with the same snippet strings must produce
        the same vectors as embed_chunks() for the same records.
        """
        embedder = _get_embedder()
        records = _make_records()

        chunk_result = embedder.embed_chunks(records)
        texts = [r.text_snippet for r in records]
        text_result = embedder.embed_texts(texts)

        np.testing.assert_array_equal(
            chunk_result.embeddings,
            text_result,
            err_msg="embed_chunks and embed_texts produced different vectors for the same text",
        )

    def test_different_texts_produce_different_embeddings(self):
        """Sanity check: semantically different sentences must not be identical."""
        embedder = _get_embedder()
        arr = embedder.embed_texts([
            "The sky is blue.",
            "Quantum mechanics describes subatomic particles.",
        ])
        self.assertFalse(
            np.array_equal(arr[0], arr[1]),
            "Different texts produced identical embedding vectors",
        )


# ---------------------------------------------------------------------------
# test_embedding_dimension_consistent
# ---------------------------------------------------------------------------

class TestEmbeddingDimensionConsistent(unittest.TestCase):
    """
    Every embedding row must have exactly the declared dimension D.
    The declared dimension must match the model's reported dimension.
    """

    def test_embedding_shape_is_n_by_d(self):
        """Output shape must be (N, D) where N = len(chunks) and D = declared dim."""
        embedder = _get_embedder()
        records = _make_records()
        result = embedder.embed_chunks(records)

        expected_d = embedder.metadata.embedding_dimension
        self.assertEqual(result.embeddings.ndim, 2)
        self.assertEqual(result.embeddings.shape[0], len(records))
        self.assertEqual(result.embeddings.shape[1], expected_d)

    def test_dimension_matches_metadata(self):
        """Actual column count must equal metadata.embedding_dimension."""
        embedder = _get_embedder()
        result = embedder.embed_chunks(_make_records())
        self.assertEqual(
            result.embeddings.shape[1],
            result.metadata.embedding_dimension,
        )

    def test_dimension_consistent_across_variable_length_inputs(self):
        """D must be the same whether we embed 1 or many chunks."""
        embedder = _get_embedder()
        declared_dim = embedder.metadata.embedding_dimension

        for n_chunks in (1, 3, 7):
            records = _make_records()[:n_chunks] if len(_make_records()) >= n_chunks else _make_records()
            result = embedder.embed_chunks(records)
            self.assertEqual(
                result.embeddings.shape[1],
                declared_dim,
                f"Dimension mismatch for input size {n_chunks}",
            )

    def test_all_miniLM_dimension_is_384(self):
        """all-MiniLM-L6-v2 must always produce 384-dimensional vectors."""
        self.assertEqual(_get_embedder().metadata.embedding_dimension, 384)

    def test_no_nan_or_inf_in_embeddings(self):
        """Embedding values must all be finite."""
        result = _get_embedder().embed_chunks(_make_records())
        self.assertTrue(
            np.all(np.isfinite(result.embeddings)),
            "Embedding array contains NaN or Inf values",
        )

    def test_embed_texts_returns_correct_shape(self):
        """embed_texts must also return (N, D) with the correct D."""
        embedder = _get_embedder()
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        arr = embedder.embed_texts(texts)
        self.assertEqual(arr.shape, (3, embedder.metadata.embedding_dimension))
        self.assertEqual(arr.dtype, np.float32)

    def test_embed_texts_empty_returns_zero_row_array(self):
        embedder = _get_embedder()
        arr = embedder.embed_texts([])
        self.assertEqual(arr.shape, (0, embedder.metadata.embedding_dimension))
        self.assertEqual(arr.dtype, np.float32)

    def test_result_metadata_dimension_equals_embedder_metadata(self):
        """EmbeddingResult.metadata must be the same object as embedder.metadata."""
        embedder = _get_embedder()
        result = embedder.embed_chunks(_make_records())
        self.assertIs(result.metadata, embedder.metadata)


# ---------------------------------------------------------------------------
# test_embedding_order_matches_chunks
# ---------------------------------------------------------------------------

class TestEmbeddingOrderMatchesChunks(unittest.TestCase):
    """
    Row i in the embedding array must correspond to chunk i in the input list.
    The chunk_ids list in EmbeddingResult must be aligned in the same order.
    """

    def test_chunk_ids_align_with_input_order(self):
        """chunk_ids in result must match the order of the input records."""
        embedder = _get_embedder()
        records = _make_records()
        result = embedder.embed_chunks(records)

        self.assertEqual(len(result.chunk_ids), len(records))
        for idx, (chunk_id, record) in enumerate(zip(result.chunk_ids, records)):
            self.assertEqual(
                chunk_id,
                record.chunk_id,
                f"chunk_ids[{idx}] does not match records[{idx}].chunk_id",
            )

    def test_row_count_equals_chunk_count(self):
        """Number of embedding rows must equal number of input chunks."""
        embedder = _get_embedder()
        records = _make_records()
        result = embedder.embed_chunks(records)
        self.assertEqual(result.embeddings.shape[0], len(records))

    def test_reversed_input_produces_reversed_output(self):
        """
        If the input order is reversed, the embedding rows must be reversed.
        This validates strict positional alignment.
        """
        embedder = _get_embedder()
        records = _make_records()
        self.assertGreater(len(records), 1, "Need at least 2 chunks for this test")

        result_fwd = embedder.embed_chunks(records)
        result_rev = embedder.embed_chunks(list(reversed(records)))

        # Row i of result_rev should equal row (N-1-i) of result_fwd
        n = len(records)
        for i in range(n):
            np.testing.assert_array_equal(
                result_rev.embeddings[i],
                result_fwd.embeddings[n - 1 - i],
                err_msg=f"Row {i} of reversed result does not match row {n-1-i} of forward result",
            )

    def test_chunk_ids_reversed_when_input_reversed(self):
        """chunk_ids in result must also reverse when input order reverses."""
        embedder = _get_embedder()
        records = _make_records()
        self.assertGreater(len(records), 1)

        result_fwd = embedder.embed_chunks(records)
        result_rev = embedder.embed_chunks(list(reversed(records)))

        self.assertEqual(result_fwd.chunk_ids, list(reversed(result_rev.chunk_ids)))

    def test_subset_embedding_matches_corresponding_rows_of_full_embedding(self):
        """
        Embedding a subset of chunks must produce vectors close to the
        corresponding rows extracted from embedding the full list.

        Note on tolerance: transformer models are not strictly associative
        across different batch sizes — float32 accumulation order changes
        with batch padding, producing differences up to ~1e-6 in practice.
        We use allclose(atol=1e-5) which is tight enough to detect real
        ordering bugs while accepting unavoidable float32 rounding.
        The key guarantee tested here is ordering, not bitwise identity
        across batch compositions.
        """
        embedder = _get_embedder()
        records = _make_records()
        self.assertGreater(len(records), 2, "Need at least 3 chunks for this test")

        result_full = embedder.embed_chunks(records)

        # Pick a contiguous subset
        subset = records[1:3]
        result_sub = embedder.embed_chunks(subset)

        np.testing.assert_allclose(
            result_sub.embeddings,
            result_full.embeddings[1:3],
            atol=1e-5,
            rtol=0,
            err_msg=(
                "Subset embeddings deviate from corresponding full-batch rows "
                "by more than the expected float32 tolerance (1e-5)"
            ),
        )

    def test_embedding_result_invariants_hold(self):
        """EmbeddingResult post-init must reject shape mismatches."""
        meta = _get_embedder().metadata
        dim = meta.embedding_dimension

        # Wrong dtype
        with self.assertRaises(ValueError):
            EmbeddingResult(
                embeddings=np.zeros((2, dim), dtype=np.float64),
                metadata=meta,
                chunk_ids=["a", "b"],
            )

        # Row count / chunk_ids length mismatch
        with self.assertRaises(ValueError):
            EmbeddingResult(
                embeddings=np.zeros((2, dim), dtype=np.float32),
                metadata=meta,
                chunk_ids=["only-one"],
            )

        # Wrong column count
        with self.assertRaises(ValueError):
            EmbeddingResult(
                embeddings=np.zeros((2, dim + 1), dtype=np.float32),
                metadata=meta,
                chunk_ids=["a", "b"],
            )


# ---------------------------------------------------------------------------
# DeterministicEmbedder construction guard tests
# ---------------------------------------------------------------------------

class TestDeterministicEmbedderConstruction(unittest.TestCase):

    def test_empty_model_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            DeterministicEmbedder("")

    def test_whitespace_model_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            DeterministicEmbedder("   ")

    def test_invalid_model_name_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            DeterministicEmbedder("this-model-does-not-exist-at-all-xyz-abc-999")

    def test_non_chunk_record_raises_type_error(self):
        """embed_chunks must reject non-ChunkRecord inputs."""
        embedder = _get_embedder()
        with self.assertRaises(TypeError):
            embedder.embed_chunks(["not a ChunkRecord"])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

