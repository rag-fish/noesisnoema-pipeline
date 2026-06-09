"""
Contract tests for LlamaCppEmbedder (RAGpack v1.2 / ADR-0011 §5).

These tests require a real embedder GGUF.  The path is taken from the
NOESIS_EMBEDDER_GGUF environment variable; if it is unset or the file is
missing, the model-dependent tests are skipped (so CI without the GGUF stays
green).  The pure-metadata identity test runs whenever the file exists.
"""

import hashlib
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from chunker import TokenChunker
from embedder.llamacpp_embedder import (
    DOCUMENT_TASK_PREFIX,
    NOMIC_EMBED_DIMENSION,
    LlamaCppEmbedder,
)


_GGUF_PATH = os.environ.get("NOESIS_EMBEDDER_GGUF")
_GGUF_AVAILABLE = bool(_GGUF_PATH) and os.path.isfile(_GGUF_PATH or "")
_SKIP_REASON = (
    "NOESIS_EMBEDDER_GGUF is not set or does not point to a file; "
    "skipping llama.cpp model tests."
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# A single shared embedder so the GGUF is loaded only once for the suite.
_EMBEDDER: "LlamaCppEmbedder | None" = None


def _get_embedder() -> LlamaCppEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = LlamaCppEmbedder(_GGUF_PATH)
    return _EMBEDDER


# ---------------------------------------------------------------------------
# Construction guards (no GGUF required)
# ---------------------------------------------------------------------------

class TestLlamaCppEmbedderGuards(unittest.TestCase):

    def test_empty_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            LlamaCppEmbedder("")

    def test_missing_file_raises_value_error(self):
        with self.assertRaises(ValueError):
            LlamaCppEmbedder("/no/such/model/file/xyz.gguf")


# ---------------------------------------------------------------------------
# Smoke / determinism / metadata (GGUF required)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_GGUF_AVAILABLE, _SKIP_REASON)
class TestLlamaCppEmbedderModel(unittest.TestCase):

    def test_smoke_hello_world_shape_and_norm(self):
        vecs = _get_embedder().embed_texts(["hello world"])
        self.assertEqual(vecs.shape, (1, NOMIC_EMBED_DIMENSION))
        self.assertEqual(vecs.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vecs[0])), 1.0, places=5)

    def test_determinism_byte_identical(self):
        emb = _get_embedder()
        a = emb.embed_texts(["determinism check sentence"])
        b = emb.embed_texts(["determinism check sentence"])
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_metadata_model_hash_is_gguf_file_hash(self):
        meta = _get_embedder().metadata
        self.assertEqual(meta.model_hash.lower(), _sha256_file(_GGUF_PATH).lower())

    def test_metadata_v1_2_embedder_fields(self):
        meta = _get_embedder().metadata
        self.assertEqual(meta.dtype, "float32")
        self.assertEqual(meta.pooling, "mean")
        self.assertTrue(meta.l2_normalized)
        self.assertEqual(meta.runtime, "llama.cpp")
        self.assertEqual(meta.embedding_dimension, NOMIC_EMBED_DIMENSION)

    def test_to_dict_carries_required_v1_2_keys(self):
        d = _get_embedder().metadata.to_dict()
        for key in ("embedding_model", "embedding_version", "embedding_dimension",
                    "model_hash", "dtype", "pooling", "l2_normalized", "runtime"):
            self.assertIn(key, d)

    def test_empty_texts_returns_zero_row_array(self):
        arr = _get_embedder().embed_texts([])
        self.assertEqual(arr.shape, (0, NOMIC_EMBED_DIMENSION))
        self.assertEqual(arr.dtype, np.float32)

    def test_order_independent_of_batch_position(self):
        """An identical text embeds to the same vector regardless of neighbours."""
        emb = _get_embedder()
        out = emb.embed_texts(["the sky is blue", "a wholly different sentence",
                               "the sky is blue"])
        np.testing.assert_array_equal(out[0], out[2])

    def test_different_texts_differ(self):
        emb = _get_embedder()
        out = emb.embed_texts(["the sky is blue",
                               "quantum mechanics describes subatomic particles"])
        self.assertFalse(np.array_equal(out[0], out[1]))

    def test_embed_chunks_aligns_with_records(self):
        records = TokenChunker(chunk_size=40, overlap=5, preserve_sentences=False) \
            .chunk_document(
                text=("Artificial intelligence is transforming every industry. "
                      "Machine learning models learn from data. ") * 3,
                source_id="src-llamacpp-test",
                source_path="tests/fixtures/sample.txt",
                source_hash="d" * 64,
            )
        self.assertGreater(len(records), 0)
        result = _get_embedder().embed_chunks(records)
        self.assertEqual(result.embeddings.shape[0], len(records))
        self.assertEqual(result.embeddings.shape[1], NOMIC_EMBED_DIMENSION)
        self.assertEqual(result.embeddings.dtype, np.float32)
        self.assertEqual(result.chunk_ids, [r.chunk_id for r in records])

    def test_task_prefix_constant(self):
        # Guards the contract: document texts use the search_document prefix.
        self.assertEqual(DOCUMENT_TASK_PREFIX, "search_document: ")


if __name__ == "__main__":
    unittest.main()
