"""
Tests for the RAGpack v1.2 nested manifest builder (build_manifest_v1_2).

The flat EPIC3 ManifestBuilder is covered by tests/test_ragpack_builder.py.
This file covers the additive v1.2 app-facing builder added for ADR-0011 §5.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ragpack.manifest_builder import build_manifest_v1_2


_CREATED_AT = "2026-03-07T00:00:00"


def _embedder_block(**overrides) -> dict:
    block = {
        "embedding_model": "nomic-embed-text-v1.5.Q5_K_M.gguf",
        "embedding_version": "0.3.28",
        "embedding_dimension": 768,
        "model_hash": "0c7930f6c4f6f29b7da5046e3a2c0832aa3f602db3de5760a95f0582dbd3d6e6",
        "dtype": "float32",
        "pooling": "mean",
        "l2_normalized": True,
        "runtime": "llama.cpp",
    }
    block.update(overrides)
    return block


def _chunker_block() -> dict:
    return {
        "method": "token_based",
        "chunk_size": 512,
        "overlap": 50,
        "tokenizer_name": "gpt2",
        "preserve_sentences": False,
        "config_hash": "a" * 64,
    }


def _build(**overrides) -> dict:
    kwargs = dict(
        pack_id="pack-test",
        created_at=_CREATED_AT,
        chunker=_chunker_block(),
        embedder=_embedder_block(),
        indexer={"document_count": 1, "chunk_count": 3, "timestamp": _CREATED_AT},
    )
    kwargs.update(overrides)
    return build_manifest_v1_2(**kwargs)


class TestBuildManifestV12(unittest.TestCase):

    def test_pack_version_is_1_2(self):
        self.assertEqual(_build()["pack_version"], "1.2")

    def test_embedder_pooling_is_mean(self):
        self.assertEqual(_build()["embedder"]["pooling"], "mean")

    def test_embedder_l2_normalized_true(self):
        self.assertIs(_build()["embedder"]["l2_normalized"], True)

    def test_embedder_dtype_float32(self):
        self.assertEqual(_build()["embedder"]["dtype"], "float32")

    def test_embedder_model_hash_preserved(self):
        self.assertEqual(
            _build()["embedder"]["model_hash"],
            "0c7930f6c4f6f29b7da5046e3a2c0832aa3f602db3de5760a95f0582dbd3d6e6",
        )

    def test_embedder_runtime_llamacpp(self):
        self.assertEqual(_build()["embedder"]["runtime"], "llama.cpp")

    def test_nested_blocks_present(self):
        m = _build()
        for key in ("pack_version", "pack_id", "created_at", "chunker",
                    "embedder", "indexer", "files"):
            self.assertIn(key, m)

    def test_default_files_block(self):
        files = _build()["files"]
        self.assertEqual(files["chunks"], "chunks.json")
        self.assertEqual(files["embeddings"], "embeddings.npy")
        self.assertEqual(files["citations"], "citations.jsonl")

    def test_created_at_preserved(self):
        self.assertEqual(_build()["created_at"], _CREATED_AT)

    # --- validation guards ---------------------------------------------------

    def test_missing_pooling_raises(self):
        block = _embedder_block()
        del block["pooling"]
        with self.assertRaises(ValueError):
            _build(embedder=block)

    def test_wrong_dtype_raises(self):
        with self.assertRaises(ValueError):
            _build(embedder=_embedder_block(dtype="float64"))

    def test_non_mean_pooling_raises(self):
        with self.assertRaises(ValueError):
            _build(embedder=_embedder_block(pooling="cls"))

    def test_l2_false_raises(self):
        with self.assertRaises(ValueError):
            _build(embedder=_embedder_block(l2_normalized=False))

    def test_empty_pack_id_raises(self):
        with self.assertRaises(ValueError):
            _build(pack_id="")

    def test_empty_created_at_raises(self):
        with self.assertRaises(ValueError):
            _build(created_at="")


if __name__ == "__main__":
    unittest.main()
