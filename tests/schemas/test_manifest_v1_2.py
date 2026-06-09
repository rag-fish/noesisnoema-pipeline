"""
Validate a built v1.2 manifest against schemas/manifest_v1_2.json.

Uses jsonschema if available; skips otherwise so the suite stays green in a
minimal environment.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ragpack.manifest_builder import build_manifest_v1_2

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_SCHEMA_PATH = os.path.join(_REPO_ROOT, "schemas", "manifest_v1_2.json")


def _load_schema() -> dict:
    with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_manifest(**embedder_overrides) -> dict:
    embedder = {
        "embedding_model": "nomic-embed-text-v1.5.Q5_K_M.gguf",
        "embedding_version": "0.3.28",
        "embedding_dimension": 768,
        "model_hash": "0c7930f6c4f6f29b7da5046e3a2c0832aa3f602db3de5760a95f0582dbd3d6e6",
        "dtype": "float32",
        "pooling": "mean",
        "l2_normalized": True,
        "runtime": "llama.cpp",
        # legacy aliases also emitted by EmbedderMetadata.to_dict()
        "name": "nomic-embed-text-v1.5.Q5_K_M.gguf",
        "version": "0.3.28",
        "dimensions": 768,
    }
    embedder.update(embedder_overrides)
    return build_manifest_v1_2(
        pack_id="pack-schema-test",
        created_at="2026-03-07T00:00:00",
        chunker={
            "method": "token_based",
            "chunk_size": 512,
            "overlap": 50,
            "tokenizer_name": "gpt2",
            "preserve_sentences": False,
            "config_hash": "b" * 64,
        },
        embedder=embedder,
        indexer={"document_count": 1, "chunk_count": 3,
                 "timestamp": "2026-03-07T00:00:00"},
        files={"chunks": "chunks.json", "embeddings": "embeddings.npy",
               "citations": "citations.jsonl"},
        source_documents=[{"doc_id": "ethica.txt", "title": "ethica"}],
    )


class TestManifestV12Schema(unittest.TestCase):

    def test_schema_file_is_valid_json(self):
        schema = _load_schema()
        self.assertEqual(schema["title"], "RAGpack Manifest v1.2")
        self.assertEqual(schema["properties"]["pack_version"]["const"], "1.2")

    @unittest.skipUnless(_HAS_JSONSCHEMA, "jsonschema not installed")
    def test_built_manifest_validates(self):
        jsonschema.validate(instance=_sample_manifest(), schema=_load_schema())

    @unittest.skipUnless(_HAS_JSONSCHEMA, "jsonschema not installed")
    def test_schema_rejects_wrong_pack_version(self):
        manifest = _sample_manifest()
        manifest["pack_version"] = "1.1"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=manifest, schema=_load_schema())

    @unittest.skipUnless(_HAS_JSONSCHEMA, "jsonschema not installed")
    def test_schema_requires_pooling_in_embedder(self):
        manifest = _sample_manifest()
        del manifest["embedder"]["pooling"]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=manifest, schema=_load_schema())

    @unittest.skipUnless(_HAS_JSONSCHEMA, "jsonschema not installed")
    def test_schema_rejects_non_float32_dtype(self):
        # build_manifest_v1_2 guards dtype, so inject post-build to test the schema.
        manifest = _sample_manifest()
        manifest["embedder"]["dtype"] = "float16"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=manifest, schema=_load_schema())


if __name__ == "__main__":
    unittest.main()
