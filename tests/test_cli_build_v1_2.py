"""
End-to-end v1.2 pack-build test (ADR-0011 §5).

Runs the llama.cpp / PackWriter pipeline (run_pipeline_v12) and asserts the
written pack matches what the NoesisNoema app validates: a v1.2 nested manifest
with the GGUF file-hash identity, mean pooling, L2 normalization, float32, and
citations keyed to the app's RAGpackReader spec.

Requires a real GGUF via NOESIS_EMBEDDER_GGUF; skipped otherwise.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cli.build_ragpack import run_pipeline_v12

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

_GGUF_PATH = os.environ.get("NOESIS_EMBEDDER_GGUF")
_GGUF_AVAILABLE = bool(_GGUF_PATH) and os.path.isfile(_GGUF_PATH or "")
_SKIP_REASON = "NOESIS_EMBEDDER_GGUF not set; skipping v1.2 e2e pack build."

_CREATION_TIME = "2026-03-07T00:00:00"
_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schemas",
                            "manifest_v1_2.json")

_DOC_A = (
    "Substance is that which is in itself and is conceived through itself.\n\n"
    "By attribute I understand that which the intellect perceives of substance.\n\n"
    "God is a being absolutely infinite, a substance consisting of infinite "
    "attributes, each of which expresses eternal and infinite essence.\n"
)
_DOC_B = (
    "The order and connection of ideas is the same as the order and connection "
    "of things.\n\nThe human mind is part of the infinite intellect of God.\n"
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@unittest.skipUnless(_GGUF_AVAILABLE, _SKIP_REASON)
class TestCliBuildV12(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls._input = root / "input"
        cls._input.mkdir()
        (cls._input / "ethica_a.txt").write_text(_DOC_A, encoding="utf-8")
        (cls._input / "ethica_b.txt").write_text(_DOC_B, encoding="utf-8")
        cls._out = root / "pack"
        cls._result = run_pipeline_v12(
            input_dir=cls._input,
            output_dir=cls._out,
            gguf_path=Path(_GGUF_PATH),
            chunk_size=64,
            overlap=8,
            creation_time=_CREATION_TIME,
        )
        cls._manifest = json.loads((cls._out / "manifest.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_all_artifacts_written(self):
        for name in ("manifest.json", "embeddings.npy", "chunks.json",
                     "citations.jsonl"):
            self.assertTrue((self._out / name).is_file(), f"{name} missing")

    def test_manifest_pack_version_1_2(self):
        self.assertEqual(self._manifest["pack_version"], "1.2")

    def test_embedder_block_identity(self):
        emb = self._manifest["embedder"]
        self.assertEqual(emb["dtype"], "float32")
        self.assertEqual(emb["pooling"], "mean")
        self.assertIs(emb["l2_normalized"], True)
        self.assertEqual(emb["embedding_dimension"], 768)
        self.assertEqual(emb["model_hash"].lower(), _sha256_file(_GGUF_PATH).lower())

    def test_embeddings_float32_unit_norm(self):
        arr = np.load(str(self._out / "embeddings.npy"))
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(arr.shape[1], 768)
        norms = np.linalg.norm(arr, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_embeddings_row_count_matches_chunks(self):
        arr = np.load(str(self._out / "embeddings.npy"))
        self.assertEqual(arr.shape[0], self._result.chunk_count)

    def test_citations_shape_matches_app_spec(self):
        lines = (self._out / "citations.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(lines), self._result.chunk_count)
        for i, line in enumerate(lines):
            cit = json.loads(line)
            for key in ("chunk_index", "doc_id", "char_start", "char_end",
                        "paragraph_boundaries"):
                self.assertIn(key, cit, f"citation missing '{key}'")
            self.assertEqual(cit["chunk_index"], i)
            self.assertIsInstance(cit["paragraph_boundaries"], list)
            self.assertNotIn("start_char", cit)  # normalized away
            self.assertNotIn("end_char", cit)

    def test_doc_id_defaults_to_filename(self):
        lines = (self._out / "citations.jsonl").read_text().strip().splitlines()
        doc_ids = {json.loads(l)["doc_id"] for l in lines}
        self.assertTrue(doc_ids.issubset({"ethica_a.txt", "ethica_b.txt"}))

    @unittest.skipUnless(_HAS_JSONSCHEMA, "jsonschema not installed")
    def test_manifest_validates_against_schema(self):
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=self._manifest, schema=schema)

    def test_pack_id_deterministic_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out2 = Path(tmp) / "pack2"
            result2 = run_pipeline_v12(
                input_dir=self._input,
                output_dir=out2,
                gguf_path=Path(_GGUF_PATH),
                chunk_size=64,
                overlap=8,
                creation_time=_CREATION_TIME,
            )
            m2 = json.loads((out2 / "manifest.json").read_text())
            self.assertEqual(self._manifest["pack_id"], m2["pack_id"])
            bytes_a = (self._out / "embeddings.npy").read_bytes()
            bytes_b = (out2 / "embeddings.npy").read_bytes()
            self.assertEqual(bytes_a, bytes_b)


if __name__ == "__main__":
    unittest.main()
