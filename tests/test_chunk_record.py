"""
Stage 1 contract tests: deterministic chunk_id, complete metadata, repeatability.

Tests in this file validate the EPIC3 Stage 1 deliverables:
- test_chunk_id_deterministic  — identical inputs always produce the same chunk_id
- test_chunk_metadata_complete — every required field is present and correctly typed
- test_chunk_repeatability     — two independent chunking runs produce identical records
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chunker import (
    ChunkRecord,
    TokenChunker,
    compute_chunk_id,
    build_snippet,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SOURCE_ID = "src-abc123"
_SOURCE_PATH = "docs/sample.txt"
_SOURCE_HASH = "a" * 64  # placeholder SHA-256 hex string
_TEXT = (
    "Artificial intelligence is transforming every industry. "
    "Machine learning models learn from data to make predictions. "
    "Natural language processing enables computers to understand human text. "
    "Deep learning uses many-layered neural networks for complex tasks. "
) * 4  # repeat to ensure multiple chunks with small chunk sizes


def _make_chunker(**overrides) -> TokenChunker:
    defaults = dict(chunk_size=20, overlap=5, preserve_sentences=False)
    defaults.update(overrides)
    return TokenChunker(**defaults)


# ---------------------------------------------------------------------------
# chunk_id helpers
# ---------------------------------------------------------------------------

class TestComputeChunkId(unittest.TestCase):
    """Unit tests for the compute_chunk_id function."""

    def _make_id(self, **overrides) -> str:
        kwargs = dict(
            source_id=_SOURCE_ID,
            char_start=0,
            char_end=100,
            chunk_text_hash="b" * 64,
            chunking_config_hash="c" * 64,
        )
        kwargs.update(overrides)
        return compute_chunk_id(**kwargs)

    def test_returns_64_char_hex_string(self):
        """chunk_id should be a 64-character lowercase hex string (SHA-256)."""
        chunk_id = self._make_id()
        self.assertEqual(len(chunk_id), 64)
        self.assertTrue(all(ch in "0123456789abcdef" for ch in chunk_id))

    def test_same_inputs_same_output(self):
        """Identical inputs must always produce the same chunk_id."""
        id_a = self._make_id()
        id_b = self._make_id()
        self.assertEqual(id_a, id_b)

    def test_different_source_id_different_output(self):
        """A different source_id must change the chunk_id."""
        id_a = self._make_id(source_id="src-aaa")
        id_b = self._make_id(source_id="src-bbb")
        self.assertNotEqual(id_a, id_b)

    def test_different_char_start_different_output(self):
        """A different char_start must change the chunk_id."""
        id_a = self._make_id(char_start=0)
        id_b = self._make_id(char_start=1)
        self.assertNotEqual(id_a, id_b)

    def test_different_char_end_different_output(self):
        """A different char_end must change the chunk_id."""
        id_a = self._make_id(char_end=100)
        id_b = self._make_id(char_end=101)
        self.assertNotEqual(id_a, id_b)

    def test_different_chunk_text_hash_different_output(self):
        """A different chunk_text_hash must change the chunk_id."""
        id_a = self._make_id(chunk_text_hash="b" * 64)
        id_b = self._make_id(chunk_text_hash="c" * 64)
        self.assertNotEqual(id_a, id_b)

    def test_different_config_hash_different_output(self):
        """A different chunking_config_hash must change the chunk_id."""
        id_a = self._make_id(chunking_config_hash="c" * 64)
        id_b = self._make_id(chunking_config_hash="d" * 64)
        self.assertNotEqual(id_a, id_b)

    def test_empty_source_id_raises(self):
        with self.assertRaises(ValueError):
            self._make_id(source_id="")

    def test_negative_char_start_raises(self):
        with self.assertRaises(ValueError):
            self._make_id(char_start=-1)

    def test_char_end_not_greater_than_start_raises(self):
        with self.assertRaises(ValueError):
            self._make_id(char_start=10, char_end=10)


# ---------------------------------------------------------------------------
# test_chunk_id_deterministic
# ---------------------------------------------------------------------------

class TestChunkIdDeterministic(unittest.TestCase):
    """chunk_id must be deterministic: same source + config → same ids."""

    def test_chunk_id_deterministic_single_run(self):
        """
        Two calls to chunk_document with identical inputs must produce
        chunk records with identical chunk_ids in the same order.
        """
        chunker = _make_chunker()
        records_a = chunker.chunk_document(
            text=_TEXT,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )
        records_b = chunker.chunk_document(
            text=_TEXT,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )

        self.assertGreater(len(records_a), 0, "Expected at least one chunk")
        self.assertEqual(len(records_a), len(records_b))
        for rec_a, rec_b in zip(records_a, records_b):
            self.assertEqual(rec_a.chunk_id, rec_b.chunk_id)

    def test_chunk_id_changes_when_source_id_changes(self):
        """Changing source_id must produce different chunk_ids."""
        chunker = _make_chunker()
        records_a = chunker.chunk_document(
            text=_TEXT,
            source_id="src-001",
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )
        records_b = chunker.chunk_document(
            text=_TEXT,
            source_id="src-002",
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )

        self.assertEqual(len(records_a), len(records_b))
        for rec_a, rec_b in zip(records_a, records_b):
            self.assertNotEqual(rec_a.chunk_id, rec_b.chunk_id)

    def test_chunk_ids_are_unique_within_document(self):
        """Each chunk in a document must have a distinct chunk_id."""
        chunker = _make_chunker()
        records = chunker.chunk_document(
            text=_TEXT,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )
        self.assertGreater(len(records), 1, "Need multiple chunks for this test")
        ids = [r.chunk_id for r in records]
        self.assertEqual(len(ids), len(set(ids)), "chunk_ids must be unique")

    def test_chunk_id_is_64_char_hex(self):
        """chunk_id must be a 64-character lowercase SHA-256 hex string."""
        chunker = _make_chunker()
        records = chunker.chunk_document(
            text=_TEXT,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )
        for record in records:
            self.assertEqual(len(record.chunk_id), 64)
            self.assertTrue(
                all(ch in "0123456789abcdef" for ch in record.chunk_id),
                f"Non-hex character in chunk_id: {record.chunk_id}",
            )

    def test_chunk_id_does_not_change_with_new_chunker_instance(self):
        """
        A new TokenChunker with identical config must produce the same
        chunk_ids as a prior instance — config_hash must be stable.
        """
        chunker_1 = _make_chunker()
        chunker_2 = _make_chunker()

        records_1 = chunker_1.chunk_document(
            text=_TEXT,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )
        records_2 = chunker_2.chunk_document(
            text=_TEXT,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )

        self.assertEqual(len(records_1), len(records_2))
        for r1, r2 in zip(records_1, records_2):
            self.assertEqual(r1.chunk_id, r2.chunk_id)


# ---------------------------------------------------------------------------
# test_chunk_metadata_complete
# ---------------------------------------------------------------------------

class TestChunkMetadataComplete(unittest.TestCase):
    """Every ChunkRecord must carry the complete EPIC3 Stage 1 field set."""

    REQUIRED_FIELDS = {
        "chunk_id",
        "source_id",
        "source_path",
        "source_hash",
        "chunk_index",
        "char_start",
        "char_end",
        "token_count",
        "text_snippet",
        "chunk_text_hash",
        "chunking_config_hash",
    }

    def _get_records(self) -> list:
        chunker = _make_chunker()
        return chunker.chunk_document(
            text=_TEXT,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )

    def test_all_required_fields_present(self):
        """to_dict() must include every required field."""
        for record in self._get_records():
            record_dict = record.to_dict()
            for field_name in self.REQUIRED_FIELDS:
                self.assertIn(
                    field_name,
                    record_dict,
                    f"Required field '{field_name}' missing from chunk record",
                )

    def test_source_id_matches_input(self):
        for record in self._get_records():
            self.assertEqual(record.source_id, _SOURCE_ID)

    def test_source_path_matches_input(self):
        for record in self._get_records():
            self.assertEqual(record.source_path, _SOURCE_PATH)

    def test_source_hash_matches_input(self):
        for record in self._get_records():
            self.assertEqual(record.source_hash, _SOURCE_HASH)

    def test_chunk_index_sequential_zero_based(self):
        """chunk_index must be a zero-based, monotonically increasing integer."""
        records = self._get_records()
        for expected_index, record in enumerate(records):
            self.assertEqual(record.chunk_index, expected_index)

    def test_char_start_and_end_are_valid_offsets(self):
        """char_start must be non-negative, char_end must exceed char_start."""
        for record in self._get_records():
            self.assertGreaterEqual(record.char_start, 0)
            self.assertGreater(record.char_end, record.char_start)

    def test_token_count_is_positive_int(self):
        for record in self._get_records():
            self.assertIsInstance(record.token_count, int)
            self.assertGreater(record.token_count, 0)

    def test_text_snippet_is_string_and_not_empty(self):
        for record in self._get_records():
            self.assertIsInstance(record.text_snippet, str)
            self.assertGreater(len(record.text_snippet), 0)

    def test_text_snippet_max_length(self):
        """text_snippet must not exceed 203 characters (200 + '...')."""
        for record in self._get_records():
            self.assertLessEqual(len(record.text_snippet), 203)

    def test_chunk_text_hash_is_64_char_hex(self):
        for record in self._get_records():
            self.assertEqual(len(record.chunk_text_hash), 64)
            self.assertTrue(
                all(ch in "0123456789abcdef" for ch in record.chunk_text_hash)
            )

    def test_chunking_config_hash_is_64_char_hex(self):
        for record in self._get_records():
            self.assertEqual(len(record.chunking_config_hash), 64)
            self.assertTrue(
                all(ch in "0123456789abcdef" for ch in record.chunking_config_hash)
            )

    def test_chunking_config_hash_same_for_all_chunks(self):
        """All chunks from the same chunker instance share the config hash."""
        records = self._get_records()
        hashes = {r.chunking_config_hash for r in records}
        self.assertEqual(len(hashes), 1, "All chunks must share the same config hash")

    def test_optional_section_defaults_to_none(self):
        for record in self._get_records():
            self.assertIsNone(record.section)

    def test_optional_page_defaults_to_none(self):
        for record in self._get_records():
            self.assertIsNone(record.page)

    def test_section_and_page_propagated_when_supplied(self):
        """section and page are forwarded to every chunk when supplied."""
        chunker = _make_chunker()
        records = chunker.chunk_document(
            text=_TEXT,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
            section="Introduction",
            page=3,
        )
        for record in records:
            self.assertEqual(record.section, "Introduction")
            self.assertEqual(record.page, 3)

    def test_chunk_id_is_not_integer_sequence(self):
        """chunk_id must be a hash string, never a plain integer index."""
        for record in self._get_records():
            self.assertIsInstance(record.chunk_id, str)
            # A SHA-256 hex string contains letters so int() will always raise
            with self.assertRaises(ValueError):
                int(record.chunk_id, 10)  # base-10 parse fails on hex letters

    def test_to_dict_serialises_all_required_fields(self):
        """to_dict() output must include every required field as a key."""
        for record in self._get_records():
            d = record.to_dict()
            for field_name in self.REQUIRED_FIELDS:
                self.assertIn(field_name, d)

    def test_get_chunker_metadata_includes_config_hash(self):
        """get_chunker_metadata() must expose 'config_hash' for manifest use."""
        chunker = _make_chunker()
        metadata = chunker.get_chunker_metadata()
        self.assertIn("config_hash", metadata)
        self.assertEqual(len(metadata["config_hash"]), 64)


# ---------------------------------------------------------------------------
# test_chunk_repeatability
# ---------------------------------------------------------------------------

class TestChunkRepeatability(unittest.TestCase):
    """
    Two independent chunking runs with identical inputs must produce
    byte-for-byte identical ChunkRecord objects.
    """

    def _run(self, text: str, **chunker_kwargs) -> list:
        chunker = TokenChunker(**chunker_kwargs)
        return chunker.chunk_document(
            text=text,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )

    def test_identical_records_on_repeated_run(self):
        """Two runs must produce identical records in order."""
        kwargs = dict(chunk_size=20, overlap=5, preserve_sentences=False)
        run_a = self._run(_TEXT, **kwargs)
        run_b = self._run(_TEXT, **kwargs)

        self.assertEqual(len(run_a), len(run_b))
        for rec_a, rec_b in zip(run_a, run_b):
            self.assertEqual(rec_a.to_dict(), rec_b.to_dict())

    def test_no_stray_nondeterminism_in_single_chunk_case(self):
        """A text shorter than chunk_size must always produce one stable record."""
        short_text = "A short sentence."
        kwargs = dict(chunk_size=100, overlap=0)
        run_a = self._run(short_text, **kwargs)
        run_b = self._run(short_text, **kwargs)

        self.assertEqual(len(run_a), 1)
        self.assertEqual(run_a[0].to_dict(), run_b[0].to_dict())

    def test_config_hash_stable_across_instances(self):
        """config_hash must be identical for two TokenChunker instances with same config."""
        chunker_a = TokenChunker(chunk_size=64, overlap=8, preserve_sentences=True)
        chunker_b = TokenChunker(chunk_size=64, overlap=8, preserve_sentences=True)
        self.assertEqual(chunker_a._config_hash, chunker_b._config_hash)

    def test_different_config_produces_different_config_hash(self):
        """Different chunking config must produce different config_hash."""
        chunker_a = TokenChunker(chunk_size=64, overlap=8)
        chunker_b = TokenChunker(chunk_size=128, overlap=8)
        self.assertNotEqual(chunker_a._config_hash, chunker_b._config_hash)

    def test_chunk_text_hash_matches_manual_computation(self):
        """chunk_text_hash must equal sha256(chunk_text, utf-8)."""
        import hashlib

        chunker = _make_chunker()
        records = chunker.chunk_document(
            text=_TEXT,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
        )
        for record in records:
            # Retrieve the original chunk text via the char offsets.
            # Do NOT strip _TEXT here: the chunker strips internally and
            # records offsets relative to the stripped text, so slicing
            # the un-stripped source with those offsets gives the wrong
            # characters. Use _TEXT directly so offsets are consistent.
            original_text = _TEXT
            chunk_text = original_text[record.char_start: record.char_end]
            expected_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            self.assertEqual(
                record.chunk_text_hash,
                expected_hash,
                f"chunk_text_hash mismatch for chunk_index={record.chunk_index}",
            )

    def test_repeatability_with_unicode_text(self):
        """Unicode source text must produce stable records across runs."""
        unicode_text = (
            "Привет мир. " * 8
            + "日本語テスト。" * 8
            + "こんにちは 🌍 " * 8
        )
        kwargs = dict(chunk_size=15, overlap=3, preserve_sentences=False)
        run_a = self._run(unicode_text, **kwargs)
        run_b = self._run(unicode_text, **kwargs)

        self.assertEqual(len(run_a), len(run_b))
        for rec_a, rec_b in zip(run_a, run_b):
            self.assertEqual(rec_a.to_dict(), rec_b.to_dict())

    def test_empty_text_produces_empty_list(self):
        """Empty / whitespace-only input must produce an empty record list."""
        chunker = _make_chunker()
        for text in ("", "   ", "\n\t"):
            records = chunker.chunk_document(
                text=text,
                source_id=_SOURCE_ID,
                source_path=_SOURCE_PATH,
                source_hash=_SOURCE_HASH,
            )
            self.assertEqual(records, [])


# ---------------------------------------------------------------------------
# ChunkRecord dataclass invariant tests
# ---------------------------------------------------------------------------

class TestChunkRecordInvariants(unittest.TestCase):
    """ChunkRecord must enforce its field invariants at construction time."""

    def _valid_kwargs(self) -> dict:
        return dict(
            chunk_id="a" * 64,
            source_id=_SOURCE_ID,
            source_path=_SOURCE_PATH,
            source_hash=_SOURCE_HASH,
            chunk_index=0,
            char_start=0,
            char_end=50,
            token_count=10,
            text_snippet="Hello world",
            chunk_text_hash="b" * 64,
            chunking_config_hash="c" * 64,
        )

    def test_valid_record_constructs_without_error(self):
        record = ChunkRecord(**self._valid_kwargs())
        self.assertIsInstance(record, ChunkRecord)

    def test_empty_chunk_id_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["chunk_id"] = ""
        with self.assertRaises(ValueError):
            ChunkRecord(**kwargs)

    def test_negative_chunk_index_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["chunk_index"] = -1
        with self.assertRaises(ValueError):
            ChunkRecord(**kwargs)

    def test_negative_char_start_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["char_start"] = -1
        with self.assertRaises(ValueError):
            ChunkRecord(**kwargs)

    def test_char_end_equal_to_char_start_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["char_start"] = 10
        kwargs["char_end"] = 10
        with self.assertRaises(ValueError):
            ChunkRecord(**kwargs)

    def test_char_end_less_than_char_start_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["char_start"] = 10
        kwargs["char_end"] = 5
        with self.assertRaises(ValueError):
            ChunkRecord(**kwargs)

    def test_zero_token_count_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["token_count"] = 0
        with self.assertRaises(ValueError):
            ChunkRecord(**kwargs)

    def test_optional_section_and_page_default_to_none(self):
        record = ChunkRecord(**self._valid_kwargs())
        self.assertIsNone(record.section)
        self.assertIsNone(record.page)

    def test_section_and_page_can_be_set(self):
        kwargs = self._valid_kwargs()
        kwargs["section"] = "Background"
        kwargs["page"] = 7
        record = ChunkRecord(**kwargs)
        self.assertEqual(record.section, "Background")
        self.assertEqual(record.page, 7)


# ---------------------------------------------------------------------------
# build_snippet helper tests
# ---------------------------------------------------------------------------

class TestBuildSnippet(unittest.TestCase):

    def test_short_text_returned_unchanged(self):
        text = "Short text."
        self.assertEqual(build_snippet(text), text)

    def test_exactly_200_chars_returned_unchanged(self):
        text = "x" * 200
        self.assertEqual(build_snippet(text), text)

    def test_201_chars_truncated_with_ellipsis(self):
        text = "x" * 201
        result = build_snippet(text)
        self.assertEqual(result, "x" * 200 + "...")
        self.assertEqual(len(result), 203)


if __name__ == "__main__":
    unittest.main()

