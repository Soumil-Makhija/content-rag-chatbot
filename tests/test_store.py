import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from content_chat.store import Store, chunk_text


class ChunkTests(unittest.TestCase):
    def test_character_windows_overlap_without_splitting_utf8(self):
        self.assertEqual(chunk_text("  café🙂東京abcdef  ", size=6, overlap=2), ["café🙂東", "🙂東京abc", "bcdef"])

    def test_exact_boundary_does_not_add_duplicate_overlap_chunk(self):
        self.assertEqual(chunk_text("abcdef", 6, 3), ["abcdef"])
        self.assertEqual(chunk_text("abcdefg", 6, 3), ["abcdef", "defg"])
        self.assertEqual(chunk_text("\n  \t"), [])

    def test_blank_interior_windows_are_skipped_without_changing_source_text(self):
        text = "first" + " \n" * 1600 + "last"
        chunks = chunk_text(text)
        self.assertEqual(chunks, [text[:1200], text[2100:]])
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_invalid_limits_and_input(self):
        for size, overlap in [(0, 0), (-1, 0), (3, -1), (3, 3), (3, 4), (True, 0), (3, False), (1.5, 0)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                chunk_text("text", size, overlap)
        with self.assertRaises(ValueError):
            chunk_text(None)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "vectors.sqlite3"
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def put(self, document_id="guide", text="Our return policy lasts thirty days.", vector=(1.0, 0.0), collection="support", model="test-embedding-v1"):
        return self.store.put(collection, document_id, text, [vector] * len(chunk_text(text)), model, document_id)

    def test_cosine_ranking_is_scale_invariant_and_ties_are_deterministic(self):
        self.put("d", vector=(-1, 0))
        self.put("b", vector=(3, 4))
        self.put("a", vector=(300, 400))
        self.put("c", vector=(0, 8))
        hits = self.store.search("support", [100, 0], "test-embedding-v1")
        self.assertEqual([hit["document_id"] for hit in hits], ["a", "b", "c", "d"])
        for hit, expected in zip(hits, [0.6, 0.6, 0, -1]):
            self.assertAlmostEqual(hit["score"], expected)
        self.assertEqual(len(self.store.search("support", [1, 0], "test-embedding-v1", limit=2)), 2)

    def test_extreme_finite_vectors_are_normalized_without_overflow(self):
        self.put(vector=[1e308, 1e308])
        hit = self.store.search("support", [1e-308, 1e-308], "test-embedding-v1")[0]
        self.assertAlmostEqual(hit["score"], 1.0)

    def test_skipped_blank_windows_keep_embeddings_and_source_ordinals_aligned(self):
        text = "first" + " \n" * 1600 + "last"
        document = self.store.put("support", "Whitespace", text, [[1, 0], [0, 1]], "test-embedding-v1", "guide")
        self.assertEqual(document["chunk_count"], 2)
        self.assertEqual(self.store.get_document("support", "guide")["text"], text)
        hits = self.store.search("support", [0, 1], "test-embedding-v1")
        self.assertEqual([hit["ordinal"] for hit in hits], [1, 0])
        self.assertEqual([hit["text"] for hit in hits], list(reversed(chunk_text(text))))
        self.assertAlmostEqual(hits[0]["score"], 1)
        self.assertAlmostEqual(hits[1]["score"], 0)

    def test_documents_vectors_and_citation_provenance_survive_restart(self):
        text = "# Règlement\nLes échanges sont possibles pendant trente jours. 東京🙂"
        document = self.put(text=text)
        hit = self.store.search("support", [1, 0], "test-embedding-v1")[0]
        self.store.close()
        self.store = Store(self.path)
        self.assertEqual(self.store.list_documents("support"), [document])
        self.assertEqual(self.store.get_document("support", document["id"]), {**document, "text": text})
        self.assertEqual(self.store.search("support", [1, 0], "test-embedding-v1"), [hit])
        source = self.store.source("support", hit["id"])
        self.assertEqual(source, {key: value for key, value in hit.items() if key != "score"})
        self.assertEqual(source["text"], text)
        self.assertEqual(self.store.collection_model("support"), {"model": "test-embedding-v1", "dimensions": 2})

    def test_update_replaces_every_chunk_and_invalidates_old_citations(self):
        original = self.put(text="Old answer. " * 250)
        self.assertGreater(original["chunk_count"], 1)
        old_hits = self.store.search("support", [1, 0], "test-embedding-v1")
        updated = self.put(text="New answer: exchanges end after fourteen days.", vector=[0, 1])
        self.assertEqual(updated["id"], original["id"])
        self.assertEqual(updated["version"], 2)
        self.assertEqual(updated["chunk_count"], 1)
        hits = self.store.search("support", [1, 0], "test-embedding-v1")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["text"], "New answer: exchanges end after fourteen days.")
        self.assertEqual(hits[0]["version"], 2)
        self.assertAlmostEqual(hits[0]["score"], 0)
        for hit in old_hits:
            self.assertIsNone(self.store.source("support", hit["id"]))

    def test_delete_removes_document_vectors_and_citations(self):
        self.put()
        source_id = self.store.search("support", [1, 0], "test-embedding-v1")[0]["id"]
        self.assertTrue(self.store.delete("support", "guide"))
        self.assertFalse(self.store.delete("support", "guide"))
        self.assertEqual(self.store.list_documents("support"), [])
        self.assertIsNone(self.store.get_document("support", "guide"))
        self.assertIsNone(self.store.source("support", source_id))
        self.assertEqual(self.store.search("support", [1, 0], "test-embedding-v1"), [])
        self.assertIsNone(self.store.collection_model("support"))

    def test_same_document_ids_and_sources_are_isolated_by_collection(self):
        first = self.put(text="Public facts.")
        self.put(text="Private facts.", vector=[0, 1, 0], collection="private", model="another-model")
        source_id = self.store.search("support", [1, 0], "test-embedding-v1")[0]["id"]
        self.assertEqual(self.store.list_documents("support"), [first])
        self.assertEqual(self.store.get_document("support", "guide")["text"], "Public facts.")
        self.assertIsNone(self.store.source("private", source_id))
        self.assertEqual(self.store.search("missing", [1, 0], "test-embedding-v1"), [])
        self.assertEqual(len(self.store.search("private", [0, 1, 0], "another-model")), 1)
        self.store.delete("support", "guide")
        self.assertIsNotNone(self.store.get_document("private", "guide"))

    def test_model_signature_is_enforced_until_collection_is_empty(self):
        self.put("one")
        self.put("two")
        for model, vector in [("different-model", [1, 0]), ("test-embedding-v1", [1, 0, 0])]:
            with self.subTest(model=model, vector=vector):
                with self.assertRaisesRegex(ValueError, "model or dimensions"):
                    self.put("three", model=model, vector=vector)
                with self.assertRaisesRegex(ValueError, "model or dimensions"):
                    self.store.search("support", vector, model)
        self.store.delete("support", "one")
        self.assertIsNotNone(self.store.collection_model("support"))
        self.store.delete("support", "two")
        self.put("new", vector=[1, 2, 3], model="new-model")
        self.assertEqual(self.store.collection_model("support"), {"model": "new-model", "dimensions": 3})

    def test_search_signature_and_chunks_use_one_snapshot_during_model_replacement(self):
        self.put(text="Old model corpus.")
        connection = self.store.connection
        with Store(self.path) as writer:
            class ReplaceBeforeChunkRead:
                """Deterministically schedule a writer just before the vector SELECT."""

                switched = False

                def execute(proxy, statement, parameters=()):
                    if not proxy.switched and "SELECT" in statement and "chunks c" in statement:
                        proxy.switched = True
                        writer.delete("support", "guide")
                        writer.put("support", "New model", "New model corpus.", [[1, 0, 1]], "new-model", "guide")
                    return connection.execute(statement, parameters)

            proxy = ReplaceBeforeChunkRead()
            self.store.connection = proxy
            try:
                with self.assertRaisesRegex(ValueError, "model or dimensions"):
                    self.store.search("support", [1, 0], "test-embedding-v1")
                self.assertTrue(proxy.switched)
            finally:
                self.store.connection = connection
        hits = self.store.search("support", [1, 0, 1], "new-model")
        self.assertEqual(hits[0]["text"], "New model corpus.")
        self.assertAlmostEqual(hits[0]["score"], 1)

    def test_bad_vectors_cannot_create_or_change_a_document(self):
        self.put()
        before = self.store.get_document("support", "guide")
        before_hits = self.store.search("support", [1, 0], "test-embedding-v1")
        invalid = [[], [0, 0], [math.nan, 1], [math.inf, 1], [-math.inf, 1], ["1", "2"], [True, False], None, "12"]
        for vector in invalid:
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    self.put(text="A malformed replacement.", vector=vector)
                self.assertEqual(self.store.get_document("support", "guide"), before)
                self.assertEqual(self.store.search("support", [1, 0], "test-embedding-v1"), before_hits)
                with self.assertRaises(ValueError):
                    self.put(collection="fresh", vector=vector)
                self.assertIsNone(self.store.collection_model("fresh"))

    def test_bad_chunk_count_or_inconsistent_dimensions_does_not_mutate(self):
        self.put()
        before = self.store.get_document("support", "guide")
        for text, embeddings in [("replacement", []), ("replacement", [[1, 0], [1, 0]]), ("x" * 1500, [[1, 0], [1, 0, 0]])]:
            with self.subTest(text_length=len(text)), self.assertRaises(ValueError):
                self.store.put("support", "guide", text, embeddings, "test-embedding-v1", "guide")
            self.assertEqual(self.store.get_document("support", "guide"), before)

    def test_mid_write_database_failure_rolls_back_whole_replacement(self):
        self.put(text="Original answer.")
        before = self.store.get_document("support", "guide")
        before_hits = self.store.search("support", [1, 0], "test-embedding-v1")
        self.store.connection.execute("""CREATE TRIGGER reject_chunk BEFORE INSERT ON chunks
            BEGIN SELECT RAISE(ABORT, 'simulated disk constraint'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.put(text="New answer.")
        self.assertEqual(self.store.get_document("support", "guide"), before)
        self.assertEqual(self.store.search("support", [1, 0], "test-embedding-v1"), before_hits)
        self.store.connection.execute("DROP TRIGGER reject_chunk")
        self.assertEqual(self.put(text="Successful retry.")["version"], 2)

    def test_invalid_names_blank_documents_and_limits_are_rejected(self):
        for collection in ["", "Support", "../support", "two words", "1start", "a" * 65, None]:
            with self.subTest(collection=collection), self.assertRaises(ValueError):
                self.put(collection=collection)
        for text in ["", " \n ", None]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.store.put("support", "Title", text, [], "model")
        self.put()
        for limit in [0, -1, 101, True, 1.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.store.search("support", [1, 0], "test-embedding-v1", limit)
        self.assertIsNone(self.store.get_document("support", "absent"))
        self.assertIsNone(self.store.source("support", "absent"))


if __name__ == "__main__":
    unittest.main()
