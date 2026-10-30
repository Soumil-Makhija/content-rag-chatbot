"""Deterministic orchestration tests with a fake provider, not live AI evidence."""

import math
from pathlib import Path
import tempfile
import unittest

from content_chat.models import ABSTENTION, ModelProtocolError, ModelUnavailable
from content_chat.service import Service
from content_chat.store import Store, chunk_text


class FixtureModel:
    embedding_model = "fixture-embed-v1"
    generation_model = "fixture-generate-v1"

    def __init__(self):
        self.embedding_calls = []
        self.generation_calls = []
        self.embedding_result = None
        self.answer_result = None
        self.embedding_error = None
        self.generation_error = None
        self.on_generate = None

    def embed(self, texts):
        self.embedding_calls.append(list(texts))
        if self.embedding_error:
            raise self.embedding_error
        if self.embedding_result is not None:
            return self.embedding_result
        return [[1.0, 0.0, 0.0] for _text in texts]

    def generate(self, question, sources):
        self.generation_calls.append((question, [dict(source) for source in sources]))
        if self.generation_error:
            raise self.generation_error
        if self.on_generate:
            self.on_generate(sources)
        if self.answer_result is not None:
            return self.answer_result
        return {"supported": True, "answer": sources[0]["text"], "citations": [sources[0]["id"]]}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.model = FixtureModel()
        self.service = Service(self.store, self.model)

    def test_import_embeds_exact_chunks_and_persists_text(self):
        text = "  " + "a" * 2200 + "  "
        metadata = self.service.import_document("support", "FAQ", text, "faq")
        self.assertEqual(self.model.embedding_calls, [chunk_text(text, 1200, 150)])
        self.assertEqual(metadata, {"id": "faq", "title": "FAQ", "version": 1, "chunk_count": 2})
        self.assertEqual(self.store.get_document("support", "faq")["text"], text)
        self.assertEqual(self.store.collection_model("support"), {"model": "fixture-embed-v1", "dimensions": 3})

    def test_large_import_batches_ordered_chunks_and_preserves_vector_mapping(self):
        text = "".join(f"{index:06d} " for index in range(10000))
        chunks = chunk_text(text)
        next_ordinal = 0

        def embed_batch(texts):
            nonlocal next_ordinal
            self.model.embedding_calls.append(list(texts))
            vectors = [[1, ordinal + 1, 2] for ordinal in range(next_ordinal, next_ordinal + len(texts))]
            next_ordinal += len(texts)
            return vectors

        self.model.embed = embed_batch
        metadata = self.service.import_document("support", "Large FAQ", text, "faq")
        self.assertEqual([len(batch) for batch in self.model.embedding_calls], [32, 32, 3])
        self.assertEqual([chunk for batch in self.model.embedding_calls for chunk in batch], chunks)
        self.assertEqual(metadata["chunk_count"], len(chunks))
        hits = self.store.search("support", [1, 0, 0], self.model.embedding_model, limit=100)
        self.assertEqual(len(hits), len(chunks))
        for hit in hits:
            ordinal = hit["ordinal"]
            self.assertEqual(hit["text"], chunks[ordinal])
            self.assertAlmostEqual(hit["score"], 1 / math.sqrt(1 + (ordinal + 1) ** 2 + 4))

    def test_later_embedding_batch_failures_preserve_existing_corpus(self):
        self.service.import_document("support", "FAQ", "Original fact.", "faq")
        before_document = self.store.get_document("support", "faq")
        before_documents = self.store.list_documents("support")
        before_hits = self.store.search("support", [1, 0, 0], self.model.embedding_model)
        text = "Replacement fact. " * 2200
        for failure in ("unavailable", "count", "zero-vector", "mixed-dimensions", "batch-dimensions"):
            for collection, document_id in (("support", "faq"), ("support", "new"), ("fresh", "new")):
                with self.subTest(failure=failure, collection=collection, document_id=document_id):
                    self.model.embedding_calls.clear()

                    def embed_batch(texts):
                        self.model.embedding_calls.append(list(texts))
                        if len(self.model.embedding_calls) == 1:
                            return [[1, 0, 0] for _text in texts]
                        if failure == "unavailable":
                            raise ModelUnavailable("Fixture failed during a later batch")
                        if failure == "count":
                            return []
                        if failure == "zero-vector":
                            return [[0, 0, 0] for _text in texts]
                        if failure == "mixed-dimensions":
                            return [[1, 0, 0]] + [[1, 0] for _text in texts[1:]]
                        return [[1, 0] for _text in texts]

                    self.model.embed = embed_batch
                    error = ModelUnavailable if failure == "unavailable" else ModelProtocolError
                    with self.assertRaises(error):
                        self.service.import_document(collection, "Replacement FAQ", text, document_id)
                    self.assertEqual(len(self.model.embedding_calls), 2)
                    self.assertEqual(self.store.get_document("support", "faq"), before_document)
                    self.assertEqual(self.store.list_documents("support"), before_documents)
                    self.assertEqual(self.store.search("support", [1, 0, 0], self.model.embedding_model), before_hits)
                    self.assertIsNone(self.store.get_document("support", "new"))
                    self.assertIsNone(self.store.collection_model("fresh"))

    def test_first_embedding_batch_checks_collection_dimensions_before_later_calls(self):
        self.service.import_document("support", "FAQ", "Original fact.", "faq")
        before_document = self.store.get_document("support", "faq")
        before_hits = self.store.search("support", [1, 0, 0], self.model.embedding_model)
        self.model.embedding_calls.clear()
        self.model.embedding_result = [[1, 0] for _index in range(32)]
        text = "Replacement fact. " * 2200
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self.service.import_document("support", "Replacement FAQ", text, "faq")
        self.assertEqual(self.model.embedding_calls, [chunk_text(text)[:32]])
        self.assertEqual(self.store.get_document("support", "faq"), before_document)
        self.assertEqual(self.store.search("support", [1, 0, 0], self.model.embedding_model), before_hits)

    def test_answer_returns_complete_resolvable_citation_and_model_names(self):
        self.service.import_document("support", "Shipping", "Atlas ships on Tuesday.", "shipping")
        result = self.service.ask("support", "  When does Atlas ship?  ", limit=1)
        self.assertTrue(result["supported"])
        self.assertEqual(result["answer"], "Atlas ships on Tuesday.")
        self.assertEqual(result["models"], {"embedding": "fixture-embed-v1", "generation": "fixture-generate-v1"})
        citation = result["citations"][0]
        self.assertEqual(citation["collection"], "support")
        self.assertEqual(citation["document_id"], "shipping")
        self.assertEqual(citation["title"], "Shipping")
        self.assertEqual(citation["version"], 1)
        self.assertEqual(citation["ordinal"], 0)
        self.assertAlmostEqual(citation["score"], 1)
        original = self.store.source("support", citation["id"])
        self.assertEqual({key: citation[key] for key in original}, original)
        self.assertEqual(self.model.embedding_calls[-1], ["When does Atlas ship?"])

    def test_empty_corpus_abstains_without_model_access(self):
        self.model.embedding_error = ModelUnavailable("Fixture model unavailable")
        result = self.service.ask("empty", "What is the policy?")
        self.assertEqual(result["answer"], ABSTENTION)
        self.assertFalse(result["supported"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(self.model.embedding_calls, [])
        self.assertEqual(self.model.generation_calls, [])

    def test_unsupported_provider_answer_is_canonical(self):
        self.service.import_document("support", "Shipping", "Atlas ships Tuesday.")
        self.model.answer_result = {"supported": False, "answer": "Untrusted speculative answer", "citations": []}
        result = self.service.ask("support", "Who founded the company?")
        self.assertFalse(result["supported"])
        self.assertEqual(result["answer"], ABSTENTION)
        self.assertEqual(result["citations"], [])

    def test_collections_never_share_retrieved_sources(self):
        self.service.import_document("alpha", "Alpha private", "Alpha secret.")
        self.service.import_document("beta", "Beta private", "Beta secret.")
        answer = self.service.ask("alpha", "What is the secret?")
        self.assertEqual(answer["answer"], "Alpha secret.")
        self.assertTrue(all(source["text"] == "Alpha secret." for source in self.model.generation_calls[-1][1]))
        self.assertIsNone(self.store.source("beta", answer["citations"][0]["id"]))

    def test_unknown_or_cross_collection_citation_is_rejected(self):
        self.service.import_document("alpha", "Alpha", "Alpha.")
        self.service.import_document("beta", "Beta", "Beta.")
        beta_source = self.store.search("beta", [1, 0, 0], self.model.embedding_model)[0]
        self.model.answer_result = {"supported": True, "answer": "Beta.", "citations": [beta_source["id"]]}
        with self.assertRaisesRegex(ModelProtocolError, "not provided"):
            self.service.ask("alpha", "Question?")

    def test_unsupported_with_citation_is_rejected(self):
        self.service.import_document("support", "FAQ", "A fact.")
        source_id = self.store.search("support", [1, 0, 0], self.model.embedding_model)[0]["id"]
        self.model.answer_result = {"supported": False, "answer": "No evidence", "citations": [source_id]}
        with self.assertRaisesRegex(ModelProtocolError, "must not include"):
            self.service.ask("support", "Question?")

    def test_service_revalidates_fake_provider_output(self):
        self.service.import_document("support", "FAQ", "A fact.")
        fixtures = [
            {"supported": True, "answer": "Uncited", "citations": []},
            {"supported": "false", "answer": "Bad boolean", "citations": []},
            {"supported": True, "answer": 10, "citations": []},
            {"supported": False, "answer": "No evidence", "citations": "invalid"},
            {"supported": False, "answer": "No evidence", "citations": [], "extra": True},
            [],
        ]
        for fixture in fixtures:
            self.model.answer_result = fixture
            with self.subTest(fixture=fixture), self.assertRaises(ModelProtocolError):
                self.service.ask("support", "Question?")

    def test_invalid_question_and_limit_fail_before_model_calls(self):
        for question, limit in [(None, 5), ("", 5), ("  ", 5), ("x" * 4001, 5), ("q", 0), ("q", 11),
                                ("q", True), ("q", 1.5), ("q", "5")]:
            with self.subTest(question=question, limit=limit), self.assertRaises(ValueError):
                self.service.ask("support", question, limit)
        self.assertEqual(self.model.embedding_calls, [])

    def test_invalid_imports_fail_before_model_calls(self):
        for collection, title, text, document_id in [
            ("Bad Collection", "FAQ", "Fact", None),
            ("support", " ", "Fact", None),
            ("support", "FAQ", " ", None),
            ("support", "FAQ", None, None),
            ("support", "FAQ", "Fact", ""),
            ("support", "Bad\ntitle", "Fact", None),
            ("support", "FAQ", "Has a \x00 NUL.", None),
            ("support", "FAQ", "Invalid \ud800 Unicode.", None),
            ("support", "Invalid \ud800 title", "Fact", None),
            ("support", "FAQ", "Fact", "invalid-\udfff-id"),
            ("support", "FAQ", "é" * (1024 * 1024 + 1), None),
        ]:
            with self.subTest(collection=collection, title=title, text=text, document_id=document_id), self.assertRaises(ValueError):
                self.service.import_document(collection, title, text, document_id)
        self.assertEqual(self.model.embedding_calls, [])

    def test_empty_search_does_not_need_a_model(self):
        self.model.embedding_error = ModelUnavailable("Fixture model unavailable")
        self.assertEqual(self.service.search("empty", "Question?", limit=100), [])
        self.assertEqual(self.model.embedding_calls, [])
        self.assertEqual(self.model.generation_calls, [])

    def test_search_returns_ranked_passages_without_generation(self):
        self.service.import_document("support", "FAQ", "A fact.")
        sources = self.service.search("support", "Question?", limit=100)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["text"], "A fact.")
        self.assertIn("score", sources[0])
        self.assertEqual(self.model.generation_calls, [])

    def test_search_validates_question_limit_collection_before_model(self):
        for collection, question, limit in [("support", "", 5), ("support", "x" * 4001, 5),
                                             ("support", "q", 101), ("support", "q", True),
                                             ("Invalid Collection", "q", 5)]:
            with self.subTest(collection=collection, question=question, limit=limit), self.assertRaises(ValueError):
                self.service.search(collection, question, limit)
        self.assertEqual(self.model.embedding_calls, [])

    def test_model_signature_mismatch_fails_without_calling_provider(self):
        self.service.import_document("support", "FAQ", "A fact.")
        previous_calls = len(self.model.embedding_calls)
        self.model.embedding_model = "different-embedding-model"
        with self.assertRaisesRegex(ValueError, "Collection uses embedding model"):
            self.service.ask("support", "Question?")
        with self.assertRaisesRegex(ValueError, "Collection uses embedding model"):
            self.service.import_document("support", "New FAQ", "Another fact.")
        self.assertEqual(len(self.model.embedding_calls), previous_calls)

    def test_dimension_change_is_rejected_for_import_and_query(self):
        self.service.import_document("support", "FAQ", "A fact.")
        self.model.embedding_result = [[1, 0]]
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self.service.import_document("support", "New", "New fact.")
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self.service.ask("support", "Question?")
        self.assertEqual(len(self.store.list_documents("support")), 1)
        self.assertEqual(self.model.generation_calls, [])

    def test_malformed_fake_embeddings_are_rejected(self):
        for vectors in ([], [[0, 0, 0]], [[float("nan"), 0, 1]], [[True, 0, 1]], [[1, 2], [3, 4]]):
            self.model.embedding_result = vectors
            with self.subTest(vectors=vectors), self.assertRaises(ModelProtocolError):
                self.service.import_document("support", "FAQ", "A fact.")
        self.assertEqual(self.store.list_documents("support"), [])

    def test_import_provider_failure_preserves_existing_document(self):
        self.service.import_document("support", "FAQ", "Original fact.", "faq")
        self.model.embedding_error = ModelUnavailable("Fixture model unavailable")
        with self.assertRaises(ModelUnavailable):
            self.service.import_document("support", "Updated FAQ", "Replacement fact.", "faq")
        self.assertEqual(self.store.get_document("support", "faq")["text"], "Original fact.")
        self.assertEqual(self.store.get_document("support", "faq")["version"], 1)

    def test_model_failure_is_reported_without_substituting_an_answer(self):
        self.service.import_document("support", "FAQ", "A fact.")
        self.model.generation_error = ModelUnavailable("Fixture model unavailable")
        with self.assertRaises(ModelUnavailable):
            self.service.ask("support", "Question?")

    def test_replacement_and_deletion_change_answerable_corpus(self):
        self.service.import_document("support", "FAQ", "Original fact.", "faq")
        original = self.service.ask("support", "Question?")
        self.service.import_document("support", "Updated FAQ", "Replacement fact.", "faq")
        updated = self.service.ask("support", "Question?")
        self.assertEqual(updated["answer"], "Replacement fact.")
        self.assertEqual(updated["citations"][0]["version"], 2)
        self.assertIsNone(self.store.source("support", original["citations"][0]["id"]))
        self.store.delete("support", "faq")
        calls = len(self.model.embedding_calls)
        self.assertFalse(self.service.ask("support", "Question?")["supported"])
        self.assertEqual(len(self.model.embedding_calls), calls)

    def test_document_changed_during_generation_requires_retry(self):
        self.service.import_document("support", "FAQ", "Original fact.", "faq")
        self.model.on_generate = lambda _sources: self.store.delete("support", "faq")
        with self.assertRaisesRegex(ModelProtocolError, "changed.*again"):
            self.service.ask("support", "Question?")

    def test_uncited_context_changed_during_generation_requires_retry(self):
        for mutation in ("replace", "delete"):
            with self.subTest(mutation=mutation):
                self.service.import_document("support", "Cited FAQ", "A retained fact.", "cited")
                self.service.import_document("support", "Other FAQ", "An old fact.", "other")
                cited = self.store.search("support", [1, 0, 0], self.model.embedding_model)[0]
                self.model.answer_result = {"supported": True, "answer": "A retained fact.", "citations": [cited["id"]]}

                def change_uncited(_sources):
                    if mutation == "delete":
                        self.store.delete("support", "other")
                    else:
                        self.service.import_document("support", "Other FAQ", "A replacement fact.", "other")

                self.model.on_generate = change_uncited
                with self.assertRaisesRegex(ModelProtocolError, "context changed.*again"):
                    self.service.ask("support", "Question?")

    def test_abstention_from_changed_context_requires_retry(self):
        self.service.import_document("support", "FAQ", "An unrelated fact.", "faq")
        self.model.answer_result = {"supported": False, "answer": "No evidence.", "citations": []}
        self.model.on_generate = lambda _sources: self.service.import_document(
            "support", "FAQ", "The requested fact is now available.", "faq"
        )
        with self.assertRaisesRegex(ModelProtocolError, "context changed.*again"):
            self.service.ask("support", "What is the requested fact?")

    def test_changes_in_another_collection_do_not_invalidate_answer(self):
        self.service.import_document("support", "FAQ", "A retained fact.", "faq")
        self.service.import_document("other", "FAQ", "An unrelated fact.", "faq")
        self.model.on_generate = lambda _sources: self.store.delete("other", "faq")
        answer = self.service.ask("support", "Question?")
        self.assertTrue(answer["supported"])
        self.assertEqual(answer["answer"], "A retained fact.")

    def test_generation_can_change_without_reembedding_collection(self):
        self.service.import_document("support", "FAQ", "A fact.")
        self.model.generation_model = "another-generator"
        self.assertEqual(self.service.ask("support", "Question?")["models"]["generation"], "another-generator")

    def test_persistent_index_is_usable_after_restart_with_fixture_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.sqlite3"
            with Store(path) as first_store:
                first = Service(first_store, FixtureModel())
                first.import_document("support", "FAQ", "Persisted fact.", "faq")
                first_id = first.ask("support", "Question?")["citations"][0]["id"]
            model = FixtureModel()
            with Store(path) as reopened_store:
                reopened = Service(reopened_store, model)
                answer = reopened.ask("support", "Question?")
                self.assertEqual(answer["answer"], "Persisted fact.")
                self.assertEqual(answer["citations"][0]["id"], first_id)
                self.assertEqual(model.embedding_calls, [["Question?"]])


if __name__ == "__main__":
    unittest.main()
