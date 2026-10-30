"""Regression tests of evaluation reporting, using an explicit fake model.

These tests do not evaluate a real embedding or generation model and are not
evidence that a model meets the project's acceptance criteria.
"""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from content_chat.models import ABSTENTION, ModelUnavailable
from tools.evaluate_live import check_answer, exit_code, main, run_evaluation


class EvaluationFixtureModel:
    """Deterministic harness fixture; no model inference or network calls."""

    base_url = "fixture://evaluation-regression-only"
    embedding_model = "fixture-embedding-not-a-model"
    generation_model = "fixture-generation-not-a-model"

    def __init__(self):
        self.generation_source_counts = []

    def health(self):
        return {"ready": True, "available_models": [self.embedding_model, self.generation_model], "missing_models": []}

    def embed(self, texts):
        # Hand-coded topic coordinates regress ranking/check accounting only.
        # They are explicitly not learned embeddings or semantic-quality evidence.
        vectors = []
        for text in texts:
            value = text.casefold()
            vector = [float(any(term in value for term in terms)) for terms in (
                ("trial", "export my workspace", "export your workspace"),
                ("refund",),
                ("two-factor authentication",),
                ("notification preferences",),
                ("keyboard shortcuts",),
                ("file attachments",),
                ("accessibility preferences",),
                ("archiving projects",),
            )]
            if not any(vector):
                vector[2] = 1.0
            vectors.append(vector)
        return vectors

    def generate(self, question, sources):
        self.generation_source_counts.append(len(sources))
        text = " ".join(source["text"] for source in sources)
        refund_sources = [source for source in sources if "refund request window" in source["text"]]
        support_sources = [source for source in sources if "trial lasts 14 days" in source["text"]]
        if "business days" in question or "Saturn" in question:
            return {"supported": False, "answer": ABSTENTION, "citations": []}
        if "trial last" in question:
            return {"supported": True, "answer": "The trial lasts 14 days.", "citations": [source["id"] for source in support_sources]}
        if "refund request window" in question and refund_sources:
            days = "45" if "now 45 days" in text else "90" if "90-day" in text else "30"
            answer = f"The refund request window is {days} days."
            selected = refund_sources
            if "export my workspace" in question:
                answer += " Open Settings and choose Export, then JSON or Markdown."
                selected += support_sources
            return {"supported": True, "answer": answer, "citations": [source["id"] for source in selected]}
        return {"supported": False, "answer": ABSTENTION, "citations": []}


class EvaluationHarnessTests(unittest.TestCase):
    def test_fixture_regresses_checks_and_lifecycle_without_claiming_accuracy(self):
        model = EvaluationFixtureModel()
        report = run_evaluation(model)
        self.assertEqual(report["status"], "checks_passed_manual_review_required", report)
        self.assertEqual(exit_code(report), 0)
        self.assertEqual(report["checks_passed"], report["checks_total"])
        self.assertEqual(len(report["cases"]), 9)
        self.assertEqual(len(report["retrieval_cases"]), 3)
        self.assertEqual(sum(document["chunk_count"] for document in report["imported_documents"]
                             if document["collection"] == "support"), 8)
        self.assertIn("not an accuracy score", report["manual_review_caveat"])
        self.assertEqual(report["models"]["generation"], "fixture-generation-not-a-model")
        for case in report["cases"]:
            self.assertIn("elapsed_seconds", case)
            self.assertIn("models", case["actual"])
            self.assertEqual(case["limit"], 2)
        self.assertEqual(len(model.generation_source_counts), 9)
        self.assertTrue(all(count <= 2 for count in model.generation_source_counts))
        for case in report["retrieval_cases"]:
            self.assertEqual(case["corpus_chunk_count"], 8)
            self.assertGreater(case["corpus_chunk_count"], case["limit"])
            self.assertIn("elapsed_seconds", case)
            self.assertTrue(all(check["passed"] for check in case["checks"]))
            self.assertTrue(all(isinstance(source["score"], float) for source in case["retrieved"]))
            self.assertTrue(all(source["text"] for source in case["retrieved"]))
        multi = next(case for case in report["retrieval_cases"] if case["id"] == "retrieve-multi-document")
        self.assertEqual({source["document_id"] for source in multi["retrieved"]}, {"support", "billing"})
        self.assertEqual([source["score"] for source in multi["retrieved"]], [1 / math.sqrt(2)] * 2)
        all_checks = [check for case in report["retrieval_cases"] + report["cases"] for check in case["checks"]]
        self.assertEqual(report["checks_total"], len(all_checks) + len(report["lifecycle_checks"]))
        self.assertTrue(all(check["passed"] for check in report["lifecycle_checks"]))

    def test_bad_embeddings_fail_selective_retrieval_despite_expected_trial_phrase(self):
        class ConstantEmbeddingModel(EvaluationFixtureModel):
            def embed(self, texts):
                return [[1.0] * 8 for _ in texts]

            def generate(self, question, sources):
                if "trial last" in question:
                    # Deliberately supply a fact without relevant evidence. The
                    # lexical check alone passes; retrieval must still fail.
                    return {"supported": True, "answer": "The trial lasts 14 days.", "citations": [sources[0]["id"]]}
                return super().generate(question, sources)

        report = run_evaluation(ConstantEmbeddingModel())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(exit_code(report), 1)
        retrieval = next(case for case in report["retrieval_cases"] if case["id"] == "retrieve-trial")
        required = next(check for check in retrieval["checks"] if check["name"] == "required-document:support")
        self.assertFalse(required["passed"])
        trial = next(case for case in report["cases"] if case["id"] == "trial")
        self.assertTrue(next(check for check in trial["checks"] if check["name"].startswith("answer-term:"))["passed"])

    def test_retrieval_checks_fail_when_corpus_fits_inside_top_k(self):
        source = Path(__file__).resolve().parents[1] / "examples" / "evaluation.json"
        manifest = json.loads(source.read_text())
        manifest["documents"] = [document for document in manifest["documents"]
                                 if document["id"] in {"support", "billing"}]
        # An absolute source path keeps the copied manifest independent of its directory.
        for document in manifest["documents"]:
            if "file" in document:
                document["file"] = str(source.parent / document["file"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps(manifest))
            report = run_evaluation(EvaluationFixtureModel(), path)
        self.assertEqual(report["status"], "failed")
        multi = next(case for case in report["retrieval_cases"] if case["id"] == "retrieve-multi-document")
        self.assertFalse(next(check for check in multi["checks"] if check["name"] == "corpus-exceeds-limit")["passed"])
        self.assertTrue(all(check["passed"] for case in report["cases"] for check in case["checks"]))

    def test_wrong_model_answer_fails_lexical_checks_instead_of_counting_as_quality(self):
        class WrongAnswerModel(EvaluationFixtureModel):
            def generate(self, question, sources):
                answer = super().generate(question, sources)
                if "trial last" in question:
                    answer["answer"] = "The trial lasts 130 days."
                return answer

        report = run_evaluation(WrongAnswerModel())
        self.assertEqual(report["status"], "failed")
        trial = next(case for case in report["cases"] if case["id"] == "trial")
        self.assertFalse(next(check for check in trial["checks"] if check["name"].startswith("answer-term:"))["passed"])
        self.assertEqual(exit_code(report), 1)

    def test_missing_models_stop_before_import_or_generation(self):
        class MissingModel(EvaluationFixtureModel):
            def health(self):
                return {"ready": False, "missing_models": [self.generation_model]}

            def embed(self, texts):
                raise AssertionError("Import must not run without models.")

        report = run_evaluation(MissingModel())
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["cases"], [])
        self.assertEqual(report["retrieval_cases"], [])
        self.assertIn("fixture-generation", report["error"]["message"])
        self.assertEqual(exit_code(report), 2)

    def test_unavailable_runtime_writes_actionable_json_report(self):
        model = EvaluationFixtureModel()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports" / "evaluation.json"
            with patch("tools.evaluate_live.Ollama", return_value=model), patch.object(
                model, "health", side_effect=ModelUnavailable("No local model service is running.")
            ):
                self.assertEqual(main(["--output", str(output)]), 2)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["cases"], [])
            self.assertEqual(report["retrieval_cases"], [])
            self.assertEqual(report["error"]["type"], "ModelUnavailable")
            self.assertIn("No local model service", report["error"]["message"])

    def test_invalid_configuration_still_writes_failed_json_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
            self.assertEqual(main(["--output", str(output), "--ollama-url", "https://example.com"]), 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"]["type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
