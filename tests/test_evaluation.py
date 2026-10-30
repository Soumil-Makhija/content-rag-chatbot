"""Regression tests of evaluation reporting, using an explicit fake model.

These tests do not evaluate a real embedding or generation model and are not
evidence that a model meets the project's acceptance criteria.
"""

import json
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

    def health(self):
        return {"ready": True, "available_models": [self.embedding_model, self.generation_model], "missing_models": []}

    def embed(self, texts):
        # Every fixture source is retrieved; this deliberately does not claim
        # to test semantic retrieval quality.
        return [[1.0, 0.0] for _ in texts]

    def generate(self, question, sources):
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
        report = run_evaluation(EvaluationFixtureModel())
        self.assertEqual(report["status"], "checks_passed_manual_review_required", report)
        self.assertEqual(exit_code(report), 0)
        self.assertEqual(report["checks_passed"], report["checks_total"])
        self.assertEqual(len(report["cases"]), 9)
        self.assertIn("not an accuracy score", report["manual_review_caveat"])
        self.assertEqual(report["models"]["generation"], "fixture-generation-not-a-model")
        for case in report["cases"]:
            self.assertIn("elapsed_seconds", case)
            self.assertIn("models", case["actual"])
        self.assertTrue(all(check["passed"] for check in report["lifecycle_checks"]))

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
