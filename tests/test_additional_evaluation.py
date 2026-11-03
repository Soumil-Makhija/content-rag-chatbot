"""Reporting regressions with explicit fixture inference, never model quality."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from content_chat.models import ABSTENTION, ModelProtocolError, ModelUnavailable
from tools.evaluate_additional import CASES, main, run_evaluation
from tools.evaluate_live import exit_code


class AlwaysAbstainsFixture:
    """Intentionally inadequate fixture; no learned embeddings or network calls."""

    base_url = "fixture://additional-reporting-only"
    embedding_model = "fixture-embedding-not-a-model"
    generation_model = "fixture-generation-not-a-model"

    def __init__(self):
        self.calls = 0

    def health(self):
        return {"ready": True, "missing_models": []}

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def generate(self, question, sources):
        self.calls += 1
        if self.calls == 2:
            raise ModelProtocolError("Deliberate fixture failure for partial reporting.")
        return {"supported": False, "answer": ABSTENTION, "citations": []}


class AdditionalEvaluationTests(unittest.TestCase):
    def test_failed_and_unsupported_cases_retain_evidence_and_do_not_hide_execution_error(self):
        report = run_evaluation(AlwaysAbstainsFixture())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(exit_code(report), 1)
        self.assertEqual(len(report["cases"]), len(CASES))
        self.assertIn("not an untouched holdout", report["scope"])
        self.assertIn("not an accuracy score", report["manual_review_caveat"])
        self.assertEqual(report["models"]["generation"], "fixture-generation-not-a-model")
        self.assertTrue(report["lifecycle_checks"][0]["passed"])
        self.assertGreater(report["checks_total"], report["checks_passed"])
        first = report["cases"][0]
        self.assertTrue(first["retrieval_probe"][0]["text"])
        self.assertIsInstance(first["retrieval_probe"][0]["score"], float)
        self.assertFalse(next(check["passed"] for check in first["checks"] if check["name"] == "support-decision"))
        second = report["cases"][1]
        self.assertEqual(second["error"]["type"], "ModelProtocolError")
        self.assertTrue(any(check["name"] == "execution" and not check["passed"] for check in second["checks"]))
        absent = next(case for case in report["cases"] if case["id"] == "absent-numeric-fact")
        self.assertTrue(all(check["passed"] for check in absent["checks"]))
        self.assertTrue(all("elapsed_seconds" in case for case in report["cases"]))
        checks = report["lifecycle_checks"] + [check for case in report["cases"] for check in case["checks"]]
        self.assertEqual(report["checks_total"], len(checks))
        self.assertEqual(report["checks_passed"], sum(check["passed"] for check in checks))

    def test_missing_access_writes_blocked_report_and_inherits_app_defaults(self):
        fixture = AlwaysAbstainsFixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports" / "additional.json"
            with patch("tools.evaluate_additional.Ollama", return_value=fixture) as constructor, patch.object(
                fixture, "health", side_effect=ModelUnavailable("Local runtime unavailable: fixture.")
            ):
                self.assertEqual(main(["--output", str(output)]), 2)
                constructor.assert_called_once_with()
            report = json.loads(output.read_text())
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["cases"], [])
        self.assertEqual(report["error"]["type"], "ModelUnavailable")
        self.assertEqual(fixture.calls, 0)


if __name__ == "__main__":
    unittest.main()
