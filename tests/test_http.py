"""Real loopback HTTP integration with an explicit non-model fixture provider."""

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from content_chat.models import ABSTENTION, ModelProtocolError, ModelUnavailable
from content_chat.server import MAX_BODY, make_server


class HTTPFixtureModel:
    """Tests HTTP/storage orchestration only; these responses are not AI evidence."""

    embedding_model = "http-fixture-embedding"
    generation_model = "http-fixture-generation"

    def __init__(self):
        self.embedding_calls = []
        self.generation_calls = []
        self.embedding_error = None
        self.generation_error = None
        self.health_error = None

    def embed(self, texts):
        self.embedding_calls.append(list(texts))
        if self.embedding_error:
            raise self.embedding_error
        return [[1.0, 0.0] for _text in texts]

    def generate(self, question, sources):
        self.generation_calls.append((question, sources))
        if self.generation_error:
            raise self.generation_error
        return {"supported": True, "answer": sources[0]["text"], "citations": [sources[0]["id"]]}

    def health(self):
        if self.health_error:
            raise self.health_error
        return {"available": True, "ready": True, "missing_models": []}


class HTTPTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model = HTTPFixtureModel()
        self.server = make_server(Path(directory.name) / "http.sqlite3", self.model, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive(), "HTTP server did not stop")

    def request(self, path, method="GET", payload=None, raw=None, headers=None):
        headers = dict(headers or {})
        if payload is not None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if raw is not None:
            headers.setdefault("Content-Type", "application/json")
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=raw, headers=headers)
            response = connection.getresponse()
            body = response.read()
            response_headers = dict(response.getheaders())
            if response.getheader("Content-Type", "").startswith("application/json"):
                body = json.loads(body.decode("utf-8"))
            return response.status, body, response_headers
        finally:
            connection.close()

    def api(self, path, method="GET", payload=None, expected=200, **kwargs):
        status, result, _headers = self.request(path, method, payload, **kwargs)
        self.assertEqual(status, expected, result)
        return result

    def import_document(self, collection="support", text="Returns last thirty days.", title="FAQ", document_id="faq"):
        return self.api(f"/api/collections/{collection}/documents", "POST", {
            "title": title, "text": text, "document_id": document_id,
        })

    def ask(self, collection="support"):
        return self.api(f"/api/collections/{collection}/ask", "POST", {"question": "What is the policy?"})

    def test_document_import_list_update_delete_and_resolvable_citation_lifecycle(self):
        original_text = "# Retours\nLes retours durent trente jours. 東京🙂"
        metadata = self.import_document(text=original_text, title="Règlement")
        self.assertEqual(metadata, {"id": "faq", "title": "Règlement", "version": 1, "chunk_count": 1})
        self.assertEqual(self.api("/api/collections/support/documents"), [metadata])
        self.assertEqual(self.api("/api/collections/support/documents/faq"), {**metadata, "text": original_text})
        answer = self.ask()
        self.assertTrue(answer["supported"])
        self.assertEqual(answer["answer"], original_text)
        citation = answer["citations"][0]
        self.assertEqual(citation["collection"], "support")
        source = self.api(citation["url"])
        self.assertEqual(source["id"], citation["id"])
        self.assertEqual(source["document_id"], "faq")
        self.assertEqual(source["text"], original_text)

        updated = self.import_document(text="Returns now last fourteen days.", title="Updated FAQ")
        self.assertEqual(updated["version"], 2)
        self.api(citation["url"], expected=404)
        revised = self.ask()
        self.assertEqual(revised["answer"], "Returns now last fourteen days.")
        revised_citation = revised["citations"][0]
        self.assertEqual(revised_citation["version"], 2)
        self.assertNotEqual(revised_citation["id"], citation["id"])
        hits = self.api("/api/collections/support/search", "POST", {"question": "Returns?", "limit": 1})
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], revised_citation["id"])
        self.assertAlmostEqual(hits[0]["score"], 1)

        self.assertEqual(self.api("/api/collections/support/documents/faq", "DELETE"), {"deleted": True})
        self.api("/api/collections/support/documents/faq", expected=404)
        self.api("/api/collections/support/documents/faq", "DELETE", expected=404)
        self.api(revised_citation["url"], expected=404)
        self.assertEqual(self.api("/api/collections/support/documents"), [])
        self.assertFalse(self.ask()["supported"])

    def test_two_collections_keep_documents_answers_and_sources_separate(self):
        alpha = self.import_document("alpha", text="Alpha policy.")
        beta = self.import_document("beta", text="Beta private policy.")
        self.assertEqual(alpha["id"], beta["id"])
        first = self.ask("alpha")
        second = self.ask("beta")
        self.assertEqual(first["answer"], "Alpha policy.")
        self.assertEqual(second["answer"], "Beta private policy.")
        for source in self.model.generation_calls[0][1]:
            self.assertEqual(source["text"], "Alpha policy.")
        alpha_source_id = first["citations"][0]["id"]
        self.api(f"/api/collections/beta/sources/{alpha_source_id}", expected=404)
        self.api("/api/collections/alpha/documents/faq", "DELETE")
        self.assertEqual(self.ask("beta")["answer"], "Beta private policy.")

    def test_empty_collection_abstains_without_model_access(self):
        self.model.embedding_error = ModelUnavailable("No local model is running.")
        answer = self.ask("empty")
        self.assertEqual(answer["answer"], ABSTENTION)
        self.assertFalse(answer["supported"])
        self.assertEqual(answer["citations"], [])
        self.assertEqual(self.model.embedding_calls, [])
        self.assertEqual(self.model.generation_calls, [])

    def test_unavailable_or_invalid_models_have_explicit_http_statuses(self):
        self.model.health_error = ModelUnavailable("No local model is running.")
        health = self.api("/api/health")
        self.assertEqual(health["application"], "ready")
        self.assertFalse(health["models"]["available"])
        self.assertFalse(health["models"]["ready"])
        self.model.embedding_error = ModelUnavailable("Embedding model is missing.")
        result = self.api("/api/collections/support/documents", "POST", {"title": "FAQ", "text": "Fact."}, expected=503)
        self.assertIn("Embedding model", result["error"])
        self.assertEqual(self.api("/api/collections/support/documents"), [])
        self.model.embedding_error = None
        self.import_document()
        for error, expected in [(ModelUnavailable("Generator is unavailable."), 503),
                                (ModelProtocolError("Invalid generated citation."), 502)]:
            self.model.generation_error = error
            result = self.api("/api/collections/support/ask", "POST", {"question": "Policy?"}, expected=expected)
            self.assertEqual(result["error"], str(error))
        self.model.health_error = ModelProtocolError("Invalid local model response.")
        self.assertFalse(self.api("/api/health")["models"]["ready"])

    def test_invalid_json_content_type_and_framing_fail_without_model_calls(self):
        cases = [
            (b"{", {}),
            (b"[]", {}),
            (b"null", {}),
            (b'{"question": NaN}', {}),
            (b"\xff", {}),
            (b"{}", {"Content-Type": "text/plain"}),
            (b"{}", {"Content-Length": "bad"}),
            (b"{}", {"Content-Length": str(MAX_BODY + 1)}),
            (b"{}", {"Transfer-Encoding": "chunked"}),
        ]
        for raw, headers in cases:
            with self.subTest(raw=raw, headers=headers):
                result = self.api("/api/collections/support/ask", "POST", raw=raw, headers=headers, expected=400)
                self.assertIn("error", result)
        self.assertEqual(self.model.embedding_calls, [])
        self.assertEqual(self.model.generation_calls, [])

    def test_host_and_origin_guards_block_cross_origin_reads_and_writes(self):
        port = self.server.server_port
        for method, path, payload in [("GET", "/api/health", None),
                                      ("POST", "/api/collections/support/documents", {"title": "FAQ", "text": "Fact."})]:
            for headers in [{"Host": "attacker.example"}, {"Host": f"127.0.0.1:{port}", "Origin": "https://attacker.example"},
                            {"Origin": "null"}, {"Host": "localhost"}]:
                with self.subTest(method=method, headers=headers):
                    self.api(path, method, payload, headers=headers, expected=403)
        self.assertEqual(self.api("/api/collections/support/documents"), [])
        self.assertEqual(self.model.embedding_calls, [])
        valid = {"Host": f"localhost:{port}", "Origin": f"http://localhost:{port}"}
        self.assertEqual(self.api("/api/health", headers=valid)["application"], "ready")

    def test_static_assets_have_security_headers_and_only_allowlisted_paths_resolve(self):
        for path, content_type, marker in [("/", "text/html", b"<!doctype html>"),
                                           ("/app.js", "text/javascript", b"fetch("),
                                           ("/style.css", "text/css", b"body")]:
            with self.subTest(path=path):
                status, body, headers = self.request(path)
                self.assertEqual(status, 200)
                self.assertTrue(headers["Content-Type"].startswith(content_type))
                self.assertIn(marker, body.lower())
                self.assertEqual(int(headers["Content-Length"]), len(body))
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertEqual(headers["X-Frame-Options"], "DENY")
                self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
                self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
                self.assertNotIn("Access-Control-Allow-Origin", headers)
        for path in ["/../store.py", "/%2e%2e/store.py", "/web/../store.py", "/.git/config", "/%2f..%2fstore.py"]:
            with self.subTest(path=path):
                self.api(path, expected=404)


if __name__ == "__main__":
    unittest.main()
