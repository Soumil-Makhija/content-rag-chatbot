#!/usr/bin/env python3
"""Opt-in fresh-install CLI/API smoke with real local Ollama inference.

Creates a temporary virtualenv/database, installs this checkout, and runs outside
it. Uses only synthetic examples. Does not download models or use hosted APIs.
Installing the package may fetch its declared setuptools build dependency.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]


def run(output: Path) -> dict:
    started = time.perf_counter()
    report = {"schema_version": 1, "status": "failed", "started_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Real local inference on synthetic FAQ inputs; not an accuracy estimate.", "commands": [], "checks": [], "answers": {}}
    temporary = None

    def save():
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if temporary:
            text = text.replace(str(temporary), "<temporary-dir>")
        text = text.replace(str(ROOT), "<checkout>")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")

    def check(name, passed):
        report["checks"].append({"name": name, "passed": bool(passed)})
        save()
        if not passed:
            raise RuntimeError("Smoke check failed: " + name)

    try:
        with tempfile.TemporaryDirectory(prefix="content-chat-installed-") as directory:
            temporary = Path(directory)

            def command(args):
                result = subprocess.run([str(arg) for arg in args], cwd=temporary, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
                report["commands"].append({"arguments": [str(arg) for arg in args], "exit_code": result.returncode, "output": result.stdout})
                save()
                if result.returncode:
                    raise RuntimeError("Installed command failed; inspect commands in the report.")
                return result.stdout

            command([sys.executable, "-m", "venv", temporary / "venv"])
            python = temporary / "venv/bin/python"
            cli = temporary / "venv/bin/content-chat"
            command([python, "-m", "pip", "install", "--cache-dir", temporary / "pip-cache", ROOT])
            command([cli, "--help"])
            database = temporary / "smoke.sqlite3"
            base = [cli, "--db", database]
            for name in ("support", "billing"):
                document = json.loads(command(base + ["import", "smoke", ROOT / f"examples/{name}.md", "--id", name]))
                check("cli-import-" + name, document["version"] == 1)
            check("cli-restart-persists-library", len(json.loads(command(base + ["list", "smoke"]))) == 2)
            server_code = (
                "from content_chat.server import make_server;from content_chat.models import Ollama;import sys;"
                "server=make_server(sys.argv[1],Ollama(),0);print(server.server_port,flush=True);server.serve_forever()"
            )
            process = subprocess.Popen([str(python), "-c", server_code, str(database)], cwd=temporary,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    if not selector.select(timeout=20):
                        raise RuntimeError("Installed HTTP server did not start within 20 seconds.")
                    port = int(process.stdout.readline().strip())
                url = f"http://127.0.0.1:{port}"
                opener = build_opener(ProxyHandler({}))

                def request(path, payload=None, method=None, expected_status=200, raw=False):
                    data = None if payload is None else json.dumps(payload).encode("utf-8")
                    req = Request(url + path, data=data, method=method,
                                  headers={"Content-Type": "application/json"})
                    try:
                        response = opener.open(req, timeout=240)
                    except HTTPError as exc:
                        response = exc
                    with response:
                        body = response.read()
                        if response.status != expected_status:
                            raise RuntimeError(f"Unexpected HTTP {response.status} for {path}.")
                    return body if raw else json.loads(body)

                health = request("/api/health")
                report["models"] = health["models"]
                check("installed-default-models-ready", health["models"]["ready"] and health["models"]["generation_model"] == "qwen2.5:3b")
                for asset in ("/", "/app.js", "/style.css"):
                    check("packaged-asset:" + asset, bool(request(asset, raw=True)))
                answer = request("/api/collections/smoke/ask", {
                    "question": "How do I export my workspace, and what is the standard annual subscription refund request window?", "limit": 2})
                report["answers"]["multipart"] = answer
                check("installed-real-multipart-answer", answer["supported"] and all(term in answer["answer"].lower() for term in ("settings", "export", "30"))
                      and {cite["document_id"] for cite in answer["citations"]} == {"support", "billing"})
                for citation in answer["citations"]:
                    check("source-url:" + citation["document_id"], request(citation["url"])["text"] == citation["text"])
                unknown = request("/api/collections/smoke/ask", {
                    "question": "Exactly how many business days will my bank take to post an approved refund?", "limit": 2})
                report["answers"]["unsupported"] = unknown
                check("installed-real-unsupported-answer", unknown["supported"] is False and not unknown["citations"])
                old = next(cite for cite in answer["citations"] if cite["document_id"] == "billing")
                updated_text = "The standard annual subscription refund request window is now 45 days from purchase. This replaces the previous policy."
                updated = request("/api/collections/smoke/documents", {"document_id": "billing", "title": "Updated billing policy", "text": updated_text})
                check("api-replacement-version", updated["version"] == 2)
                request(old["url"], expected_status=404)
                check("replaced-source-invalidated", True)
                updated_answer = request("/api/collections/smoke/ask", {"question": "What is the standard annual subscription refund request window?", "limit": 2})
                report["answers"]["updated"] = updated_answer
                check("updated-real-answer", updated_answer["supported"] and "45" in updated_answer["answer"] and "30" not in updated_answer["answer"])
                request("/api/collections/other/documents", {"document_id": "billing", "title": "Enterprise billing", "text": "Enterprise annual subscriptions have a 90-day refund request window from purchase."})
                isolated = request("/api/collections/other/ask", {"question": "What is the enterprise annual subscription refund request window?", "limit": 2})
                report["answers"]["isolated"] = isolated
                check("isolated-real-answer", isolated["supported"] and "90" in isolated["answer"] and all(c["collection"] == "other" for c in isolated["citations"]))
                request(updated_answer["citations"][0]["url"].replace("/smoke/", "/other/"), expected_status=404)
                check("cross-collection-source-404", True)
                deleted = request("/api/collections/smoke/documents/billing", method="DELETE")
                check("api-delete-document", deleted["deleted"] is True)
                request(updated_answer["citations"][0]["url"], expected_status=404)
                check("deleted-source-invalidated", True)
                absent = request("/api/collections/smoke/ask", {"question": "What is the standard annual subscription refund request window?", "limit": 2})
                report["answers"]["after-delete"] = absent
                check("deleted-policy-abstains", absent["supported"] is False and not absent["citations"])
                check("other-collection-retained", len(request("/api/collections/other/documents")) == 1)
            finally:
                process.terminate()
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=10)
        report["status"] = "checks_passed_manual_review_required"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    report["checks_passed"] = sum(check["passed"] for check in report["checks"])
    report["checks_total"] = len(report["checks"])
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(f"{result['status']}: {args.output}")
    return 0 if result["status"] == "checks_passed_manual_review_required" else 1


if __name__ == "__main__":
    raise SystemExit(main())
