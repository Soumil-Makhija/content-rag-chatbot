#!/usr/bin/env python3
"""Run a small real-model evaluation against synthetic documents.

No model downloads or hosted providers are used. Exit 2 means model access is
blocked, 1 means checks or execution failed, and 0 means the automated checks
passed. Passing lexical checks is not a factual-quality or accuracy score.
Tests inject an explicitly synthetic model to regress the evaluation harness.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from content_chat.models import ModelUnavailable, Ollama  # noqa: E402
from content_chat.service import Service  # noqa: E402
from content_chat.store import Store  # noqa: E402


DISCLAIMER = (
    "Automated checks inspect selective top-k retrieval, lexical answer terms, citation titles and resolution, "
    "abstention, and corpus lifecycle. They are not an accuracy score or proof of "
    "factual grounding. Manually review every answer and its cited passages."
)


def _contains(text: str, term: str) -> bool:
    """Match whole phrases, so 30 does not accidentally match 130."""
    return re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", text.casefold()) is not None


def check_answer(actual: dict, expected: dict, collection: str, store: Store) -> list[dict]:
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("support-decision", actual.get("supported") is expected["supported"],
          f"Expected supported={expected['supported']}.")
    citations = actual.get("citations", [])
    answer = actual.get("answer", "")
    check("answer-present", isinstance(answer, str) and bool(answer.strip()), "Answer must be nonempty text.")
    if not isinstance(answer, str):
        answer = ""
    if not isinstance(citations, list) or any(not isinstance(cite, dict) for cite in citations):
        check("citations-format", False, "Expected a list of resolved source objects.")
        citations = []
    if not expected["supported"]:
        check("abstention-has-no-citations", not citations, "Unsupported answers must not cite evidence for an answer.")
    for alternatives in expected.get("answer_terms", []):
        check("answer-term:" + "|".join(alternatives), any(_contains(answer, term) for term in alternatives),
              "At least one expected phrase must appear; this is only a lexical check.")
    for forbidden in expected.get("forbidden_terms", []):
        check("forbidden-term:" + forbidden, not _contains(answer, forbidden), "A conflicting policy value must not appear.")
    titles = {cite.get("title") for cite in citations if isinstance(cite.get("title"), str)}
    for title in expected.get("citation_titles", []):
        check("citation-title:" + title, title in titles, "Expected document must be cited.")
    for index, cite in enumerate(citations):
        source_id = cite.get("id")
        resolved = store.source(collection, source_id) if isinstance(source_id, str) and source_id else None
        matches = resolved is not None and all(
            resolved.get(key) == cite.get(key) for key in ("document_id", "title", "version", "ordinal", "text")
        )
        check(f"citation-resolves:{index}", matches, "Citation must resolve to the stored passage in this collection.")
        check(f"citation-collection:{index}", cite.get("collection") == collection,
              "Citation must remain inside the requested collection.")
    return checks


def run_evaluation(model, manifest_path: Path | None = None) -> dict:
    """Exercise the actual Service and Store; injected models are for tests only."""
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "check_scope": "selective top-k retrieval, lexical and structural checks; manual factual review required",
        "manual_review_caveat": DISCLAIMER,
        "models": {
            "provider": "ollama",
            "url": getattr(model, "base_url", None),
            "embedding": getattr(model, "embedding_model", None),
            "generation": getattr(model, "generation_model", None),
        },
        "cases": [],
        "retrieval_cases": [],
        "lifecycle_checks": [],
    }

    def lifecycle(name: str, passed: bool, detail: str) -> None:
        report["lifecycle_checks"].append({"name": name, "passed": bool(passed), "detail": detail})

    def ask_case(service: Service, store: Store, case: dict) -> dict:
        case_started = time.perf_counter()
        result = {key: case[key] for key in ("id", "collection", "question", "expected")}
        result["limit"] = case.get("limit", 5)
        # Append before the request so a model failure leaves a useful record.
        report["cases"].append(result)
        try:
            actual = service.ask(case["collection"], case["question"], limit=result["limit"])
            result["actual"] = actual
            result["checks"] = check_answer(actual, case["expected"], case["collection"], store)
        except ModelUnavailable as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            result["checks"] = [{"name": "model-call", "passed": False, "detail": str(exc)}]
            raise
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            result["checks"] = [{"name": "execution", "passed": False, "detail": str(exc)}]
        finally:
            result["elapsed_seconds"] = round(time.perf_counter() - case_started, 6)
        return result

    def retrieval_case(service: Service, store: Store, case: dict) -> dict:
        case_started = time.perf_counter()
        result = {key: case[key] for key in ("id", "collection", "question", "limit")}
        result["expected"] = {"required_document_ids": case["required_document_ids"]}
        result["checks"] = []
        report["retrieval_cases"].append(result)

        def check(name: str, passed: bool, detail: str) -> None:
            result["checks"].append({"name": name, "passed": bool(passed), "detail": detail})

        try:
            count = sum(document["chunk_count"] for document in store.list_documents(case["collection"]))
            result["corpus_chunk_count"] = count
            check("corpus-exceeds-limit", count > case["limit"],
                  "The collection must contain more passages than top-k so retrieval cannot return the entire corpus.")
            retrieved = service.search(case["collection"], case["question"], limit=case["limit"])
            # Keep exact cosine scores and source text for inspection; rounded
            # timings do not alter the reported ranking or scores.
            result["retrieved"] = retrieved
            check("retrieval-limit", len(retrieved) <= case["limit"], "Retrieved passages must respect top-k.")
            found = {source["document_id"] for source in retrieved}
            for document_id in case["required_document_ids"]:
                check("required-document:" + document_id, document_id in found,
                      "An expected supporting document must appear in the selected top-k passages.")
            for index, source in enumerate(retrieved):
                resolved = store.source(case["collection"], source["id"])
                check(f"retrieved-source-resolves:{index}", resolved is not None and all(
                    resolved.get(key) == source.get(key) for key in ("document_id", "title", "version", "ordinal", "text")
                ), "Retrieved passages must resolve unchanged inside the selected collection.")
        except ModelUnavailable as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            check("model-call", False, str(exc))
            raise
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            check("execution", False, str(exc))
        finally:
            result["elapsed_seconds"] = round(time.perf_counter() - case_started, 6)
        return result

    try:
        # Health is deliberately first: missing access must not appear as a run
        # with successful fixture answers, and this tool never downloads models.
        report["health"] = model.health()
        if not report["health"].get("ready"):
            raise ModelUnavailable("Configured embedding and generation models are not both available: "
                                   + ", ".join(report["health"].get("missing_models", [])))
        manifest_path = manifest_path or ROOT / "examples" / "evaluation.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["corpus"] = manifest["description"]
        with tempfile.TemporaryDirectory(prefix="content-chat-evaluation-") as directory:
            database = Path(directory) / "index.sqlite3"
            with Store(database) as store:
                service = Service(store, model)
                imported = []
                for document in manifest["documents"]:
                    text = document.get("text")
                    if text is None:
                        text = (manifest_path.parent / document["file"]).read_text(encoding="utf-8")
                    metadata = service.import_document(document["collection"], document["title"], text, document["id"])
                    imported.append({"collection": document["collection"], **metadata})
                report["imported_documents"] = imported
            # Close and reopen the actual SQLite database before any question.
            with Store(database) as store:
                service = Service(store, model)
                lifecycle("restart-persists-documents", all(
                    store.get_document(document["collection"], document["id"]) is not None for document in imported
                ), "The vector store was closed and reopened before questions.")
                for case in manifest["retrieval_cases"]:
                    retrieval_case(service, store, case)
                for case in manifest["cases"]:
                    ask_case(service, store, case)

                change = manifest["lifecycle"]
                collection = change["collection"]
                document_id = change["document_id"]
                baseline = next((case for case in report["cases"] if case["id"] == "standard-refund"), {})
                old_ids = [cite["id"] for cite in baseline.get("actual", {}).get("citations", [])
                           if cite.get("document_id") == document_id]
                updated = service.import_document(collection, change["title"], change["replacement_text"], document_id)
                lifecycle("update-increments-version", updated["version"] == 2, "Replacing a document preserves its ID and increments its version.")
                lifecycle("update-invalidates-citations", bool(old_ids) and all(store.source(collection, source_id) is None for source_id in old_ids),
                          "Citations from the replaced billing document must stop resolving.")
                updated_case = ask_case(service, store, {
                    "id": "updated-refund", "collection": collection, "question": change["question"],
                    "limit": change["limit"], "expected": change["updated_expected"],
                })
                new_ids = [cite["id"] for cite in updated_case.get("actual", {}).get("citations", [])
                           if cite.get("document_id") == document_id]
                lifecycle("cross-collection-source-lookup", bool(new_ids) and all(store.source("enterprise", source_id) is None for source_id in new_ids),
                          "Support source IDs cannot resolve through the enterprise collection.")
                lifecycle("delete-document", store.delete(collection, document_id), "The updated support billing document was deleted.")
                lifecycle("delete-removes-document", store.get_document(collection, document_id) is None, "The deleted document is absent from the corpus.")
                lifecycle("delete-invalidates-citations", bool(new_ids) and all(store.source(collection, source_id) is None for source_id in new_ids),
                          "Deleted-document citations no longer resolve.")
                ask_case(service, store, {
                    "id": "deleted-refund", "collection": collection, "question": change["question"],
                    "limit": change["limit"],
                    "expected": {"supported": False, "answer_terms": [], "citation_titles": []},
                })
                # The other collection uses the same document ID; deleting or
                # updating support must leave its original 90-day policy intact.
                enterprise_case = next(case for case in manifest["cases"] if case["id"] == "enterprise-refund")
                ask_case(service, store, {**enterprise_case, "id": "enterprise-after-support-delete"})
        all_checks = [check for case in report["retrieval_cases"] + report["cases"] for check in case["checks"]] + report["lifecycle_checks"]
        report["checks_passed"] = sum(check["passed"] for check in all_checks)
        report["checks_total"] = len(all_checks)
        report["status"] = "checks_passed_manual_review_required" if all(check["passed"] for check in all_checks) else "failed"
    except ModelUnavailable as exc:
        report["status"] = "blocked"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return report


def exit_code(report: dict) -> int:
    return {"blocked": 2, "checks_passed_manual_review_required": 0}.get(report["status"], 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Write the evaluation JSON, including blocked and failed results.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--generation-model", default="qwen2.5:1.5b")
    args = parser.parse_args(argv)
    try:
        model = Ollama(args.ollama_url, args.embedding_model, args.generation_model)
        report = run_evaluation(model)
    except Exception as exc:
        report = {"schema_version": 1, "status": "failed", "manual_review_caveat": DISCLAIMER,
                  "error": {"type": type(exc).__name__, "message": str(exc)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.output}")
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
