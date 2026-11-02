# Validation record

Latest recovery: **November 1, 2026, America/Vancouver**. Status: **in progress because of an observed model-quality failure**. Local model availability is resolved. Real inference used Ollama 0.40.13 with `nomic-embed-text` and `qwen2.5:1.5b`; no paid model API, API credentials, billing setup, or private documents were used. The application defaults now select that tested pair.

## Current evidence

| Check | Observed result |
| --- | --- |
| `python3 -m unittest discover -v` | 86 passed; fixture inference tests protocol, storage, service, CLI and actual loopback HTTP behavior |
| `node --test tests/web-state.test.js` | 3 passed; handlers use a synthetic DOM/fetch harness |
| Real Chrome 156/macOS smoke | All 12 checks passed with actual 1.5b generation and embeddings; imports, library display, supported trial answer, visible citations, source resolution, 390px layout, unsupported abstention and no JavaScript page errors |
| Live selective retrieval | 3/3 cases met expectations at top-k 1/1/2 against eight support passages, including six distractor documents |
| Live question cases | 8/9 cases met every expected check; the multi-document answer omitted the refund window |
| Live lifecycle checks | 7/7 passed: reopen/persistence, replacement/version changes, stale citation invalidation, deletion and collection-scoped source lookup |
| Default live evaluation | 85/86 lexical/structural checks passed in 22.82 seconds; `status: failed`, exit 1, correctly preserving the quality failure |

The real browser smoke also checked the visible unsupported-answer message and source records. Desktop and narrow screenshots were visually reviewed. A favicon 404 was cosmetic; there were no application page errors. This does not establish other browser engines, screen-reader behavior, real-world model accuracy, or performance guarantees.

Regression fixtures remain explicitly synthetic. Their passing results do not replace the separate live-model evidence. CI runs deterministic tests and packaging rather than downloading model weights; see the repository's [Actions runs](https://github.com/Soumil-Makhija/content-rag-chatbot/actions) for exact-commit results.

## Raw live evidence and the remaining gap

The committed [1.5b evaluation JSON](docs/live-evaluation-2026-11-01.json) is an unchanged copy of the final recovery run. It records synthetic source text, actual retrieved passages and cosine scores, question expectations, model names, full answers/citations, individual checks and timings. Three selective retrieval probes and nine question cases execute after closing/reopening SQLite. Question retrieval is limited to at most two passages; a corpus-size guard rejects retrieval that can return the entire collection.

The failed question asks how to export a workspace **and** the standard annual subscription refund window. Retrieval returned both relevant documents, and the answer cited them, but generated text described only Settings → Export → JSON/Markdown. It omitted the supported **30-day** refund window. The `answer-term:30|thirty` check remains failed. The evaluator and expected answers have not been weakened to make this result pass.

The [original 3b evaluation JSON](docs/live-evaluation-3b-2026-11-01.json) records a separate run with `qwen2.5:3b`: 69/78 checks passed. It answered the combined question but falsely abstained on the supported standard 30-day refund and updated 45-day refund questions. It is not the later experimental balanced-prompt run. Check totals vary with emitted answers/citations; these few synthetic runs do not establish a general model ranking or prove a larger model better. Explicit model overrides remain available.

**85/86 checks and 8/9 cases are observed test counts, not accuracy estimates.** Valid source IDs and expected phrases cannot establish entailment, completeness, hallucination resistance, or prompt-injection resistance. Manual review of the real answers identified the omission. Broader and repeated evaluation remains needed.

## Reproduce

Start a local-only Ollama server and install the selected models yourself if they are not already installed:

```sh
OLLAMA_NO_CLOUD=1 ollama serve
# In another terminal:
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
python3 -m content_chat doctor
python3 tools/evaluate_live.py --output local-evaluation.json
# Optional explicit comparison; never inferred to be an upgrade:
python3 tools/evaluate_live.py --generation-model qwen2.5:3b --output local-evaluation-3b.json
```

The app does not pull weights or alter host model settings. Evaluation creates a disposable database and does not modify an existing user collection. Exit 2 denotes unavailable model access; 1 denotes failed checks/execution; 0 denotes passed automated checks with manual review still required. Exact generated wording may vary with runtime/model weights and hardware. No paid provider is needed for the local workflow.

## Acceptance status

| Criterion | Evidence and remaining work |
| --- | --- |
| Import, restart, persisted vector index | Observed with real embeddings and restarted SQLite; fixture atomicity regressions also pass |
| Real embedding/generation with resolvable citations | Observed through evaluator and browser; source existence is verified, but complete grounded answers are not guaranteed |
| Answerable and unsupported questions | Real unsupported related/unrelated cases abstain; one supported multi-part case remains incomplete |
| Updates/deletion and collection isolation | Real 45-day update, deletion abstention and conflicting 90-day enterprise policy behaved as expected with the default model |

The project remains in progress until the multi-part quality gap is fixed and the unchanged evaluation is rerun and reviewed. Runtime installation alone does not complete the AI acceptance criteria.

## Earlier validation history

October 29 delivered the initial CLI, API and persistent vector store with 75 regressions. October 30 (`2026-10-30_090531`) improved freshness checks, malformed HTTP handling, bounded atomic imports, selective retrieval evaluation and collection UI state; 86 Python and 3 JavaScript tests passed, along with fresh installation and packaged-asset checks. At that time no local model runtime/server was available: live evaluation exited 2 with zero cases, and browser work was partial. Those were accurate historical access limitations, now superseded by November 1 real-model and browser evidence. The fixture evaluator's historical 82/82 checks were not a real-model evaluation.
