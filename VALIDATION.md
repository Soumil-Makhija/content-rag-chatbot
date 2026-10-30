# Validation record

Implementation date: 2026-10-30 America/Vancouver, continuing scheduled run `2026-10-29_090848`.

Status: **blocked on real local model availability**, not complete. No paid provider or usage budget was authorized; no credentials, gated models, model downloads, or billing setup were used.

## Observed checks

| Check | Observed result |
| --- | --- |
| `python3 -m unittest discover -v` | 75 passed in 2.895s on the final integration on Python 3.14.2 |
| Exact cosine and persistence | Passed known-vector, disk reopen, Unicode chunk tests |
| Corpus lifecycle | Passed update/delete, stale sources, transaction rollback, concurrent model-change regression |
| Isolation and provenance | Passed collection-scoped retrieval and citation lookups |
| API and CLI | Passed actual loopback HTTP requests and persisted CLI subprocess use, with explicit fixture inference |
| Package installation | Passed clean temporary venv install, installed `content-chat --help`, packaged HTML/JS/CSS inspection |
| Compilation / JS syntax / whitespace | `compileall`, `node --check`, `git diff --check` passed |
| Real embedding / language inference | **Not run successfully**; no local runtime/models/service available |
| Browser visual QA | Unavailable: browser-control surfaces `iab` and `chrome` not available |

Regression inference fixtures test orchestration and error handling, not the AI acceptance criteria. CI validates deterministic tests and packaging; it does not establish real-model quality. GitHub Actions results are available in the repository's [checks](https://github.com/Soumil-Makhija/content-rag-chatbot/actions).

## Live evaluation attempt

```sh
python3 tools/evaluate_live.py --output local-evaluation.json
```

The actual attempt was retried outside the restricted sandbox so a denied socket would not be mistaken for missing access. Result: exit **2**, `status: blocked`, no model cases executed.

Configured provider: local Ollama at `http://127.0.0.1:11434`. Configured embedding: `nomic-embed-text`; generation: `qwen2.5:3b`. Error: `Cannot reach Ollama at http://127.0.0.1:11434. Start Ollama and install the configured models, then try again.` Local ports 11434, 1234 and 8080 refused connections; no Ollama/LM Studio applications, binaries, or common model caches were found. These are configuration names, not observed inference identities.

No accuracy, latency benchmark, grounding success, or prompt-injection resistance is claimed. The evaluation report's short health-check duration measures a failed availability check, not inference latency.

## Acceptance evidence still needed

| Acceptance criterion | Current evidence and outstanding work |
| --- | --- |
| Import, restart, persisted vector index | Persistence and restart pass with fixture embeddings; repeat through the real embedding workflow |
| Real embeddings + generation with resolvable citations | Adapter and source resolution regressions pass; actual model calls and grounded answers remain unverified |
| Answerable and unsupported evaluation questions | Evaluation set/harness implemented; run real models and manually inspect all answers/citations/abstentions |
| Updates/deletion and collection isolation | Deterministic lifecycle and HTTP tests pass; repeat real model-backed lifecycle evaluation |

To finish, supply/start a local Ollama runtime with suitable installed embedding and generation models, run `doctor`, run the evaluator, and inspect the recorded answers. Fix actual model-specific failures before marking complete. A hosted alternative would require a separately authorized provider, credentials, usage limits and an implemented adapter; the app currently supports local Ollama only.

Final integration adds an explicit known-cloud-model-name guard and documents the local-server trust boundary. Loopback alone is not proof of local inference; operators must disable Ollama cloud features and use installed models. The builder did not change host Ollama settings.
