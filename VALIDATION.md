# Validation record

Latest run: `2026-10-30_090531`, America/Vancouver. Continued the earliest unfinished project; no new repository was created. The previous run delivered the initial CLI, local web API and persistent vector store with 75 regression tests.

Status: **blocked on real local model availability**, not complete. No paid provider or usage budget was authorized; no credentials, gated models, model downloads, or billing setup were used. An optional request for authorized model access remains unanswered at this record.

## Observed checks

| Check | Observed result |
| --- | --- |
| Baseline `python3 -m unittest discover -v` | 75 passed in 3.003s, Python 3.14.2 |
| Correctness milestone | 81 passed in 2.973s |
| Import/retrieval/UI milestone | 86 passed in 3.098s |
| `node --test tests/web-state.test.js` | 3 passed; real browser event handlers executed with a minimal synthetic DOM/fetch harness |
| Exact cosine and persistence | Known-vector ranking, restart and collection isolation regressions pass |
| Large import atomicity | 67 chunks issue ordered requests of 32/32/3; vector-to-chunk mapping checked. Fifteen later-batch failure scenarios preserve the existing document/source IDs and avoid partial documents/collections |
| Context freshness | Changes/deletion of cited or uncited retrieved passages and stale abstention context require retry; unrelated collection changes remain valid |
| Protocol/input failure paths | Invalid UTF-8 identifiers rejected before inference; malformed/truncated HTTP responses become sanitized model protocol errors |
| CLI and HTTP | Persisted subprocess workflow and actual loopback HTTP lifecycle/error/isolation requests pass with explicit fixture inference |
| Package installation | Fresh temporary virtualenv install passed; installed entry point and empty-corpus abstention from outside the checkout passed; all HTML/JS/CSS assets are packaged |
| Compilation / JavaScript syntax / whitespace | `compileall`, `node --check`, and `git diff --check` pass |
| Real embedding / language inference | **Not run successfully**; no installed runtime/models or reachable model service |
| Browser QA | Chrome rendered the synthetic library; changing the collection cleared stale controls. Further interaction hit an automatic-review timeout. Full visual and live-model browser QA remain unverified |

Tests requiring local sockets were rerun with automatic-review escalation after sandbox binding was denied. This does not weaken global sandbox settings. Regression inference fixtures test orchestration and failure handling, not AI acceptance. GitHub Actions validates deterministic tests and packaging; its results are available in the repository's [checks](https://github.com/Soumil-Makhija/content-rag-chatbot/actions).

## Stronger evaluation, still awaiting real inference

```sh
python3 -m content_chat doctor
python3 tools/evaluate_live.py --output local-evaluation.json
```

The evaluator now imports eight support documents (two relevant and six distractors), plus a conflicting enterprise policy. After reopening SQLite, three retrieval probes require the correct documents at top-k 1/1/2 and record exact scores, full passages, expectations, corpus size and timing. A guard rejects a corpus small enough to fit entirely inside top-k. Nine answer/lifecycle cases retrieve at most two passages and cover answerable, multi-document, related/unrelated unsupported, conflicting collections, replacement and deletion behavior.

The explicitly synthetic harness run passed **82/82 structural and lexical checks**, with three retrieval and nine answer cases. Those hand-coded fixture vectors and answers are not a model evaluation. Regressions show that constant bad embeddings fail selective retrieval even when a canned answer contains an expected phrase; removing distractors also fails the corpus-size guard despite passing answer checks.

The actual live attempt was retried outside restricted sandbox networking. Result: exit **2**, `status: blocked`, zero retrieval cases and zero answer cases executed. Configured provider: Ollama at `http://127.0.0.1:11434`; embedding `nomic-embed-text`; generation `qwen2.5:3b`. A direct unsandboxed socket check returned `ConnectionRefusedError: [Errno 61] Connection refused`. No Ollama/llama runtime, relevant Python inference packages, or common model caches were found. Configuration names are not observed inference identities.

No accuracy, inference latency, grounding success or prompt-injection resistance is claimed. Passing lexical/structural checks will still require manual review of every real answer and its supporting passages. Loopback alone cannot prove local inference: disable Ollama cloud features and use installed local models. The builder changed no host model settings.

## Acceptance evidence still needed

| Acceptance criterion | Current evidence and outstanding work |
| --- | --- |
| Import, restart, persisted vector index | Fixture persistence/restart and atomic batched imports pass; repeat using real embeddings |
| Real embeddings + generation with resolvable citations | Adapter, ranking and source resolution regressions pass; actual calls and grounded answers remain unverified |
| Answerable and unsupported questions | Selective retrieval/answer evaluation set implemented; run real models and manually inspect every answer/citation/abstention |
| Updates/deletion and collection isolation | Deterministic lifecycle, stale context and HTTP tests pass; repeat real model-backed lifecycle evaluation |

To finish, make an authorized local runtime and embedding/generation models available, then run doctor and the live evaluator. Inspect actual answers and retrieval ranking, fix model-specific failures, and record evidence before marking complete. A hosted alternative requires a separately authorized provider, credential, usage limits and an implemented adapter; this app currently supports local Ollama only.
