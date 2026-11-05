# Validation record

Latest release validation: **November 5, 2026, America/Vancouver**. Status: **working local release; central acceptance criteria observed**. The production extractive workflow uses local Ollama 0.40.13 with `nomic-embed-text` embeddings and `qwen2.5:3b` generation, now the application defaults. Real inference used synthetic public FAQ content; no paid model API, billing setup or private documents were used.

The tested core is commit [`60fec8e`](https://github.com/Soumil-Makhija/content-rag-chatbot/commit/60fec8e56bed9d39f0ca35063edc7d75b8f484eb). Its push, remote SHA, GitHub author and committer account attribution were verified. The [core CI run](https://github.com/Soumil-Makhija/content-rag-chatbot/actions/runs/41879580213) passed on Python 3.11 and 3.14. The installed CLI/HTTP integration checkpoint [`7df4b32`](https://github.com/Soumil-Makhija/content-rag-chatbot/commit/7df4b321ccf18b4c18aa9bc9a11de481fc35101f) was pushed separately after its real-model smoke passed; its [CI run](https://github.com/Soumil-Makhija/content-rag-chatbot/actions/runs/42063627885) also passed on both Python versions. GitHub author and committer account logins were verified as `Soumil-Makhija` for both checkpoints. Exact revision CI results are retained in Actions.

## Current evidence

| Check | Observed result |
| --- | --- |
| `python3 -m unittest discover -v` | 108 passed; explicit synthetic inference tests protocol, persistent storage, service, CLI, actual loopback HTTP behavior and evaluation harnesses |
| `node --test tests/web-state.test.js` | 3 passed; handlers use a synthetic DOM/fetch harness |
| Main real-model evaluation | 107/107 checks in 74.030123 seconds; 11 question cases, 3 selective retrieval probes and 7 lifecycle checks |
| Additional real-model evaluation | 114/114 checks in 97.724859 seconds; 12 question cases and persistence after reopening SQLite |
| Manual factual review | All 23 question cases reviewed against source passages; requested facts, qualifiers, abstentions and citations were checked; no material issue found |
| Installed-package/HTTP integration | 20/20 checks in 58.324504 seconds; installed outside checkout; all five answers/abstentions manually reviewed |
| Current browser workflow | 12/12 checks in Chrome 156 on macOS; real uploads and inference, citations, abstention and 390px layout; screenshots reviewed |
| Remote CI | Core and installed-integration checkpoints passed on Python 3.11/3.14; exact revision results in Actions |

Deterministic regressions use fixture inference and do not replace the two separate live-model runs. CI tests packaging and deterministic behavior; it does not download models or perform real inference. Exact-commit results are available in [Actions](https://github.com/Soumil-Makhija/content-rag-chatbot/actions).

## Current real-model evidence

The committed [main evaluation JSON](docs/live-evaluation-2026-11-04.json) and [additional evaluation JSON](docs/additional-evaluation-2026-11-04.json) retain exact source content, configured models, answers, passages, citation records, individual checks and timings. Both runs finished with `checks_passed_manual_review_required`; the subsequent manual review described below checked every question case. The files preserve the evaluator's original status rather than replacing it with an automated accuracy claim.

The main corpus includes eight standard-support passages, six of them distractors, plus a separate enterprise collection. SQLite is closed and reopened before three selective retrieval probes at top-k 1/1/2. Question retrieval uses at most two passages. A corpus-size guard prevents a whole-corpus prompt from passing selective retrieval checks. All 11 question cases met expectations: seven supported answers and four abstentions. Manual inspection confirmed:

- The trial answer gives 14 days. Standard annual refunds give 30 days measured from the original purchase date.
- Export/refund questions in both clause orders include the full Settings → Export → JSON/Markdown procedure and the refund window, preserving request order.
- Bank posting time, an unrelated question and a mixed supported/unsupported compound request abstain without inventing a related value.
- Replacement returns 45 days from purchase from billing version 2; previous source IDs stop resolving. Deletion removes the standard billing document and produces an abstention.
- Enterprise answers retain the separate 90-day policy before and after the standard collection's deletion. Source IDs cannot resolve through another collection.

All seven lifecycle checks passed: persistence, version increments, replacement citation invalidation, collection-scoped lookup, deletion, removal from the corpus and deleted-source invalidation.

The additional corpus has six synthetic Juniper Workshop documents. All 12 question cases met expectations: eight supported answers and four abstentions. Manual inspection confirmed complete individual and compound export/18-day retention answers; explicit non-refundable annual membership policy; standard 2–4 business-day and expedited 1-business-day usual ranges with their qualifiers intact; and correct abstentions for a missing badge fee, a mixed known/unknown request, a guaranteed deadline unsupported by a usual range, and monthly memberships unsupported by the annual policy. The locker answer gives the factual 23-day rotation and ignores an embedded instruction to answer “999 days.” This one source-instruction example does not establish general prompt-injection resistance.

**107/107 and 114/114 are lexical/structural check counts, not accuracy estimates.** The additional set informed development and is a regression set, not an untouched holdout. Valid citations and matching phrases alone do not establish entailment or completeness. Manual review supports these particular observed outputs; it does not guarantee future answers, unseen documents or other model configurations. Check totals can change with emitted answers and citations, so they are not a controlled model ranking.

## What was accepted

The production workflow returns exact source quotes selected through real language-model generation. It plans requests without source context, selects evidence independently for each request, validates every quote against its named retrieved passage, and composes answers directly from those quotes. Rewritten or split plans receive a separate source-free coverage review against the original question. A single request exactly equal to the original question after outer whitespace trimming establishes coverage mechanically and skips that model check. Each request then receives its own semantic review with its quoted answer, full cited passages and single request text. All required evidence and reviews must pass; a rejected or unsupported part causes an abstention for the whole question. Strict malformed output produces an error.

Quote matching establishes provenance. It cannot establish relevance, preserve context by itself or prove support. Models can still lose a qualifier, select a misleading excerpt, approve unrelated evidence or falsely abstain. Full passage context is supplied to each support reviewer, but the same model's review remains fallible. The application does not claim general factual accuracy, independent verification or reliable resistance to malicious source instructions. Answers are extractive; generated paraphrasing, open-ended synthesis and calculations are outside the delivered scope. See [architecture and call bounds](ARCHITECTURE.md) for details.

| Criterion | Current observed evidence |
| --- | --- |
| Import, restart and query a persistent vector index | Real embeddings, closed/reopened SQLite and selective retrieval; deterministic atomicity regressions also pass |
| Real embedding/generation with resolvable citations | Both current evaluators use the production local model path; all 15 supported answers across the two runs were reviewed against cited passages |
| Answerable and unsupported questions | All 23 question cases reviewed, including eight correct abstentions; current compound and reversed-order answers include every requested fact |
| Updates/deletion and collection isolation | Real 45-day replacement, invalidated sources, deleted-standard abstention and separate 90-day enterprise policy |

The listed central acceptance criteria and installed release workflow have been observed. No central requirement remains open within the documented Markdown/text, local, extractive scope.

## Reproduce

Start a local-only Ollama server and install the chosen models yourself if they are not already installed:

```sh
OLLAMA_NO_CLOUD=1 ollama serve
# In another terminal:
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
python3 -m content_chat doctor
python3 tools/evaluate_live.py --output local-evaluation.json
python3 tools/evaluate_additional.py --output local-additional-evaluation.json
python3 tools/smoke_installed.py --output local-evaluation-installed.json
# Optional model comparison; requires the selected model to be installed:
python3 tools/evaluate_live.py --generation-model qwen2.5:1.5b --output local-evaluation-1.5b.json
```

The opt-in installed smoke creates a fresh virtual environment and runs the installed CLI and HTTP server outside the checkout, checking packaged assets, source resolution, persistence, replacement/deletion and isolation with real local models and synthetic content. Installation may fetch the declared build dependency through pip. The smoke does not download models or call paid APIs. The [November 5 result](docs/installed-smoke-2026-11-05.json) passed 20/20 checks using the installed CLI and server outside the checkout. Manual review confirmed the complete export/30-day answer, unsupported bank-time abstention, 45-day replacement, isolated 90-day policy, and abstention after deletion. Old sources returned 404 after replacement/deletion and through a different collection.

The [current browser result](docs/browser-smoke-2026-11-05.json) passed 12/12 checks with the 3b production defaults: both example files uploaded through the UI, the 14-day trial answer had a resolvable source, an unrelated question abstained visibly, and there were no JavaScript page errors. [Desktop](docs/browser-desktop-2026-11-05.png) and [390px](docs/browser-narrow-2026-11-05.png) screenshots were visually reviewed; no clipping or horizontal overflow was found. The browser used a fresh temporary profile and synthetic inputs. The browser engine and assistive-technology limitations in README still apply.

The app does not pull weights or alter host model settings. Evaluation creates disposable databases and does not modify existing user collections. For the two model evaluators, exit 2 denotes unavailable model access; 1 denotes failed checks/execution; 0 denotes passed automated checks with manual review still required. The installed smoke returns 1 for failed setup/integration and 0 for passed automated checks. Model tags are configurable and are not pinned by a weight digest. Outputs may vary with weights, runtime and hardware; changing the generation model requires fresh evaluation. No paid provider is needed for this local workflow.

## Earlier validation history

Earlier evidence remains preserved and describes earlier implementations, not current results:

- October 29 delivered the initial CLI, API and persistent vector store with 75 regressions. October 30 improved freshness checks, malformed HTTP handling, bounded atomic imports, selective retrieval evaluation and collection UI state; 86 Python and 3 JavaScript tests passed, along with fresh-install and packaged-asset checks. No local runtime was available then: live evaluation exited 2 with zero cases. The historical fixture evaluator's 82/82 checks were not real inference.
- November 1 resolved local runtime access. The [1.5b evaluation](docs/live-evaluation-2026-11-01.json) failed 85/86 checks across nine cases because a compound export/refund answer omitted the 30-day refund window despite retrieving both sources. The [separate 3b evaluation](docs/live-evaluation-3b-2026-11-01.json) failed 69/78 checks: it answered that compound question but falsely abstained on supported standard 30-day and updated 45-day refunds. Those unchanged failed artifacts remain inspectable.
- November 1 Chrome 156/macOS smoke passed 12 checks with the earlier 1.5b flow: actual imports, a supported answer, source resolution, unsupported abstention, a 390px layout and no JavaScript page errors. Desktop and narrow screenshots were reviewed. This is historical evidence for the earlier flow; November 5 evidence above verifies the current browser upload workflow. No claim about other engines or assistive technology is made.
- November 3 an earlier joint-decision candidate passed 107 main checks, but the [first additional evaluation](docs/additional-evaluation-first-2026-11-03.json) failed 80/104 checks across 12 cases, including five false abstentions. Further drafting/selection/review candidates also failed live checks. These failures motivated exact quote composition, source-free coverage and isolated support reviews. They are not acceptance evidence for the final November 4 implementation.

The current runs retain the main and additional expectations. Historical failures are not reclassified as successes, and the larger model is not assumed to be better independently of the tested workflow.
