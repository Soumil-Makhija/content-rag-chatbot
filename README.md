# Content Chat

A local support and FAQ chatbot for your own documents. Import UTF-8 Markdown or text, retrieve relevant passages from a persistent SQLite vector store, and ask local Ollama models for answers with inspectable source citations. Use the CLI, a loopback JSON API, or the included browser interface.

**Delivery status: real-model validation is blocked.** This build environment has no Ollama runtime, installed embedding/language models, or reachable local model server. The application, 86 Python regression tests, and 3 JavaScript handler tests are implemented; inference in those tests uses explicit fixtures. No AI quality or live grounding results are claimed. See [validation and outstanding acceptance evidence](VALIDATION.md).

## Run locally

Requires Python 3.11+. The application has no Python runtime dependencies. From a checkout:

```sh
python3 -m content_chat --help
python3 -m content_chat doctor
```

Alternatively install into a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
content-chat --help
```

Inference requires an already-running [Ollama](https://ollama.com/download) server with an embedding model and a text-generation model. Defaults are [nomic-embed-text](https://ollama.com/library/nomic-embed-text) and [qwen2.5:3b](https://ollama.com/library/qwen2.5:3b). Model availability is checked by `doctor`; successful name checks do not prove inference quality. Models have their own licenses and hardware/storage requirements.

For local-only operation, start Ollama with cloud features disabled. If it is already running, configure/restart it using the [official local-only guidance](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features). This project does not change your server settings:

```sh
OLLAMA_NO_CLOUD=1 ollama serve
```

In another terminal, if you choose the defaults and have sufficient disk and memory, install them yourself:

```sh
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

The app never downloads models or configures billing. It requests only loopback Ollama URLs, disables redirects/environment proxies, and rejects obvious `:cloud`/`-cloud` model names. A local server can still proxy remote inference or use aliases; the client cannot verify that boundary. **Keep server cloud features disabled and use installed local models** before importing documents. No hosted API spending or document transmission was performed during this build. Use current Ollama with the [batch embedding API](https://docs.ollama.com/api/embed) and [structured chat output](https://docs.ollama.com/capabilities/structured-outputs).

## Import, ask, and inspect

The included Acorn Workspace documents are fictional, public test data.

```sh
python3 -m content_chat --db demo.sqlite3 import support examples/support.md --id support
python3 -m content_chat --db demo.sqlite3 import support examples/billing.md --id billing
python3 -m content_chat --db demo.sqlite3 ask support "How do I export my workspace, and what is the standard refund window?"
python3 -m content_chat --db demo.sqlite3 search support "refund request window" --limit 3
python3 -m content_chat --db demo.sqlite3 list support
python3 -m content_chat --db demo.sqlite3 show support billing
```

An answer object contains `supported`, `answer`, `citations`, and `models`. Each citation includes its document ID, title, version, chunk ordinal, exact passage text, source ID, collection, and cosine score. Resolve a source ID from that output:

```sh
python3 -m content_chat --db demo.sqlite3 source support SOURCE_ID
```

The expected fictional facts are a 14-day trial, export through Settings → Export, and a 30-day standard annual refund request window. These are evaluation expectations, not an example of a validated real-model answer. Questions such as bank refund processing time are unsupported by the supplied corpus.

When sources are insufficient, the model must return `supported:false`; the app responds: “I don't have enough information in this collection to answer that question.” Empty collections abstain without inference. An unavailable or malformed model produces an error rather than a substituted answer.

Change a file and reimport it with the **same `--id`** to replace the document. Imports embed at most 32 chunks per request and validate every batch before writing. Replacement increments its version and atomically replaces all chunks/vectors after every batch succeeds. Without `--id`, every import adds a new document, even if its title matches. Failed embeddings preserve the previous document.

```sh
python3 -m content_chat --db demo.sqlite3 import support examples/billing.md --id billing
python3 -m content_chat --db demo.sqlite3 delete support billing
```

Replaced/deleted source IDs stop resolving. Deletion is immediate; make your own database backups. Other collections are unaffected. The SQLite file persists across process and server restarts.

Global options precede the command:

```sh
python3 -m content_chat --db demo.sqlite3 \
  --ollama-url http://127.0.0.1:11434 \
  --embedding-model nomic-embed-text \
  --generation-model qwen2.5:3b \
  --timeout 180 ask support "How long is the trial?"
```

Commands emit JSON; errors go to stderr and exit 2. `doctor` exits 2 when models are missing/unavailable. `list`, `show`, `source`, and `delete` need no model access. Empty `ask`/`search` need none either.

## Browser interface and API

```sh
python3 -m content_chat --db demo.sqlite3 serve --port 8765
```

Open `http://127.0.0.1:8765`. Select a collection and click **Open collection**, upload a `.md`/`.txt` file, optionally select an existing document to replace, and ask a question. Expand supporting passages or open their source records. Import/delete clears the previous displayed answer. Editing the collection field clears the displayed library, replacement choices, and answer; open the new collection to continue. Stale delete controls cannot mutate the previously opened collection. Stop the server with Ctrl-C.

The server binds `127.0.0.1` only. It checks Host/Origin, accepts JSON writes, serves only packaged assets, and includes browser content restrictions. It is a single-user local tool with no authentication. Collections isolate retrieval; they do not give mutually untrusted users separate permissions. Other programs on this computer can access the local API. Keep Ollama itself configured for local use.

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/api/health` | App status and installed model-name readiness |
| GET | `/api/collections/COLLECTION/documents` | List document metadata |
| POST | `/api/collections/COLLECTION/documents` | Import `{title,text,document_id?}` |
| GET | `/api/collections/COLLECTION/documents/ID` | Original document and version |
| DELETE | `/api/collections/COLLECTION/documents/ID` | Remove document and vectors |
| POST | `/api/collections/COLLECTION/search` | Retrieve `{question,limit?}` |
| POST | `/api/collections/COLLECTION/ask` | Answer `{question,limit?}` with source URLs |
| GET | `/api/collections/COLLECTION/sources/ID` | Resolve the exact stored passage |

```sh
curl http://127.0.0.1:8765/api/collections/support/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How long is the trial?"}'
```

Invalid input returns 400; missing documents/sources return 404; malformed model output returns 502; unavailable models return 503. `/api/health` returns HTTP 200 for the app even when its `models.ready` is false. HTTP import uses extracted text in JSON, not multipart uploads. The browser decodes files locally before sending text to this local API.

## Tests and real-model evaluation

```sh
python3 -m unittest discover -v
python3 -m compileall -q content_chat tests tools
node --check content_chat/web/app.js
node --test tests/web-state.test.js  # Node 18+; no npm dependencies
python3 tools/evaluate_live.py --output local-evaluation.json
```

Tests exercise exact cosine ranking, restart persistence, bounded embedding batches and late-batch rollback, concurrent model changes, replacements/deletion, collection isolation, stale retrieved context, citation validation, CLI subprocess use, real local HTTP requests, truncated model HTTP responses, collection control safety, and the evaluation harness. All test inference is explicitly synthetic. CI runs Python 3.11/3.14 tests, packaging, entry-point smoke checks, JavaScript syntax and handler tests; it does not download/run models or send messages.

The separate evaluation script calls real configured local models on synthetic FAQ documents, including six unrelated support documents. It closes/reopens the vector store, runs three selective retrieval probes at top-k 1/1/2 against eight support passages, and runs nine answer/lifecycle cases with at most two retrieved passages each. It checks answerable/multi-document/unsupported questions, conflicting policies in a separate collection, document replacement, and deletion. Output records retrieved passages and exact scores, corpus counts, expected document IDs, actual answers/citations, configured models, timings and individual checks. A retrieval check fails if the corpus fits entirely inside top-k, so a trivial whole-corpus prompt cannot pass that check. It never modifies your existing database.

Exit 2 means access is blocked, 1 means checks/execution failed, and 0 means lexical/structural checks passed and **manual factual review is still required**. Expected phrases and valid citations do not prove an answer is correct. Inspect every answer and its supporting passages, particularly unsupported questions. Options `--ollama-url`, `--embedding-model`, and `--generation-model` select installed models.

## Supported scope and limits

- UTF-8 `.txt`/`.md` documents, up to 2 MiB of encoded text each; no PDFs, office formats, HTML parsing or OCR. NUL/binary/invalid UTF-8 input is rejected.
- Collections use lowercase slugs starting with a letter, 1–64 characters. Questions are at most 4000 characters. Answer retrieval defaults to 5 passages, maximum 10; standalone search allows 100.
- Character windows are 1200 characters with 150 overlap; whitespace-only windows are omitted. Windows can split sentences; there is no tokenizer-aware chunking. Embedding truncation is disabled so oversized model input fails explicitly. Model context limits remain provider-dependent.
- SQLite stores normalized vectors and performs exact cosine search in Python, linear in the selected collection's chunks. This is a small local vector database, without approximate-neighbor indexing or a total corpus quota. Data is plaintext at rest.
- Embedding identity is tracked by model name and dimensions, not a weight digest. Do not replace a model tag's weights beneath an existing collection. Use a new collection or delete/reimport all documents when changing the embedding space. The generation model can change independently.
- Every question is independent; there is no conversational memory. Only retrieved passages are supplied to generation, so missing a relevant passage can cause abstention.
- Every retrieved passage is rechecked after generation, including uncited passages and abstention context; changed/deleted context requires a retry. This check does not detect newly added or unretrieved documents or freeze the corpus after a response. Citation validation proves the cited passage exists in the requested collection. It cannot prove entailment, prevent all hallucinations, or guarantee resistance to malicious instructions in imported documents. Documents are prompted as untrusted evidence; generated text is rendered as text in the UI.
- The HTTP server handles one request at a time. Long model calls block other requests until completion/timeout. No multi-user hosting, remote providers, account connectors or outbound messaging are included.
- Live Chrome smoke checks confirmed the synthetic library renders and editing the collection field clears stale document controls. Further browser interaction was blocked by an automatic-review timeout, so full visual QA remains unverified. HTTP and JavaScript handler regressions pass. Real model-backed acceptance remains outstanding.

[Architecture](ARCHITECTURE.md) · [Validation](VALIDATION.md) · [MIT application license](LICENSE)
