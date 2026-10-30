# Content Chat

A local support and FAQ chatbot for your own UTF-8 Markdown and text documents. SQLite persists original documents, chunks and embedding vectors. Local Ollama models supply embeddings and answers with inspectable source passages.

**Real-model validation is blocked in the build environment:** no Ollama runtime/models or local model server are installed. Regression tests use explicit fixtures. The app has no heuristic fallback and does not claim verified AI answer quality.

Python 3.11+; no runtime Python dependencies. From this checkout:

```sh
python3 -m content_chat doctor
python3 -m content_chat import support examples/support.md --id support
python3 -m content_chat import support examples/billing.md --id billing
python3 -m content_chat ask support "How do I export my workspace, and what is the refund window?"
python3 -m content_chat serve
```

The import and answer commands require an already-running local Ollama server with the selected embedding and generation models installed. The defaults are `nomic-embed-text` and `qwen2.5:3b`; global flags `--embedding-model`, `--generation-model` and `--ollama-url` precede the command. No automatic downloads or paid API calls occur.

Open `http://127.0.0.1:8765` to upload documents, replace an existing document, ask questions, inspect citations and delete documents. The single-user web interface binds loopback only. Collection names are lowercase slugs; collections separate retrieval but are not accounts or access control.

```sh
python3 -m content_chat list support
python3 -m content_chat show support billing
python3 -m content_chat import support examples/billing.md --id billing
python3 -m content_chat delete support billing
python3 -m unittest discover -v
python3 tools/evaluate_live.py --output local-evaluation.json
```

Evaluation uses synthetic fictional FAQ documents and exits 2 with a recorded blocker when local models are unavailable. Passing automated checks requires manual review of every answer and cited passage. Fixtures and CI do not establish that real models work. Full operational documentation and acceptance evidence follow in the final integration checkpoint.

See [architecture](ARCHITECTURE.md). MIT license for application code; model licenses are separate.
