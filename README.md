# Content Chat

A local support and FAQ chatbot for your own UTF-8 Markdown and text documents. SQLite stores document chunks and embedding vectors across restarts; exact cosine retrieval keeps collections separate. Local Ollama models supply embeddings and grounded answers with resolvable source citations.

**Build in progress. Real-model validation is pending:** this build environment has no installed Ollama models or running model server. Deterministic tests are regression evidence, not evidence of successful AI inference.

The first checkpoint implements persistent retrieval and a checked local-model adapter. CLI, local web interface, evaluation tooling and full usage documentation follow in the next checkpoints.

```sh
python3 -m unittest discover -v
```

No runtime Python dependencies. Python 3.11 or newer. No credentials, paid API usage, automatic model downloads or outbound messages.
