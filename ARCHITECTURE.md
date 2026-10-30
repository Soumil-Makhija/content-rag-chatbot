# Design and acceptance boundaries

Documents are user-supplied UTF-8 `.txt` and `.md` files. Each collection uses a single embedding model and vector dimension. Character windows with overlap retain original text, document identity, version and ordinal. Source IDs identify a specific chunk version; replacement and deletion invalidate stale source URLs.

The SQLite database holds original documents, chunks and embedding vectors. Retrieval scans only the selected collection, calculates exact cosine similarity, and uses stable tie breaking. This is a persistent vector database with linear search, intended for small local FAQ corpora rather than millions of chunks. A configurable top-k supplies sources to the generator. Similarity does not establish whether a question is answerable.

Ollama provides local embeddings and JSON generation. The application rejects remote endpoints, redirects, invalid vectors, malformed generation and citations outside the retrieved set. The generator must abstain when passages do not support a question. An accepted citation proves that the referenced passage exists; it cannot mechanically prove the generated claim follows from it. Imported document text is untrusted evidence, never a trusted instruction.

Import embeds the complete replacement before a database transaction, preserving an existing document when the model fails. Collection model checks prevent mixing vector spaces; deleting the final document permits selecting a new model. All source and document lookups include collection scope.

The local HTTP interface binds loopback only, validates Host and Origin, uses JSON requests and resolves source links through database IDs rather than filesystem paths. It has no user authentication and is a single-user local tool. Collection separation is an application boundary, not an access-control system for mutually untrusted users.

## Completion evidence

Automated regression tests must exercise persistent retrieval, atomic lifecycle changes, scoped citation resolution, CLI and HTTP error paths. The included separate live evaluation must call an installed embedding model and language model on synthetic FAQ documents, reopen the database, check answerable/unsupported questions and inspect source-grounded answers. No model runtime is installed in the current build environment; fixture tests do not satisfy that live-inference requirement.

## Modules and failure boundaries

- `store.py`: schema, chunk provenance, atomic writes and single-snapshot exact cosine reads.
- `models.py`: loopback Ollama HTTP protocol, embedding/output validation and JSON generation prompt.
- `service.py`: input limits, model consistency, import orchestration, retrieval, abstention and citation re-resolution.
- `cli.py` / `server.py` / `web/`: JSON CLI, single-threaded local API and browser client. Document content is rendered as text.
- `tools/evaluate_live.py`: disposable persistent corpus and actual-model evaluation; missing runtime reports blocked, never fixture success.

Vectors are normalized before storage. Signature and chunk reads use a single SQLite statement so another process cannot swap vector spaces between metadata checks and scoring. A cited chunk is re-resolved after generation; a concurrent update/delete returns a retry error. This cannot guarantee a source will remain unchanged after the response reaches a reader. Original text and vectors share the same database transaction during replacement.

Stored model signatures include the configured name and vector dimension. Weight digests and task-prefix configuration are not pinned. Replacing a model tag or changing its embedding behavior requires rebuilding the affected collection. Exact retrieval and character chunking support modest local corpora; this project does not claim scalable approximate indexing or tokenizer-aware context guarantees.

The loopback HTTP boundary does not prove local inference: Ollama can support cloud models or a server can proxy requests. Obvious `:cloud`/`-cloud` names are rejected, but aliases cannot be verified. Operators must disable server cloud features (for example `OLLAMA_NO_CLOUD=1` when starting Ollama) and use installed local models. The app does not change global settings or provision model access.
