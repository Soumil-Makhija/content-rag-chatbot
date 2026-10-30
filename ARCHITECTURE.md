# Design and acceptance boundaries

Documents are user-supplied UTF-8 `.txt` and `.md` files. Each collection uses a single embedding model and vector dimension. Character windows with overlap retain original text, document identity, version and ordinal. Source IDs identify a specific chunk version; replacement and deletion invalidate stale source URLs.

The SQLite database holds original documents, chunks and embedding vectors. Retrieval scans only the selected collection, calculates exact cosine similarity, and uses stable tie breaking. This is a persistent vector database with linear search, intended for small local FAQ corpora rather than millions of chunks. A configurable top-k supplies sources to the generator. Similarity does not establish whether a question is answerable.

Ollama provides local embeddings and JSON generation. The application rejects remote endpoints, redirects, invalid vectors, malformed generation and citations outside the retrieved set. The generator must abstain when passages do not support a question. An accepted citation proves that the referenced passage exists; it cannot mechanically prove the generated claim follows from it. Imported document text is untrusted evidence, never a trusted instruction.

Import embeds the complete replacement before a database transaction, preserving an existing document when the model fails. Collection model checks prevent mixing vector spaces; deleting the final document permits selecting a new model. All source and document lookups include collection scope.

The planned local HTTP interface binds loopback only, validates Host and Origin, uses JSON requests and resolves source links through database IDs rather than filesystem paths. It has no user authentication and is a single-user local tool. Collection separation is an application boundary, not an access-control system for mutually untrusted users.

## Completion evidence

Automated regression tests must exercise persistent retrieval, atomic lifecycle changes, scoped citation resolution, CLI and HTTP error paths. A separate live evaluation must call an installed embedding model and language model on synthetic FAQ documents, reopen the database, check answerable/unsupported questions and inspect source-grounded answers. No model runtime is installed in the current build environment; fixture tests do not satisfy that live-inference requirement.
