"""Document import and grounded retrieval orchestration."""

from __future__ import annotations

from .models import ABSTENTION, ModelProtocolError, validate_answer, validate_embeddings
from .store import chunk_text

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_QUESTION_CHARS = 4000


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise ValueError(f"{name} must be a nonempty string of at most 1000 characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must use valid UTF-8 text encoding") from exc
    return value


def _question(question: str, limit: int, maximum_limit: int) -> str:
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_CHARS:
        raise ValueError("Question must contain 1–4000 characters.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum_limit:
        raise ValueError(f"Retrieval limit must be an integer between 1 and {maximum_limit}.")
    return question.strip()


class Service:
    """Combine a persistent Store with an embedding and generation provider.

    The model interface is intentionally small so deterministic fixtures can
    test orchestration separately from real-model evaluations. No heuristic
    fallback is substituted when model access fails.
    """

    def __init__(self, store, model):
        self.store = store
        self.model = model

    def _signature(self, collection: str) -> dict | None:
        signature = self.store.collection_model(collection)
        if signature and signature["model"] != self.model.embedding_model:
            raise ValueError(
                f"Collection uses embedding model {signature['model']!r}; the configured model is "
                f"{self.model.embedding_model!r}. Select the original model or use a separate collection."
            )
        return signature

    def _models(self) -> dict:
        return {"embedding": self.model.embedding_model, "generation": self.model.generation_model}

    def _abstain(self) -> dict:
        return {"supported": False, "answer": ABSTENTION, "citations": [], "models": self._models()}

    def import_document(
        self, collection: str, title: str, text: str, document_id: str | None = None
    ) -> dict:
        """Embed all chunks before inserting or atomically replacing a document."""
        signature = self._signature(collection)
        title = _identifier(title, "title")
        if document_id is not None:
            document_id = _identifier(document_id, "document ID")
        if not isinstance(text, str):
            raise ValueError("Document text must be a string.")
        if "\x00" in text:
            raise ValueError("Document contains NUL characters and is not supported text.")
        try:
            encoded_size = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("Document must use valid UTF-8 text encoding.") from exc
        if encoded_size > MAX_DOCUMENT_BYTES:
            raise ValueError("Document exceeds the 2 MiB limit.")
        chunks = chunk_text(text, size=1200, overlap=150)
        if not chunks:
            raise ValueError("document text cannot be blank")
        vectors = validate_embeddings(self.model.embed(chunks), len(chunks))
        if signature and signature["dimensions"] != len(vectors[0]):
            raise ValueError("Embedding dimensions differ from this collection; select the original embedding model.")
        return self.store.put(
            collection, title, text, vectors, self.model.embedding_model, document_id=document_id
        )

    def _retrieve(self, collection: str, question: str, limit: int) -> list[dict]:
        signature = self._signature(collection)
        if signature is None:
            return []
        vectors = validate_embeddings(self.model.embed([question]), 1)
        if signature["dimensions"] != len(vectors[0]):
            raise ValueError("Query embedding dimensions differ from this collection; select the original embedding model.")
        return self.store.search(collection, vectors[0], self.model.embedding_model, limit=limit)

    def search(self, collection: str, question: str, limit: int = 5) -> list[dict]:
        """Retrieve passages without generation, or return [] for an empty corpus."""
        question = _question(question, limit, maximum_limit=100)
        return self._retrieve(collection, question, limit)

    def ask(self, collection: str, question: str, limit: int = 5) -> dict:
        """Answer from at most ten chunks and return verified retrieved citations."""
        question = _question(question, limit, maximum_limit=10)
        sources = self._retrieve(collection, question, limit)
        if not sources:
            return self._abstain()
        by_id = {source["id"]: source for source in sources}
        # The provider is not trusted to validate itself, including fake/custom
        # providers used by an embedding host application.
        generated = validate_answer(self.model.generate(question, sources), set(by_id))
        # Every retrieved passage can influence generation, including passages
        # the model does not cite or uses to decide that evidence is missing.
        for retrieved in sources:
            current = self.store.source(collection, retrieved["id"])
            if current is None or any(current.get(key) != retrieved.get(key) for key in current):
                raise ModelProtocolError("Retrieved document context changed while the answer was generated. Ask the question again.")
        if not generated["supported"]:
            return self._abstain()
        citations = []
        for source_id in generated["citations"]:
            retrieved = by_id[source_id]
            citations.append({**retrieved, "collection": collection})
        return {
            "supported": True,
            "answer": generated["answer"],
            "citations": citations,
            "models": self._models(),
        }
