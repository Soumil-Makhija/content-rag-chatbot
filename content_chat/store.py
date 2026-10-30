"""Persistent exact cosine search for small, private content collections.

SQLite owns the documents, provenance, and vectors together. Search scans only
the selected collection; its cost is linear in that collection's chunk count.
Embeddings are supplied by the caller and must all use the same model signature
and dimensions. This module never calls a provider or re-embeds stored content.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Sequence
import uuid


_COLLECTION = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Split nonblank text into overlapping Unicode-character windows.

    The first and last whitespace are trimmed once. Interior text is preserved
    in each retained window; whitespace-only windows are skipped. Boundaries
    count Python characters rather than UTF-8 bytes. An empty or whitespace-only
    input has no chunks. ``size`` must exceed ``overlap``.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("chunk size must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < size:
        raise ValueError("chunk overlap must be an integer between zero and size - 1")
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        window = text[start:end]
        if window.strip():
            chunks.append(window)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _collection_name(collection: str) -> str:
    if not isinstance(collection, str) or not _COLLECTION.fullmatch(collection):
        raise ValueError("collection must be a lowercase slug of 1–64 characters, starting with a letter")
    return collection


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise ValueError(f"{name} must be a nonempty string of at most 1000 characters")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} cannot contain control characters")
    return value


def _unit_vector(vector: Iterable[float]) -> list[float]:
    if isinstance(vector, (str, bytes, dict)):
        raise ValueError("embedding must be a sequence of finite numbers")
    try:
        values = list(vector)
    except TypeError as exc:
        raise ValueError("embedding must be a sequence of finite numbers") from exc
    if not values or any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("embedding must be a nonempty sequence of finite numbers")
    try:
        values = [float(value) for value in values]
    except (ValueError, OverflowError) as exc:
        raise ValueError("embedding must contain finite numbers") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding must contain finite numbers")
    # Scaling before taking the norm handles both very small and very large
    # finite vectors without underflowing or overflowing their squared norm.
    scale = max(abs(value) for value in values)
    if scale == 0:
        raise ValueError("embedding must have nonzero length")
    scaled = [value / scale for value in values]
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return [value / norm for value in scaled]


class Store:
    """A SQLite-backed vector database with atomic document replacement."""

    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(str(path), timeout=15)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL CHECK (dimensions > 0)
            );
            CREATE TABLE IF NOT EXISTS documents (
                collection TEXT NOT NULL REFERENCES collections(name),
                id TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version > 0),
                PRIMARY KEY (collection, id)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                document_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                text TEXT NOT NULL,
                vector TEXT NOT NULL,
                FOREIGN KEY (collection, document_id)
                    REFERENCES documents(collection, id) ON DELETE CASCADE,
                UNIQUE (collection, document_id, ordinal)
            );
            CREATE INDEX IF NOT EXISTS chunks_collection ON chunks(collection);
            """
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def collection_model(self, collection: str) -> dict | None:
        collection = _collection_name(collection)
        row = self.connection.execute(
            "SELECT model, dimensions FROM collections WHERE name = ?", (collection,)
        ).fetchone()
        return dict(row) if row else None

    def put(
        self,
        collection: str,
        title: str,
        text: str,
        embeddings: Sequence[Iterable[float]],
        model: str,
        document_id: str | None = None,
    ) -> dict:
        """Insert or replace one document and return its persisted metadata.

        The caller must embed ``chunk_text(text)`` in order. Validation happens
        before writes. A replacement preserves its document ID, increments the
        version, and invalidates old chunk/source IDs in a single transaction.
        """
        collection = _collection_name(collection)
        title = _identifier(title, "title")
        model = _identifier(model, "model")
        document_id = _identifier(document_id, "document ID") if document_id is not None else uuid.uuid4().hex
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("document text cannot be blank")
        if isinstance(embeddings, (str, bytes, dict)):
            raise ValueError("one embedding is required for each document chunk")
        try:
            vectors = [_unit_vector(vector) for vector in embeddings]
        except TypeError as exc:
            raise ValueError("one embedding is required for each document chunk") from exc
        if len(vectors) != len(chunks):
            raise ValueError(f"expected {len(chunks)} embeddings, got {len(vectors)}")
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise ValueError("all document embeddings must have the same dimensions")

        # A write lock keeps checking the signature and mutating the collection
        # indivisible even when another process imports into this same database.
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            signature = self.collection_model(collection)
            if signature and signature != {"model": model, "dimensions": dimensions}:
                raise ValueError("collection embedding model or dimensions differ; use a separate collection or delete all its documents")
            if not signature:
                self.connection.execute(
                    "INSERT INTO collections(name, model, dimensions) VALUES (?, ?, ?)",
                    (collection, model, dimensions),
                )
            current = self.connection.execute(
                "SELECT version FROM documents WHERE collection = ? AND id = ?", (collection, document_id)
            ).fetchone()
            version = current["version"] + 1 if current else 1
            if current:
                self.connection.execute(
                    "DELETE FROM chunks WHERE collection = ? AND document_id = ?", (collection, document_id)
                )
                self.connection.execute(
                    "UPDATE documents SET title = ?, text = ?, version = ? WHERE collection = ? AND id = ?",
                    (title, text, version, collection, document_id),
                )
            else:
                self.connection.execute(
                    "INSERT INTO documents(collection, id, title, text, version) VALUES (?, ?, ?, ?, ?)",
                    (collection, document_id, title, text, version),
                )
            self.connection.executemany(
                "INSERT INTO chunks(id, collection, document_id, ordinal, text, vector) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (uuid.uuid4().hex, collection, document_id, ordinal, chunk, json.dumps(vector, allow_nan=False))
                    for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors))
                ],
            )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        return {"id": document_id, "title": title, "version": version, "chunk_count": len(chunks)}

    def list_documents(self, collection: str) -> list[dict]:
        collection = _collection_name(collection)
        rows = self.connection.execute(
            """SELECT d.id, d.title, d.version, COUNT(c.id) AS chunk_count
               FROM documents d LEFT JOIN chunks c
                 ON c.collection = d.collection AND c.document_id = d.id
               WHERE d.collection = ? GROUP BY d.id, d.title, d.version
               ORDER BY d.title COLLATE NOCASE, d.id""",
            (collection,),
        )
        return [dict(row) for row in rows]

    def get_document(self, collection: str, document_id: str) -> dict | None:
        collection = _collection_name(collection)
        document_id = _identifier(document_id, "document ID")
        row = self.connection.execute(
            """SELECT d.id, d.title, d.text, d.version, COUNT(c.id) AS chunk_count
               FROM documents d LEFT JOIN chunks c
                 ON c.collection = d.collection AND c.document_id = d.id
               WHERE d.collection = ? AND d.id = ?
               GROUP BY d.id, d.title, d.text, d.version""",
            (collection, document_id),
        ).fetchone()
        return dict(row) if row else None

    def delete(self, collection: str, document_id: str) -> bool:
        collection = _collection_name(collection)
        document_id = _identifier(document_id, "document ID")
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM documents WHERE collection = ? AND id = ?", (collection, document_id)
            )
            deleted = cursor.rowcount > 0
            # No live embeddings remain, so the next import may choose a new
            # signature. Other collections and their signatures are untouched.
            self.connection.execute(
                "DELETE FROM collections WHERE name = ? AND NOT EXISTS (SELECT 1 FROM documents WHERE collection = ?)",
                (collection, collection),
            )
        return deleted

    def search(self, collection: str, vector: Iterable[float], model: str, limit: int = 5) -> list[dict]:
        collection = _collection_name(collection)
        model = _identifier(model, "model")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("search limit must be an integer between 1 and 100")
        query = _unit_vector(vector)
        # The signature and vectors must come from one SQLite read snapshot.
        # Separate SELECTs would allow another process to empty and re-create
        # this collection with a different model between those reads.
        rows = self.connection.execute(
            """SELECT k.model AS embedding_model, k.dimensions AS embedding_dimensions,
                      c.id, c.document_id, d.title, d.version, c.ordinal, c.text, c.vector
               FROM collections k LEFT JOIN chunks c ON c.collection = k.name
               LEFT JOIN documents d
                 ON d.collection = c.collection AND d.id = c.document_id
               WHERE k.name = ?""",
            (collection,),
        ).fetchall()
        if not rows:
            return []
        if rows[0]["embedding_model"] != model or rows[0]["embedding_dimensions"] != len(query):
            raise ValueError("query embedding model or dimensions differ from the collection")
        results = []
        for row in rows:
            if row["id"] is None:
                continue
            source = dict(row)
            source.pop("embedding_model")
            source.pop("embedding_dimensions")
            stored = json.loads(source.pop("vector"))
            score = math.fsum(a * b for a, b in zip(query, stored))
            source["score"] = max(-1.0, min(1.0, score))
            results.append(source)
        results.sort(key=lambda item: (-item["score"], item["document_id"], item["ordinal"], item["id"]))
        return results[:limit]

    def source(self, collection: str, source_id: str) -> dict | None:
        collection = _collection_name(collection)
        source_id = _identifier(source_id, "source ID")
        row = self.connection.execute(
            """SELECT c.id, c.document_id, d.title, d.version, c.ordinal, c.text
               FROM chunks c JOIN documents d
                 ON d.collection = c.collection AND d.id = c.document_id
               WHERE c.collection = ? AND c.id = ?""",
            (collection, source_id),
        ).fetchone()
        return dict(row) if row else None
