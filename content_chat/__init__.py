"""A local, persistent retrieval-augmented content assistant."""

from .store import Store, chunk_text

__all__ = ["Store", "chunk_text"]
