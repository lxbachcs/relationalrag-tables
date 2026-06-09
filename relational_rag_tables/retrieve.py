"""Retrieval over cell, column, and edge indexes."""

from __future__ import annotations

from .index import search_entries
from .schemas import IndexEntry, Retrieved


def retrieve(entries: list[IndexEntry], question: str, top_k: int) -> Retrieved:
    return Retrieved(
        cells=search_entries(entries, question, "cell", top_k),
        columns=search_entries(entries, question, "column", top_k),
        edges=search_entries(entries, question, "edge", top_k),
    )


def retrieved_values(retrieved: Retrieved) -> set[str]:
    values: set[str] = set()
    for entry in [*retrieved.cells, *retrieved.columns, *retrieved.edges]:
        values.add(entry.text)
        for value in entry.payload.values():
            if isinstance(value, str):
                values.add(value)
            elif isinstance(value, dict):
                values.update(str(item) for item in value.values())
    return values
