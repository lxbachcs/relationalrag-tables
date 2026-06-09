"""Deterministic local vector indexing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .schemas import Corpus, IndexEntry

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
DIM = 64


def normalize_text(text: object) -> str:
    return " ".join(TOKEN_RE.findall(str(text).lower()))


def normalize_answer(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_identifier(text: object) -> str:
    return "".join(TOKEN_RE.findall(str(text).upper()))


def embed_text(text: str, dim: int = DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        tokens = [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def build_entries(corpus: Corpus) -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    for table in corpus.tables.values():
        columns = list(table.rows[0])
        for column in columns:
            text = f"{table.name} column {column}"
            entries.append(
                IndexEntry(
                    kind="column",
                    id=f"{table.name}.{column}",
                    text=text,
                    payload={"table": table.name, "column": column},
                    vector=embed_text(text),
                )
            )
        for row_idx, row in enumerate(table.rows):
            row_text = " ".join(f"{key} {value}" for key, value in row.items())
            for column, value in row.items():
                text = f"{table.name} row {row_idx} {column} {value} {row_text}"
                entries.append(
                    IndexEntry(
                        kind="cell",
                        id=f"{table.name}[{row_idx}].{column}",
                        text=text,
                        payload={
                            "table": table.name,
                            "row_index": row_idx,
                            "column": column,
                            "value": value,
                            "row": row,
                        },
                        vector=embed_text(text),
                    )
                )
    for idx, triple in enumerate(corpus.triples):
        text = f"{triple.subject} {triple.relation} {triple.object}"
        entries.append(
            IndexEntry(
                kind="edge",
                id=f"edge:{idx}",
                text=text,
                payload={
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object": triple.object,
                },
                vector=embed_text(text),
            )
        )
    return entries


def write_index(corpus: Corpus, out_dir: str | Path) -> list[IndexEntry]:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    entries = build_entries(corpus)
    by_kind = {
        "cells": [entry for entry in entries if entry.kind == "cell"],
        "columns": [entry for entry in entries if entry.kind == "column"],
        "edges": [entry for entry in entries if entry.kind == "edge"],
    }
    for name, items in by_kind.items():
        with (path / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump([_entry_to_json(item) for item in items], handle, indent=2)
    with (path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "backend": corpus.config.get("index_backend", "deterministic-local"),
                "embedding_dim": DIM,
                "counts": {name: len(items) for name, items in by_kind.items()},
            },
            handle,
            indent=2,
        )
    return entries


def load_index(index_dir: str | Path) -> list[IndexEntry]:
    path = Path(index_dir)
    entries: list[IndexEntry] = []
    for name in ("cells", "columns", "edges"):
        file_path = path / f"{name}.json"
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        with file_path.open(encoding="utf-8") as handle:
            entries.extend(_entry_from_json(item) for item in json.load(handle))
    return entries


def search_entries(entries: list[IndexEntry], query: str, kind: str, top_k: int) -> list[IndexEntry]:
    query_vector = embed_text(query)
    filtered = [entry for entry in entries if entry.kind == kind]
    scored = sorted(
        filtered,
        key=lambda entry: (cosine(query_vector, entry.vector), entry.id),
        reverse=True,
    )
    return scored[:top_k]


def _entry_to_json(entry: IndexEntry) -> dict[str, object]:
    return {
        "kind": entry.kind,
        "id": entry.id,
        "text": entry.text,
        "payload": entry.payload,
        "vector": entry.vector,
    }


def _entry_from_json(item: dict[str, object]) -> IndexEntry:
    return IndexEntry(
        kind=str(item["kind"]),
        id=str(item["id"]),
        text=str(item["text"]),
        payload=dict(item["payload"]),
        vector=[float(value) for value in item["vector"]],
    )
