"""HybridQA loader: questions, tables, linked passages, and gold answer locations.

Data layout (downloaded under ``data/hybridqa/``):
* ``dev.json`` / ``train.json`` -- questions. Train carries ``answer-node`` (the
  gold cell/passage where the answer lives); dev carries only ``answer-text``.
* ``WikiTables-WithLinks-master/tables_tok/{table_id}.json`` -- a table with
  ``header`` (list of ``[name, links]``) and ``data`` (rows of ``[text, links]`` cells).
* ``WikiTables-WithLinks-master/request_tok/{table_id}.json`` -- ``{/wiki/...: passage}``.

The retrieval/verification "unit" is a cell, a column header, or a linked passage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "hybridqa"
CORPUS_ROOT = DATA_ROOT / "WikiTables-WithLinks-master"


@dataclass
class Unit:
    kind: str          # "cell" | "column" | "passage"
    uid: str           # stable id within a table
    text: str          # the text to embed / read
    table_id: str
    row: int = -1
    col: int = -1
    links: list[str] = field(default_factory=list)


def questions(split: str = "dev") -> list[dict]:
    path = DATA_ROOT / f"{split}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=4096)
def load_table(table_id: str) -> dict:
    with (CORPUS_ROOT / "tables_tok" / f"{table_id}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=4096)
def load_passages(table_id: str) -> dict[str, str]:
    path = CORPUS_ROOT / "request_tok" / f"{table_id}.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _header_names(table: dict) -> list[str]:
    out = []
    for h in table.get("header", []):
        out.append(h[0] if isinstance(h, (list, tuple)) else str(h))
    return out


def _cell(cell) -> tuple[str, list[str]]:
    """Normalize a cell to (text, links). Cells are ``[text, [links]]``."""
    if isinstance(cell, (list, tuple)):
        text = cell[0] if cell else ""
        links = cell[1] if len(cell) > 1 and isinstance(cell[1], list) else []
        return str(text), [str(x) for x in links]
    return str(cell), []


def table_units(table_id: str, *, include_passages: bool = True) -> list[Unit]:
    """All retrieval/verification units for a table: columns, cells, passages."""
    table = load_table(table_id)
    title = table.get("title", "")
    section = table.get("section_title", "")
    headers = _header_names(table)
    units: list[Unit] = []

    for col, name in enumerate(headers):
        units.append(Unit("column", f"col:{col}", f"{title} | column: {name}", table_id, -1, col))

    for r, row in enumerate(table.get("data", [])):
        for col, cell in enumerate(row):
            text, links = _cell(cell)
            if not text:
                continue
            header = headers[col] if col < len(headers) else f"col{col}"
            ctx = f"{title} [{section}] {header} = {text}"
            units.append(Unit("cell", f"cell:{r}:{col}", ctx, table_id, r, col, links))

    if include_passages:
        for link, passage in load_passages(table_id).items():
            units.append(Unit("passage", f"psg:{link}", passage, table_id, -1, -1, [link]))
    return units


def linearize_table(table_id: str, *, max_rows: int = 30) -> str:
    """A compact textual view of the table for an LLM planner/reader prompt."""
    table = load_table(table_id)
    headers = _header_names(table)
    lines = [f"Title: {table.get('title','')}", "Columns: " + " | ".join(headers)]
    for r, row in enumerate(table.get("data", [])[:max_rows]):
        cells = [text for text, _links in (_cell(c) for c in row)]
        lines.append(f"r{r}: " + " | ".join(cells))
    return "\n".join(lines)


_WS = re.compile(r"\s+")


def normalize(text: object) -> str:
    return _WS.sub(" ", str(text).strip().lower())


def answer_bearing_units(question: dict, units: list[Unit]) -> list[Unit]:
    """Distant supervision: units whose text contains the answer string.

    The released splits give only ``answer-text`` (no gold answer location), so we
    label a unit positive when the normalized answer string occurs in it. This is the
    standard weak-supervision signal for HybridQA-style evidence localisation.
    """
    answer = normalize(question.get("answer-text", ""))
    if not answer:
        return []
    return [u for u in units if answer and answer in normalize(u.text)]


def corpus_available() -> bool:
    return (CORPUS_ROOT / "tables_tok").exists() and (DATA_ROOT / "dev.json").exists()
