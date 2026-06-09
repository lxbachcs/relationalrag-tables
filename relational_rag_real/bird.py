"""BIRD loader + value-linking extraction.

The on-thesis task in BIRD: a question entity (e.g. "direct charter-funded") must be
linked to the correct database value (e.g. ``'Directly funded'``) so the generated
SQL filters on the right literal. Exact/substring matching fails on such drifted
values; a learned verifier over identifier-style features should recover them.

We parse gold SQL for ``col <op> 'value'`` conditions, confirm the (table, column)
that actually holds the value, and expose the column's distinct values as the
candidate pool for linking.
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "bird" / "dev_20240627"
DB_ROOT = DATA_ROOT / "dev_databases"

_WS = re.compile(r"\s+")
# col <op> 'value' ; col may be `quoted`, "quoted", [bracketed], or alias.name
_COND = re.compile(r"""(`[^`]+`|"[^"]+"|\[[^\]]+\]|[A-Za-z_][\w.]*)\s*(=|LIKE|like|In|IN)\s*'([^']*)'""")


def normalize(text: object) -> str:
    return _WS.sub(" ", str(text).strip().lower())


def questions() -> list[dict]:
    import json

    with (DATA_ROOT / "dev.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def db_path(db_id: str) -> Path:
    return DB_ROOT / db_id / f"{db_id}.sqlite"


@lru_cache(maxsize=64)
def _conn(db_id: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "ignore")
    return con


@lru_cache(maxsize=64)
def schema(db_id: str) -> dict[str, list[str]]:
    cur = _conn(db_id).cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    out = {}
    for t in tables:
        try:
            out[t] = [r[1] for r in cur.execute(f'PRAGMA table_info("{t}")')]
        except sqlite3.Error:
            out[t] = []
    return out


@lru_cache(maxsize=4096)
def column_values(db_id: str, table: str, col: str, cap: int = 4000) -> tuple[str, ...]:
    cur = _conn(db_id).cursor()
    try:
        rows = cur.execute(
            f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT {cap}'
        ).fetchall()
    except sqlite3.Error:
        return ()
    return tuple(str(r[0]) for r in rows if r[0] is not None and str(r[0]).strip())


def extract_conditions(sql: str) -> list[tuple[str, str]]:
    """Return [(column_name, string_value)] from ``col <op> 'value'`` conditions."""
    out = []
    for col, _op, val in _COND.findall(sql):
        col = col.strip('`"[]')
        if "." in col:
            col = col.split(".")[-1]
        v = val.strip()
        digitish = v.replace(".", "").replace("-", "").replace("%", "").replace("/", "")
        if v and not digitish.isdigit() and len(v) >= 2 and "%" not in v[:2]:
            out.append((col, v))
    return out


def value_link_instances(qset: list[dict], *, max_instances: int = 100000) -> list[dict]:
    """Build value-linking instances: (question, gold value, candidate value pool)."""
    inst: list[dict] = []
    for q in qset:
        conds = extract_conditions(q["SQL"])
        if not conds:
            continue
        sch = schema(q["db_id"])
        for col, val in conds:
            target = None
            for table, cols in sch.items():
                lower = {c.lower(): c for c in cols}
                if col.lower() not in lower:
                    continue
                real = lower[col.lower()]
                vals = column_values(q["db_id"], table, real)
                if any(normalize(v) == normalize(val) for v in vals):
                    target = (table, real, vals)
                    break
            if target is None:
                continue
            table, real, vals = target
            if len(vals) < 2:
                continue
            inst.append({
                "question": q["question"], "evidence": q.get("evidence", ""),
                "db_id": q["db_id"], "table": table, "column": real,
                "gold": val, "candidates": list(vals),
                "drift": normalize(val) not in normalize(q["question"]),
            })
            if len(inst) >= max_instances:
                return inst
    return inst


def available() -> bool:
    return (DATA_ROOT / "dev.json").exists() and DB_ROOT.exists()
