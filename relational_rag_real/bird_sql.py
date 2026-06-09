"""End-to-end text-to-SQL harness for BIRD: schema serialization, SQL extraction,
and Execution Accuracy (set-compare of result rows, the official BIRD criterion)."""

from __future__ import annotations

import re
import sqlite3

from . import bird as B

_SQL_START = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)


def serialize_schema(db_id: str, *, sample_vals: int = 4, max_card: int = 40, tables=None) -> str:
    """Tables + columns; with ``sample_vals>0`` also list a few distinct values for
    low-cardinality text columns. ``tables`` (a set of names) restricts to a pruned,
    schema-linked subset."""
    sch = B.schema(db_id)
    keep = {t.lower() for t in tables} if tables else None
    lines = []
    for table, cols in sch.items():
        if keep is not None and table.lower() not in keep:
            continue
        lines.append(f"Table {table} ( {', '.join(cols)} )")
        if sample_vals <= 0:
            continue
        for col in cols:
            vals = B.column_values(db_id, table, col, cap=max_card + 1)
            text_vals = [v for v in vals if not v.replace(".", "").replace("-", "").isdigit()]
            if 0 < len(text_vals) <= max_card and any(not v.isdigit() for v in text_vals):
                shown = ", ".join(repr(v) for v in text_vals[:sample_vals])
                lines.append(f"  {table}.{col} values: {shown}")
    return "\n".join(lines)


def build_prompt(db_id: str, question: str, evidence: str, *,
                 include_values: bool = True, hints: list[str] | None = None, tables=None) -> str:
    schema = serialize_schema(db_id, sample_vals=4 if include_values else 0, tables=tables)
    ext = f"\nExternal knowledge: {evidence}" if evidence else ""
    vh = ("\nLikely value filters (from value linking): " + "; ".join(hints)) if hints else ""
    return (
        "You are an expert SQLite analyst. Using ONLY the schema below, write a single "
        "valid SQLite query that answers the question. Use exact column names in "
        "backticks if they contain spaces. Output ONLY the SQL, no explanation.\n\n"
        f"Schema:\n{schema}{ext}{vh}\n\nQuestion: {question}\nSQL:"
    )


def extract_sql(text: str) -> str:
    t = re.sub(r"```sql|```", " ", text)
    t = t.split("<|")[0]                      # strip chat special tokens (<|im_end|>, ...)
    t = re.sub(r"<\|.*?\|>", " ", t).strip()
    m = _SQL_START.search(t)
    if m:
        t = t[m.start():]
    if ";" in t:
        t = t[: t.index(";") + 1]
    return " ".join(t.split())


def execute(db_id: str, sql: str, timeout: float = 8.0):
    try:
        con = sqlite3.connect(f"file:{B.db_path(db_id)}?mode=ro", uri=True, timeout=timeout)
        con.text_factory = lambda b: b.decode("utf-8", "ignore")
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        con.close()
        return rows, None
    except Exception as exc:  # noqa: BLE001 - any SQL/DB error means a failed prediction
        return None, str(exc)


def execution_match(db_id: str, pred_sql: str, gold_sql: str) -> bool:
    pred, perr = execute(db_id, pred_sql)
    gold, gerr = execute(db_id, gold_sql)
    if perr is not None or gerr is not None or gold is None:
        return False
    return {tuple(r) for r in pred} == {tuple(r) for r in gold}
