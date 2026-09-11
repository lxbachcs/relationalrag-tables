#!/usr/bin/env python3
"""Maintainer command: regenerate the offline context that makes this audit executable
without a database or an external code table.

Annotators never run this. It needs the artifact checkout (``relational_rag_real`` and
``data/``), the BIRD dev SQLite databases, and the ICD-10-CM code table shipped with the
artifact. It writes:

  bird_schema.csv / bird_schema.md   every table and column of the 11 BIRD dev databases
                                     used by the sample (from ``dev_tables.json``);
  bird_column_values.csv             per BIRD row: distinct-count and a bounded sample of
                                     the cell values of the protocol column, plus every
                                     raw cell value that equals the protocol literal;
  icd_icd10cm_reference.csv          per ICD row: every valid ICD-10-CM code under the
                                     protocol node's code, with descriptions;

and it appends frozen context columns to the BIRD and ICD samples and to the four
annotator worksheets (gold SQL, tables in the SQL, the SQL column reference and the table
it resolves to, same-name columns across the schema, matching cell values; ICD-10-CM
validity and description of the record code, nearest valid codes, the node's Wikidata
ICD-9 claim). Existing annotation columns are preserved verbatim.

The frozen samples are re-derived from the same driver logic as ``run_e6_labelnoise``
(seed 0 shuffle, seed 1 sample) and every original frozen field is asserted equal to the
shipped sample before anything is written, so this command cannot silently change the
audited items.

Run: ``python3 build_offline_context.py --artifact-root /path/to/relationalrag-tables``
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

BIRD_CONTEXT = (
    "question_id",
    "gold_sql",
    "sql_tables",
    "sql_column_ref",
    "sql_column_table",
    "same_name_columns",
    "matching_cell_values",
    "column_distinct_count",
)
ICD_CONTEXT = (
    "record_code_in_icd10cm",
    "record_code_icd10cm_desc",
    "record_category_code",
    "record_category_title",
    "nearest_valid_icd10cm",
    "node_wikidata_icd9_claim",
    "node_all_icd10_codes",
    "n_valid_codes_under_node",
)
SAMPLE_VALUE_CAP = 60
NEAREST_CAP = 5
NEAREST_MAX_DIST = 2



def _strip(ident: str) -> str:
    return ident.strip().strip('`"[]')


def parse_from_tables(sql: str, schema: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    """Return (tables referenced in FROM/JOIN, alias -> table)."""
    known = {t.lower(): t for t in schema}
    tables: list[str] = []
    alias: dict[str, str] = {}
    for m in re.finditer(r"\b(?:FROM|JOIN)\s+([`\"\[]?)([A-Za-z_][^`\"\]\s,()]*)\1(?:\s+(?:AS\s+)?(?!ON\b|WHERE\b|INNER\b|LEFT\b|JOIN\b|GROUP\b|ORDER\b|LIMIT\b|WHERE\b)([A-Za-z_]\w*))?", sql, re.I):
        name = m.group(2)
        if name.lower() not in known:
            continue
        real = known[name.lower()]
        if real not in tables:
            tables.append(real)
        if m.group(3):
            alias[m.group(3).lower()] = real
    return tables, alias


def scope_tables(sql: str, pos: int, schema: dict[str, list[str]]) -> list[str]:
    """Tables named in FROM/JOIN of the SELECT scope (parenthesis level) enclosing ``pos``."""
    known = {t.lower(): t for t in schema}
    depth = [0] * (len(sql) + 1)
    d = 0
    for i, ch in enumerate(sql):
        depth[i] = d
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
    target = depth[pos]
    start = 0
    for i in range(pos, -1, -1):
        if depth[i] < target:
            start = i
            break
    out = []
    for m in re.finditer(r"\b(?:FROM|JOIN)\s+([`\"\[]?)([A-Za-z_][^`\"\]\s,()]*)\1", sql[start:pos], re.I):
        if depth[start + m.start()] != target:
            continue
        name = m.group(2).lower()
        if name in known and known[name] not in out:
            out.append(known[name])
    return out


def resolve_condition(sql: str, column: str, gold: str, schema: dict[str, list[str]], cond_re) -> tuple[str, str, list[str]]:
    """Locate ``col <op> 'gold'`` in the SQL and resolve which table it filters."""
    tables, alias = parse_from_tables(sql, schema)
    ref, pos = "", -1
    for m in cond_re.finditer(sql):
        col, val = m.group(1), m.group(3)
        bare = _strip(col).split(".")[-1]
        if bare.lower() == column.lower() and val.strip() == gold:
            ref, pos = _strip(col), m.start()
            break
    if not ref:
        return "", "UNRESOLVED", tables
    if "." in ref:
        qual = ref.split(".")[0].lower()
        if qual in alias:
            return ref, alias[qual], tables
        for t in schema:
            if t.lower() == qual:
                return ref, t, tables
        return ref, "UNRESOLVED", tables
    has_col = lambda t: column.lower() in {c.lower() for c in schema[t]}  # noqa: E731
    scoped = [t for t in scope_tables(sql, pos, schema) if has_col(t)]
    if len(scoped) == 1:
        return ref, scoped[0], tables
    cands = scoped or [t for t in tables if has_col(t)]
    if len(cands) == 1:
        return ref, cands[0], tables
    if not cands:
        return ref, "UNRESOLVED", tables
    return ref, "AMBIGUOUS: " + " | ".join(cands), tables


def levenshtein(a: str, b: str, cap: int) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            best = min(best, v)
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        r = csv.DictReader(fh)
        return list(r.fieldnames or []), list(r)


def write_rows(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in header})


def add_context(sample_file: str, worksheets: list[str], old_frozen: list[str], context_cols: tuple[str, ...],
                ann_cols: list[str], context: list[dict[str, str]]) -> None:
    """Append context columns to the frozen sample and each worksheet, keeping annotations."""
    for name in [sample_file, *worksheets]:
        path = HERE / name
        header, rows = read_rows(path)
        if len(rows) != len(context):
            raise SystemExit(f"{name}: expected {len(context)} rows, found {len(rows)}")
        new_header = old_frozen + list(context_cols) + ann_cols
        out = []
        for row, ctx in zip(rows, context):
            if row["id"] != ctx["id"]:
                raise SystemExit(f"{name}: row id {row['id']} does not match rebuilt id {ctx['id']}")
            merged = {k: row.get(k, "") for k in old_frozen + ann_cols}
            merged.update({k: ctx[k] for k in context_cols})
            out.append(merged)
        write_rows(path, new_header, out)
        print(f"wrote {name} ({len(out)} rows, {len(new_header)} columns)")


# ------------------------------------------------------------------------------ BIRD
def build_bird(root: Path) -> None:
    from relational_rag_real import bird as B

    qs = B.questions()
    inst = B.value_link_instances(qs, max_instances=10000)
    random.Random(0).shuffle(inst)
    sample = random.Random(1).sample(inst, min(100, len(inst)))
    by_q = {}
    for q in qs:
        by_q.setdefault((q["db_id"], q["question"]), []).append(q)

    old_header, frozen = read_rows(HERE / "annotation_sample_bird.csv")
    old_frozen = [c for c in old_header if c not in ("annotator_gold", "annotator_drift", "note")]
    old_frozen = [c for c in old_frozen if c not in BIRD_CONTEXT]
    for row, s in zip(frozen, sample):
        got = [s["db_id"], s["question"], s["evidence"], s["table"], s["column"], s["gold"], str(int(s["drift"]))]
        want = [row[k] for k in ("db_id", "question", "evidence", "table", "column", "protocol_gold", "protocol_drift")]
        if got != want:
            raise SystemExit(f"BIRD row {row['id']}: rebuilt instance differs from frozen sample; refusing to write")
    print("BIRD: rebuilt sample matches frozen sample on all 100 rows")

    # schema of the 11 databases from dev_tables.json (declared metadata, not extracted from the DB)
    dev_tables = {d["db_id"]: d for d in json.load((B.DATA_ROOT / "dev_tables.json").open(encoding="utf-8"))}
    db_ids = sorted({s["db_id"] for s in sample})
    schema_rows = []
    md = ["# BIRD dev schemas for the 11 databases in the audit sample", "",
          "Source: `dev_tables.json` from BIRD dev_20240627 (table and column names as declared; "
          "`original` names are what gold SQL uses). PK = primary key; FK = foreign key target.", ""]
    for db in db_ids:
        d = dev_tables[db]
        cols = d["column_names_original"]
        types = d["column_types"]
        pks = set()
        for p in d["primary_keys"]:
            pks.update(p if isinstance(p, list) else [p])
        fks = {}
        for a, b in d["foreign_keys"]:
            fks[a] = f"{d['table_names_original'][cols[b][0]]}.{cols[b][1]}"
        md.append(f"## {db}")
        md.append("")
        for ti, t in enumerate(d["table_names_original"]):
            md.append(f"### {t}")
            md.append("")
            md.append("| column | type | key |")
            md.append("|---|---|---|")
            for ci, (tix, cname) in enumerate(cols):
                if tix != ti:
                    continue
                key = ("PK" if ci in pks else "") + (f" FK->{fks[ci]}" if ci in fks else "")
                schema_rows.append({"db_id": db, "table": t, "column": cname, "type": types[ci],
                                    "primary_key": int(ci in pks), "foreign_key_to": fks.get(ci, "")})
                md.append(f"| {cname} | {types[ci]} | {key.strip()} |")
            md.append("")
    write_rows(HERE / "bird_schema.csv", ["db_id", "table", "column", "type", "primary_key", "foreign_key_to"], schema_rows)
    (HERE / "bird_schema.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote bird_schema.csv ({len(schema_rows)} columns over {len(db_ids)} databases) and bird_schema.md")

    context, value_rows = [], []
    for row, s in zip(frozen, sample):
        sch = B.schema(s["db_id"])
        matches = by_q[(s["db_id"], s["question"])]
        # if the question text is duplicated, keep the one whose SQL carries the condition
        q = next((m for m in matches if any(_strip(c).split(".")[-1].lower() == s["column"].lower() and v.strip() == s["gold"]
                                              for c, _o, v in B._COND.findall(m["SQL"]))), matches[0])
        ref, table, tables = resolve_condition(q["SQL"], s["column"], s["gold"], sch, B._COND)
        same = []
        for t, cs in sch.items():
            for c in cs:
                if c.lower() == s["column"].lower():
                    vals = B.column_values(s["db_id"], t, c)
                    present = any(B.normalize(v) == B.normalize(s["gold"]) for v in vals)
                    same.append(f"{t}.{c}=" + ("present" if present else "absent"))
        vals = B.column_values(s["db_id"], s["table"], s["column"])
        matching = sorted({v for v in vals if B.normalize(v) == B.normalize(s["gold"])})
        context.append({
            "id": row["id"],
            "question_id": str(q["question_id"]),
            "gold_sql": q["SQL"],
            "sql_tables": " | ".join(tables),
            "sql_column_ref": ref,
            "sql_column_table": table,
            "same_name_columns": "; ".join(same),
            "matching_cell_values": " | ".join(matching),
            "column_distinct_count": str(len(vals)),
        })
        sample_vals = list(vals[:SAMPLE_VALUE_CAP])
        for m in matching:
            if m not in sample_vals:
                sample_vals.append(m)
        value_rows.append({
            "id": row["id"], "db_id": s["db_id"], "table": s["table"], "column": s["column"],
            "protocol_gold": s["gold"], "distinct_count_capped_4000": str(len(vals)),
            "matching_cell_values": " | ".join(matching),
            "sample_values": " | ".join(sample_vals),
            "sample_truncated": str(int(len(vals) > SAMPLE_VALUE_CAP)),
        })
    write_rows(HERE / "bird_column_values.csv",
               ["id", "db_id", "table", "column", "protocol_gold", "distinct_count_capped_4000",
                "matching_cell_values", "sample_values", "sample_truncated"], value_rows)
    print("wrote bird_column_values.csv")
    add_context("annotation_sample_bird.csv", ["bird_annotator_A.csv", "bird_annotator_B.csv"], old_frozen,
                BIRD_CONTEXT, ["annotator_gold", "annotator_drift", "note"], context)
    unresolved = sum(1 for c in context if c["sql_column_table"].startswith(("UNRESOLVED", "AMBIGUOUS")))
    differs = sum(1 for c, s in zip(context, sample) if c["sql_column_table"] != s["table"])
    print(f"BIRD: {unresolved} unresolved/ambiguous SQL table references; {differs} rows where the SQL table differs from the protocol table")


# ------------------------------------------------------------------------------- ICD
def build_icd(root: Path) -> None:
    from relational_rag_real import icd_kg as K
    from relational_rag_real import run_icd_hetero as H

    recs, kg = H.build(seed=0)
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    sample = random.Random(1).sample(recs, min(100, len(recs)))
    old_header, frozen = read_rows(HERE / "annotation_sample_icd.csv")
    ann = ["annotator_gold_qid", "annotator_drift_type", "note"]
    old_frozen = [c for c in old_header if c not in ann and c not in ICD_CONTEXT]
    for row, r in zip(frozen, sample):
        got = [r.table_code, r.table_desc, r.gold_qid, kg[qi[r.gold_qid]]["norm"], getattr(r, "dtype", "hier")]
        want = [row[k] for k in ("table_code", "table_desc", "protocol_gold_qid", "protocol_gold_code", "protocol_drift_type")]
        if got != want:
            raise SystemExit(f"ICD row {row['id']}: rebuilt record differs from frozen sample; refusing to write")
    print("ICD: rebuilt sample matches frozen sample on all 100 rows")

    icd = K.load_icd10()
    desc_of = dict(icd)
    cat_title = {}
    with (K.DATA / "icd10.csv").open(encoding="utf-8", errors="ignore") as fh:
        for r in csv.reader(fh):
            if len(r) >= 6 and r[2]:
                cat_title.setdefault(K.norm(r[2])[:3], r[5])
    icd9 = {}
    for b in json.loads((K.DATA / "wikidata_icd9_icd10.json").read_text(encoding="utf-8"))["results"]["bindings"]:
        icd9.setdefault(b["item"]["value"].split("/")[-1], []).append(b["icd9"]["value"])

    # a QID can carry several ICD-10 codes; the KG keeps one entry per code, and the
    # protocol reports the last entry's code, so list valid codes under every code of the node
    codes_of: dict[str, list[str]] = {}
    for n in kg:
        codes_of.setdefault(n["qid"], []).append(n["norm"])
    context, ref_rows = [], []
    for row, r in zip(frozen, sample):
        node = kg[qi[r.gold_qid]]["norm"]
        node_codes = codes_of[r.gold_qid]
        under = []
        for nc in node_codes:
            under.extend((nc, c, d) for c, d in icd if c.startswith(nc))
        for nc, c, d in under:
            ref_rows.append({"id": row["id"], "protocol_gold_qid": r.gold_qid, "protocol_gold_code": node, "node_code": nc,
                             "icd10cm_code": c, "icd10cm_desc": d,
                             "desc_equals_record_desc": str(int(d.strip().lower() == r.table_desc.strip().lower()))})
        near = []
        for c, d in icd:
            dist = levenshtein(r.table_code, c, NEAREST_MAX_DIST)
            if dist <= NEAREST_MAX_DIST:
                near.append((dist, 0 if d.strip().lower() == r.table_desc.strip().lower() else 1, c, d))
        near.sort()
        valid = r.table_code in desc_of
        context.append({
            "id": row["id"],
            "record_code_in_icd10cm": "yes" if valid else "no",
            "record_code_icd10cm_desc": desc_of.get(r.table_code, ""),
            "record_category_code": r.table_code[:3],
            "record_category_title": cat_title.get(r.table_code[:3], ""),
            "nearest_valid_icd10cm": " | ".join(f"{c} d={dist} ({d})" for dist, _m, c, d in near[:NEAREST_CAP]),
            "node_wikidata_icd9_claim": " | ".join(sorted(set(icd9.get(r.gold_qid, [])))),
            "node_all_icd10_codes": " | ".join(node_codes),
            "n_valid_codes_under_node": str(len(under)),
        })
    write_rows(HERE / "icd_icd10cm_reference.csv",
               ["id", "protocol_gold_qid", "protocol_gold_code", "node_code", "icd10cm_code", "icd10cm_desc", "desc_equals_record_desc"],
               ref_rows)
    print(f"wrote icd_icd10cm_reference.csv ({len(ref_rows)} code rows)")
    add_context("annotation_sample_icd.csv", ["icd_annotator_A.csv", "icd_annotator_B.csv"], old_frozen,
                ICD_CONTEXT, ann, context)
    print("ICD: record code valid in ICD-10-CM for", sum(1 for c in context if c["record_code_in_icd10cm"] == "yes"), "of 100 rows")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact-root", required=True, type=Path, help="checkout containing relational_rag_real/ and data/")
    ap.add_argument("--only", choices=["bird", "icd"], default=None)
    args = ap.parse_args()
    root = args.artifact_root.resolve()
    if not (root / "relational_rag_real").is_dir():
        print(f"ERROR: {root} has no relational_rag_real/ package", file=sys.stderr)
        return 2
    sys.path.insert(0, str(root))
    if args.only in (None, "bird"):
        build_bird(root)
    if args.only in (None, "icd"):
        build_icd(root)
    print("Now update SPECS in compute_agreement.py if the context columns changed, run verify_package.py, "
          "and regenerate SOURCE_MANIFEST.sha256.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
