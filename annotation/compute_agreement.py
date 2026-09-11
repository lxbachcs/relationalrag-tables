#!/usr/bin/env python3
"""Validate the annotation package and compute two-annotator audit statistics.

This script uses only the Python standard library and resolves every input path
relative to its own directory, so it can be run from any working directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_file: str
    immutable_fields: tuple[str, ...]
    annotation_fields: tuple[str, ...]
    categorical_field: str
    allowed_categories: frozenset[str]


SPECS = {
    "bird": DatasetSpec(
        name="bird",
        raw_file="annotation_sample_bird.csv",
        immutable_fields=(
            "id",
            "db_id",
            "question",
            "evidence",
            "table",
            "column",
            "protocol_gold",
            "protocol_drift",
            # offline context (added so the audit is executable without the database)
            "question_id",
            "gold_sql",
            "sql_tables",
            "sql_column_ref",
            "sql_column_table",
            "same_name_columns",
            "matching_cell_values",
            "column_distinct_count",
        ),
        annotation_fields=("annotator_gold", "annotator_drift", "note"),
        categorical_field="annotator_drift",
        allowed_categories=frozenset({"0", "1"}),
    ),
    "icd": DatasetSpec(
        name="icd",
        raw_file="annotation_sample_icd.csv",
        immutable_fields=(
            "id",
            "table_code",
            "table_desc",
            "protocol_gold_qid",
            "protocol_gold_code",
            "protocol_drift_type",
            # offline context (added so the audit is executable without an external code table)
            "record_code_in_icd10cm",
            "record_code_icd10cm_desc",
            "record_category_code",
            "record_category_title",
            "nearest_valid_icd10cm",
            "node_wikidata_icd9_claim",
            "node_all_icd10_codes",
            "n_valid_codes_under_node",
        ),
        annotation_fields=("annotator_gold_qid", "annotator_drift_type", "note"),
        categorical_field="annotator_drift_type",
        allowed_categories=frozenset({"hier", "corrupt", "xvocab", "none"}),
    ),
}


class AuditInputError(ValueError):
    """Raised when editable audit files no longer match the frozen samples."""


def norm(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def kappa(a: list[Any], b: list[Any]) -> float | None:
    """Return unweighted Cohen's kappa, or None for empty/degenerate data."""
    if len(a) != len(b):
        raise ValueError("kappa inputs must have equal length")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    categories = set(a) | set(b)
    pe = sum((a.count(category) / n) * (b.count(category) / n) for category in categories)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise AuditInputError(f"missing required file: {path.name}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def load_dataset(spec: DatasetSpec) -> dict[str, list[dict[str, str]]]:
    expected_header = list(spec.immutable_fields + spec.annotation_fields)
    header, raw = read_csv(HERE / spec.raw_file)
    if header != expected_header:
        raise AuditInputError(
            f"{spec.raw_file}: expected columns {expected_header}, found {header}"
        )
    if len(raw) != 100:
        raise AuditInputError(f"{spec.raw_file}: expected 100 rows, found {len(raw)}")
    ids = [row["id"] for row in raw]
    if len(ids) != len(set(ids)):
        raise AuditInputError(f"{spec.raw_file}: duplicate row IDs")

    loaded: dict[str, list[dict[str, str]]] = {"raw": raw}
    for annotator in "AB":
        filename = f"{spec.name}_annotator_{annotator}.csv"
        ann_header, rows = read_csv(HERE / filename)
        if ann_header != expected_header:
            raise AuditInputError(
                f"{filename}: expected columns {expected_header}, found {ann_header}"
            )
        if [row["id"] for row in rows] != ids:
            raise AuditInputError(f"{filename}: row IDs/order differ from {spec.raw_file}")
        for index, (source, row) in enumerate(zip(raw, rows), start=2):
            for field in spec.immutable_fields:
                if row[field] != source[field]:
                    raise AuditInputError(
                        f"{filename}:{index}: frozen field {field!r} differs from {spec.raw_file}"
                    )
            required = spec.annotation_fields[:2]
            present = [bool(row[field].strip()) for field in required]
            if any(present) and not all(present):
                raise AuditInputError(
                    f"{filename}:{index}: fill both {required[0]} and {required[1]}, or leave both blank"
                )
            category = norm(row[spec.categorical_field])
            if category and category not in spec.allowed_categories:
                allowed = ", ".join(sorted(spec.allowed_categories))
                raise AuditInputError(
                    f"{filename}:{index}: invalid {spec.categorical_field}={category!r}; use {allowed}"
                )
        loaded[annotator] = rows
    return loaded


def validate_reference(rows: list[dict[str, str]]) -> dict[str, Any]:
    header, reference = read_csv(HERE / "icd_wikidata_reference.csv")
    required = {
        "qid",
        "label_en",
        "description_en",
        "wikidata_icd10_codes",
        "protocol_codes_in_sample",
        "last_revision_id",
        "source_url",
        "retrieved_utc",
    }
    missing_columns = sorted(required - set(header))
    if missing_columns:
        raise AuditInputError(
            "icd_wikidata_reference.csv: missing columns " + ", ".join(missing_columns)
        )
    reference_qids = {row["qid"] for row in reference}
    sample_qids = {row["protocol_gold_qid"] for row in rows}
    missing_qids = sorted(sample_qids - reference_qids)
    extra_qids = sorted(reference_qids - sample_qids)
    if missing_qids or extra_qids:
        raise AuditInputError(
            "icd_wikidata_reference.csv QID mismatch; "
            f"missing={missing_qids}, extra={extra_qids}"
        )
    return {
        "rows": len(reference),
        "sample_qids_covered": len(sample_qids),
        "retrieved_utc": sorted({row["retrieved_utc"] for row in reference}),
    }


def validate_offline_context(bird_rows: list[dict[str, str]], icd_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Check the schema, column-value, and ICD-10-CM files that make the audit executable offline."""
    header, schema = read_csv(HERE / "bird_schema.csv")
    for field in ("db_id", "table", "column"):
        if field not in header:
            raise AuditInputError(f"bird_schema.csv: missing column {field!r}")
    columns = {(r["db_id"], r["table"].lower(), r["column"].lower()) for r in schema}
    databases = {r["db_id"] for r in schema}
    missing_db = sorted({r["db_id"] for r in bird_rows} - databases)
    if missing_db:
        raise AuditInputError("bird_schema.csv: no schema for databases " + ", ".join(missing_db))
    for row in bird_rows:
        if (row["db_id"], row["table"].lower(), row["column"].lower()) not in columns:
            raise AuditInputError(
                f"bird_schema.csv: protocol column {row['table']}.{row['column']} of row {row['id']} is not in the schema"
            )
        if not row["gold_sql"].strip():
            raise AuditInputError(f"annotation_sample_bird.csv row {row['id']}: gold_sql is empty")
    header, values = read_csv(HERE / "bird_column_values.csv")
    for field in ("id", "matching_cell_values", "sample_values"):
        if field not in header:
            raise AuditInputError(f"bird_column_values.csv: missing column {field!r}")
    if [r["id"] for r in values] != [r["id"] for r in bird_rows]:
        raise AuditInputError("bird_column_values.csv: row IDs/order differ from annotation_sample_bird.csv")
    header, codes = read_csv(HERE / "icd_icd10cm_reference.csv")
    for field in ("id", "protocol_gold_code", "node_code", "icd10cm_code", "icd10cm_desc"):
        if field not in header:
            raise AuditInputError(f"icd_icd10cm_reference.csv: missing column {field!r}")
    per_id: dict[str, int] = {}
    for r in codes:
        per_id[r["id"]] = per_id.get(r["id"], 0) + 1
    for row in icd_rows:
        if int(row["n_valid_codes_under_node"]) != per_id.get(row["id"], 0):
            raise AuditInputError(
                f"icd_icd10cm_reference.csv: row {row['id']} lists {per_id.get(row['id'], 0)} codes, "
                f"worksheet says {row['n_valid_codes_under_node']}"
            )
    return {
        "bird_schema_databases": len(databases),
        "bird_schema_columns": len(schema),
        "bird_column_value_rows": len(values),
        "icd10cm_reference_rows": len(codes),
    }


def completed(row: dict[str, str], spec: DatasetSpec) -> bool:
    return all(row[field].strip() for field in spec.annotation_fields[:2])


def read_resolutions(path: Path, resolution_fields: tuple[str, str]) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    _, rows = read_csv(path)
    return {
        row["id"]: row
        for row in rows
        if row.get("id") and any(row.get(field, "").strip() for field in resolution_fields)
    }


def write_adjudication(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    resolution_fields: tuple[str, str],
) -> list[dict[str, str]]:
    previous = read_resolutions(path, resolution_fields)
    for row in rows:
        old = previous.get(row["id"], {})
        for field in (*resolution_fields, "resolution_note"):
            row[field] = old.get(field, row.get(field, ""))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def audit_bird(loaded: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    spec = SPECS["bird"]
    common = [
        i for i in range(len(loaded["raw"])) if completed(loaded["A"][i], spec) and completed(loaded["B"][i], spec)
    ]
    counts = {ann: sum(completed(row, spec) for row in loaded[ann]) for ann in "AB"}
    if not common:
        write_adjudication(
            HERE / "adjudicate_bird.csv",
            [
                "id", "question", "protocol_gold", "protocol_drift", "A_gold", "B_gold",
                "A_drift", "B_drift", "resolved_gold", "resolved_drift", "resolution_note",
            ],
            [],
            ("resolved_gold", "resolved_drift"),
        )
        return {"status": "not_started", "completed_by_annotator": counts, "n_paired": 0}

    correct = {
        ann: [int(norm(loaded[ann][i]["annotator_gold"]) == norm(loaded[ann][i]["protocol_gold"])) for i in common]
        for ann in "AB"
    }
    drift = {ann: [norm(loaded[ann][i]["annotator_drift"]) for i in common] for ann in "AB"}
    drift_ok = {
        ann: [int(drift[ann][j] == norm(loaded[ann][i]["protocol_drift"])) for j, i in enumerate(common)]
        for ann in "AB"
    }
    needs_review: list[dict[str, str]] = []
    for j, i in enumerate(common):
        if (
            correct["A"][j] != correct["B"][j]
            or drift["A"][j] != drift["B"][j]
            or not correct["A"][j]
            or not correct["B"][j]
            or not drift_ok["A"][j]
            or not drift_ok["B"][j]
        ):
            source = loaded["raw"][i]
            needs_review.append(
                {
                    "id": source["id"],
                    "question": source["question"],
                    "protocol_gold": source["protocol_gold"],
                    "protocol_drift": source["protocol_drift"],
                    "A_gold": loaded["A"][i]["annotator_gold"],
                    "B_gold": loaded["B"][i]["annotator_gold"],
                    "A_drift": loaded["A"][i]["annotator_drift"],
                    "B_drift": loaded["B"][i]["annotator_drift"],
                    "resolved_gold": "",
                    "resolved_drift": "",
                    "resolution_note": "",
                }
            )
    adjudication = write_adjudication(
        HERE / "adjudicate_bird.csv",
        [
            "id", "question", "protocol_gold", "protocol_drift", "A_gold", "B_gold",
            "A_drift", "B_drift", "resolved_gold", "resolved_drift", "resolution_note",
        ],
        needs_review,
        ("resolved_gold", "resolved_drift"),
    )
    resolved = [row for row in adjudication if row["resolved_gold"].strip() and row["resolved_drift"].strip()]
    pending = len(adjudication) - len(resolved)
    adjudicated_gold_error = None
    adjudicated_drift_error = None
    if not pending:
        adjudicated_gold_error = round(
            sum(norm(row["resolved_gold"]) != norm(row["protocol_gold"]) for row in resolved) / len(common), 3
        )
        adjudicated_drift_error = round(
            sum(norm(row["resolved_drift"]) != norm(row["protocol_drift"]) for row in resolved) / len(common), 3
        )
    status = "complete" if len(common) == 100 and pending == 0 else ("pending_adjudication" if len(common) == 100 else "in_progress")
    return {
        "status": status,
        "completed_by_annotator": counts,
        "n_paired": len(common),
        "protocol_gold_error_rate": {ann: round(1 - sum(correct[ann]) / len(common), 3) for ann in "AB"},
        "protocol_drift_error_rate": {ann: round(1 - sum(drift_ok[ann]) / len(common), 3) for ann in "AB"},
        "kappa_protocol_correct": kappa(correct["A"], correct["B"]),
        "kappa_drift_label": kappa(drift["A"], drift["B"]),
        "adjudication": {
            "rows_requiring_review": len(adjudication),
            "resolved": len(resolved),
            "pending": pending,
            "protocol_gold_error_rate": adjudicated_gold_error,
            "protocol_drift_error_rate": adjudicated_drift_error,
        },
    }


def audit_icd(loaded: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    spec = SPECS["icd"]
    common = [
        i for i in range(len(loaded["raw"])) if completed(loaded["A"][i], spec) and completed(loaded["B"][i], spec)
    ]
    counts = {ann: sum(completed(row, spec) for row in loaded[ann]) for ann in "AB"}
    if not common:
        write_adjudication(
            HERE / "adjudicate_icd.csv",
            [
                "id", "table_code", "table_desc", "protocol_gold_qid", "protocol_drift_type",
                "A_qid", "B_qid", "A_type", "B_type", "resolved_qid", "resolved_type", "resolution_note",
            ],
            [],
            ("resolved_qid", "resolved_type"),
        )
        return {"status": "not_started", "completed_by_annotator": counts, "n_paired": 0}

    correct = {
        ann: [int(norm(loaded[ann][i]["annotator_gold_qid"]) == norm(loaded[ann][i]["protocol_gold_qid"])) for i in common]
        for ann in "AB"
    }
    drift_type = {ann: [norm(loaded[ann][i]["annotator_drift_type"]) for i in common] for ann in "AB"}
    type_ok = {
        ann: [int(drift_type[ann][j] == norm(loaded[ann][i]["protocol_drift_type"])) for j, i in enumerate(common)]
        for ann in "AB"
    }
    needs_review: list[dict[str, str]] = []
    for j, i in enumerate(common):
        if (
            correct["A"][j] != correct["B"][j]
            or drift_type["A"][j] != drift_type["B"][j]
            or not correct["A"][j]
            or not correct["B"][j]
            or not type_ok["A"][j]
            or not type_ok["B"][j]
        ):
            source = loaded["raw"][i]
            needs_review.append(
                {
                    "id": source["id"],
                    "table_code": source["table_code"],
                    "table_desc": source["table_desc"],
                    "protocol_gold_qid": source["protocol_gold_qid"],
                    "protocol_drift_type": source["protocol_drift_type"],
                    "A_qid": loaded["A"][i]["annotator_gold_qid"],
                    "B_qid": loaded["B"][i]["annotator_gold_qid"],
                    "A_type": loaded["A"][i]["annotator_drift_type"],
                    "B_type": loaded["B"][i]["annotator_drift_type"],
                    "resolved_qid": "",
                    "resolved_type": "",
                    "resolution_note": "",
                }
            )
    adjudication = write_adjudication(
        HERE / "adjudicate_icd.csv",
        [
            "id", "table_code", "table_desc", "protocol_gold_qid", "protocol_drift_type",
            "A_qid", "B_qid", "A_type", "B_type", "resolved_qid", "resolved_type", "resolution_note",
        ],
        needs_review,
        ("resolved_qid", "resolved_type"),
    )
    resolved = [row for row in adjudication if row["resolved_qid"].strip() and row["resolved_type"].strip()]
    pending = len(adjudication) - len(resolved)
    adjudicated_gold_error = None
    adjudicated_type_error = None
    if not pending:
        adjudicated_gold_error = round(
            sum(norm(row["resolved_qid"]) != norm(row["protocol_gold_qid"]) for row in resolved) / len(common), 3
        )
        adjudicated_type_error = round(
            sum(norm(row["resolved_type"]) != norm(row["protocol_drift_type"]) for row in resolved) / len(common), 3
        )
    status = "complete" if len(common) == 100 and pending == 0 else ("pending_adjudication" if len(common) == 100 else "in_progress")
    return {
        "status": status,
        "completed_by_annotator": counts,
        "n_paired": len(common),
        "protocol_gold_error_rate": {ann: round(1 - sum(correct[ann]) / len(common), 3) for ann in "AB"},
        "protocol_drift_type_error_rate": {ann: round(1 - sum(type_ok[ann]) / len(common), 3) for ann in "AB"},
        "kappa_protocol_correct": kappa(correct["A"], correct["B"]),
        "kappa_drift_type": kappa(drift_type["A"], drift_type["B"]),
        "adjudication": {
            "rows_requiring_review": len(adjudication),
            "resolved": len(resolved),
            "pending": pending,
            "protocol_gold_error_rate": adjudicated_gold_error,
            "protocol_drift_type_error_rate": adjudicated_type_error,
        },
    }


def validate_package() -> tuple[dict[str, dict[str, list[dict[str, str]]]], dict[str, Any]]:
    loaded = {name: load_dataset(spec) for name, spec in SPECS.items()}
    reference = validate_reference(loaded["icd"]["raw"])
    context = validate_offline_context(loaded["bird"]["raw"], loaded["icd"]["raw"])
    summary = {
        "status": "valid",
        "datasets": {
            name: {
                "sample_rows": len(data["raw"]),
                "annotator_A_rows": len(data["A"]),
                "annotator_B_rows": len(data["B"]),
            }
            for name, data in loaded.items()
        },
        "offline_icd_reference": reference,
        "offline_context": context,
        "external_python_dependencies": [],
    }
    return loaded, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate schemas, frozen fields, editable values, and offline references without writing outputs",
    )
    args = parser.parse_args()
    try:
        loaded, validation = validate_package()
        if args.validate_only:
            print(json.dumps(validation, indent=2, ensure_ascii=False))
            return 0
        result = {
            "package_validation": validation,
            "bird": audit_bird(loaded["bird"]),
            "icd": audit_icd(loaded["icd"]),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        (HERE / "agreement.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return 0
    except (AuditInputError, OSError, csv.Error, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
