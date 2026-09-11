#!/usr/bin/env python3
"""Re-run BIRD value linking after resolving SQL aliases before column lookup.

This is a maintainer command, not part of the offline annotation workflow.  It applies
the adjudicated BIRD correction to a checkout of the released experiment harness without
modifying that checkout, runs ``run_bird.main`` for seeds 0--4, and writes the aggregate
to ``protocol_correction_impact.json`` next to this script.

Run:
    python3 recompute_bird_protocol_impact.py --artifact-root /path/to/relationalrag-tables

The artifact checkout must have BIRD dev_20240627 installed at the path expected by
``relational_rag_real.bird`` and the real-data Python dependencies available.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_offline_context import resolve_condition


HERE = Path(__file__).resolve().parent


def aggregate(values: list[float]) -> dict[str, object]:
    return {
        "mean": round(float(np.mean(values)), 4),
        "sd": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "protocol_correction_impact.json")
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    import sys

    sys.path.insert(0, str(root))
    from relational_rag_real import bird as bird_data
    from relational_rag_real import run_bird
    from relational_rag_real import run_e6_labelnoise

    original_instances = bird_data.value_link_instances(bird_data.questions())

    def corrected_instances(qset: list[dict], *, max_instances: int = 100000) -> list[dict]:
        instances: list[dict] = []
        for question in qset:
            schema = bird_data.schema(question["db_id"])
            for column, value in bird_data.extract_conditions(question["SQL"]):
                _, table, _ = resolve_condition(
                    question["SQL"], column, value, schema, bird_data._COND
                )
                if table not in schema:
                    continue
                real_columns = {name.lower(): name for name in schema[table]}
                if column.lower() not in real_columns:
                    continue
                real_column = real_columns[column.lower()]
                candidates = bird_data.column_values(
                    question["db_id"], table, real_column
                )
                if len(candidates) < 2 or not any(
                    bird_data.normalize(candidate) == bird_data.normalize(value)
                    for candidate in candidates
                ):
                    continue
                instances.append(
                    {
                        "question": question["question"],
                        "evidence": question.get("evidence", ""),
                        "db_id": question["db_id"],
                        "table": table,
                        "column": real_column,
                        "gold": value,
                        "candidates": list(candidates),
                        "drift": bird_data.normalize(value)
                        not in bird_data.normalize(question["question"]),
                    }
                )
                if len(instances) >= max_instances:
                    return instances
        return instances

    corrected = corrected_instances(bird_data.questions())
    old_by_key: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for instance in original_instances:
        key = (
            instance["db_id"],
            instance["question"],
            instance["column"].lower(),
            bird_data.normalize(instance["gold"]),
        )
        old_by_key[key].append(instance)
    changed_tables = 0
    for instance in corrected:
        key = (
            instance["db_id"],
            instance["question"],
            instance["column"].lower(),
            bird_data.normalize(instance["gold"]),
        )
        previous = old_by_key[key].pop(0) if old_by_key[key] else None
        changed_tables += int(
            previous is not None and previous["table"].lower() != instance["table"].lower()
        )

    bird_data.value_link_instances = corrected_instances
    with tempfile.TemporaryDirectory(prefix="relationalrag-bird-correction-") as temporary:
        run_bird.RESULTS = Path(temporary) / "results.json"
        runs = [run_bird.main(seed=seed) for seed in range(5)]
        run_e6_labelnoise.OUT = Path(temporary) / "e6"
        noise = run_e6_labelnoise.bird(seed=0)

    metrics: dict[str, object] = {}
    for method, subset in (
        ("exact", "all"),
        ("exact", "drift"),
        ("fuzzy", "all"),
        ("fuzzy", "drift"),
        ("embed", "all"),
        ("embed", "drift"),
        ("verifier", "all"),
        ("verifier", "drift"),
    ):
        metrics[f"{method}_{subset}"] = aggregate(
            [run["recall@1_test"][method][subset] for run in runs]
        )
    for scorer in ("verifier", "embed_cos", "fuzzy", "exact"):
        metrics[f"auroc_{scorer}"] = aggregate(
            [run["auroc_test"][scorer] for run in runs]
        )

    seed_zero_raw = runs[0]
    false_positive = seed_zero_raw["failure_test"]["fp_rate"]
    false_negative = seed_zero_raw["failure_test"]["fn_rate"]
    positive = seed_zero_raw["failure_test"]["tp"] + seed_zero_raw["failure_test"]["fn"]
    negative = seed_zero_raw["failure_test"]["tn"] + seed_zero_raw["failure_test"]["fp"]
    prevalence = positive / (positive + negative)
    critical_penalty = prevalence * false_negative / (
        (1 - prevalence) * (1 - false_positive)
    )

    output = {
        "protocol": "Alias-resolve each SQL condition before confirming the literal in the filtered table column.",
        "original_instance_count": len(original_instances),
        "corrected_instance_count": len(corrected),
        "corrected_drift_instance_count": sum(instance["drift"] for instance in corrected),
        "table_assignments_changed": changed_tables,
        "seeds": list(range(5)),
        "recall_at_1": metrics,
        "seed_0": {
            "instances": seed_zero_raw["config"]["instances"],
            "test_instances": seed_zero_raw["config"]["test_rows"],
            "test_drift_instances": seed_zero_raw["config"]["test_drift_instances"],
            "verifier_recall_all": seed_zero_raw["recall@1_test"]["verifier"]["all"],
            "verifier_recall_drift": seed_zero_raw["recall@1_test"]["verifier"]["drift"],
            "verifier_auroc": seed_zero_raw["auroc_test"]["verifier"],
            "verifier_ece": seed_zero_raw["calibration_test"]["ece"],
            "false_positive_rate": false_positive,
            "false_negative_rate": false_negative,
        },
        "seed_0_theory_inputs": {
            "match_prevalence": round(prevalence, 4),
            "false_positive_rate": false_positive,
            "false_negative_rate": false_negative,
            "critical_penalty": round(critical_penalty, 4),
        },
        "label_noise_seed_0": {
            key: {
                "recall_all": row["recall@1_all"],
                "recall_drift": row["recall@1_drift"],
                "false_positive_rate": row["gate_at_0.5"]["fp_rate"],
                "false_negative_rate": row["gate_at_0.5"]["fn_rate"],
            }
            for key, row in noise["by_rho"].items()
        },
        "interpretation": {
            "bird": "Correcting table attribution changes candidate domains, so the BIRD component and label-noise results were rerun.",
            "icd": "The five adjudicated ICD cases change only drift-subtype metadata from corrupt to hier; record codes, gold nodes, pair features, aggregate matching, and end-to-end QA outcomes are unchanged. Uncorrected subtype-stratified rows must not be cited.",
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
