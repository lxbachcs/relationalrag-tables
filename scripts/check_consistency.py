#!/usr/bin/env python3
"""
check_consistency.py -- anti-drift gate for RelationalRAG-Tables.

Every headline number printed in the manuscript MUST be present in a committed
result JSON. This script encodes each headline claim as (paper_value, json_file,
key_path, tolerance) and re-reads the measured value from the artifact's result
JSONs. It FAILS (exit 1) if any paper number drifts from the measured value, so a
stale number in the paper cannot pass CI silently.

Run from the repository root:
    python3 scripts/check_consistency.py
    python3 scripts/check_consistency.py --artifacts artifacts   # custom dir

Key paths use dotted notation; integer tokens index into lists, e.g.
"selective_prediction.risk_coverage.4.accuracy".
"""
from __future__ import annotations
import argparse
import json
import os
import sys

# (id, paper_value, json_file, key_path, abs_tol, note)
# paper_value is the number AS PRINTED in the manuscript (rounded as shown).
# abs_tol allows for the paper's display rounding vs. the JSON's full precision.
CLAIMS = [
    # --- Section V: controlled synthetic study (artifacts/experiments/results.json) ---
    ("synth.corpus_accounts",   600,    "experiments/results.json", "corpus.num_accounts", 0,      "600-account corpus"),
    ("synth.num_questions",     1392,   "experiments/results.json", "meta.num_questions", 0,        "1,392 QA questions"),
    ("synth.verif_train",       4000,   "experiments/results.json", "meta.num_verifier_train", 0,   "4,000 verifier-training pairs"),
    ("synth.verif_eval",        1500,   "experiments/results.json", "meta.num_verifier_eval", 0,    "1,500 held-out verifier pairs"),
    ("synth.drift_frac",        0.6,    "experiments/results.json", "corpus.qa_join_drift", 0.001,  "drift fraction 0.6"),
    ("synth.em.exact",          0.569,  "experiments/results.json", "qa_by_verifier.exact_match.em", 0.001, "exact-match aggregate EM 0.569"),
    ("synth.em.learned",        0.856,  "experiments/results.json", "qa_by_verifier.learned.em", 0.001,    "learned EM 0.856"),
    ("synth.em.ensemble",       0.861,  "experiments/results.json", "qa_by_verifier.ensemble.em", 0.001,   "ensemble EM 0.861"),
    ("synth.disc.exact_recall", 0.36,   "experiments/results.json", "verifier_discrimination.exact_match.recall", 0.005, "exact held-out recall 0.36"),
    ("synth.disc.learned_recall",0.81,  "experiments/results.json", "verifier_discrimination.learned.recall", 0.005,     "learned recall 0.81"),
    ("synth.disc.learned_auroc",0.81,   "experiments/results.json", "verifier_discrimination.learned.auroc", 0.005,      "learned AUROC 0.81"),
    ("synth.disc.ens_auroc",    0.82,   "experiments/results.json", "verifier_discrimination.ensemble.auroc", 0.005,     "ensemble AUROC 0.82"),
    ("synth.disc.learned_brier",0.174,  "experiments/results.json", "verifier_discrimination.learned.brier", 0.001,      "learned Brier 0.174"),
    ("synth.disc.ens_brier",    0.173,  "experiments/results.json", "verifier_discrimination.ensemble.brier", 0.001,     "ensemble Brier 0.173"),
    # --- Section V: calibration (artifacts/calibration/results.json) ---
    ("calib.ece.ensemble",      0.027,  "calibration/results.json", "calibration.ensemble.ece", 0.001,   "ECE ensemble 0.027"),
    ("calib.ece.exact",         0.369,  "calibration/results.json", "calibration.exact_match.ece", 0.001,"ECE exact-match 0.369"),
    ("calib.mce.ensemble",      0.046,  "calibration/results.json", "calibration.ensemble.mce", 0.001,   "MCE ensemble 0.046"),
    # --- Section V: latency stage breakdown (artifacts/diagnostics/results.json) ---
    # Reconciled 2026-06 to the committed diagnostics run (see PROVENANCE.md, Residual R1):
    # the paper sentence (sec 05, ~l.182) now prints the measured 11.1/16.9 and 0.7/1.8 ms.
    ("lat.retrieval_p50",       11.1,   "diagnostics/results.json", "efficiency.retrieval_ms.p50", 0.05, "retrieval P50 11.1 ms"),
    ("lat.retrieval_p95",       16.9,   "diagnostics/results.json", "efficiency.retrieval_ms.p95", 0.05, "retrieval P95 16.9 ms"),
    ("lat.exec_p50",            0.7,    "diagnostics/results.json", "efficiency.execution_ms.p50", 0.05, "execution P50 0.7 ms"),
    ("lat.exec_p95",            1.8,    "diagnostics/results.json", "efficiency.execution_ms.p95", 0.05, "execution P95 1.8 ms"),
    ("lat.total_p50",           11.9,   "diagnostics/results.json", "efficiency.total_ms.p50", 0.05,    "total P50 11.9 ms (supplement Table s-eff)"),
    ("lat.total_p95",           17.9,   "diagnostics/results.json", "efficiency.total_ms.p95", 0.05,    "total P95 17.9 ms (supplement Table s-eff)"),
    # §VII restates the §V-F latency; guard it too so the two sections cannot drift apart.
    ("lat7.retrieval_p50",      11.1,   "diagnostics/results.json", "efficiency.retrieval_ms.p50", 0.05,"§VII retrieval P50 11.1 ms"),
    ("lat7.retrieval_p95",      16.9,   "diagnostics/results.json", "efficiency.retrieval_ms.p95", 0.05,"§VII retrieval P95 16.9 ms"),
    ("lat7.exec_p50",           0.7,    "diagnostics/results.json", "efficiency.execution_ms.p50", 0.05,"§VII execution P50 0.7 ms"),
    ("lat7.exec_p95",           1.8,    "diagnostics/results.json", "efficiency.execution_ms.p95", 0.05,"§VII execution P95 1.8 ms"),
    # §VII scalability Table (artifacts/scalability/results.json): row index = corpus step.
    ("scale.r0_p50",            10.5,   "scalability/results.json", "0.lat_p50_ms", 0.06,  "scalability 6.5k P50"),
    ("scale.r0_p95",            11.5,   "scalability/results.json", "0.lat_p95_ms", 0.06,  "scalability 6.5k P95"),
    ("scale.r1_p95",            28.3,   "scalability/results.json", "1.lat_p95_ms", 0.06,  "scalability 13k P95"),
    ("scale.r2_p95",            51.5,   "scalability/results.json", "2.lat_p95_ms", 0.06,  "scalability 26k P95"),
    ("scale.r3_p95",            105.3,  "scalability/results.json", "3.lat_p95_ms", 0.06,  "scalability 52k P95"),
    ("scale.r4_p50",            201.1,  "scalability/results.json", "4.lat_p50_ms", 0.06,  "scalability 104k P50"),
    ("scale.r4_p95",            207.1,  "scalability/results.json", "4.lat_p95_ms", 0.06,  "scalability 104k P95"),
    # --- Section VI-A: ICD-10 + Wikidata join (artifacts/icd_kg/results.json) ---
    ("icd.kg_nodes",            590,    "icd_kg/results.json", "meta.kg_nodes", 0,             "590 Wikidata disease nodes"),
    ("icd.exact_drift",         0.000,  "icd_kg/results.json", "r1.exact.drift", 0.001,        "exact code drift recall 0.000"),
    ("icd.verifier_drift",      0.746,  "icd_kg/results.json", "r1.verifier.drift", 0.002,     "verifier drift recall 0.746"),
    ("icd.verifier_all",        0.83,   "icd_kg/results.json", "r1.verifier.all", 0.005,       "verifier overall 0.83 (0.64->0.83)"),
    ("icd.exact_all",           0.64,   "icd_kg/results.json", "r1.exact.all", 0.005,          "exact overall 0.64"),
    ("icd.prefix_drift",        0.729,  "icd_kg/results.json", "r1.prefix.drift", 0.002,       "prefix-rule drift 0.729"),
    # --- Section VI-A: heterogeneous (artifacts/icd_kg/results_hetero_aggregate.json) ---
    ("hetero.verifier",         0.752,  "icd_kg/results_hetero_aggregate.json", "mean.verifier", 0.002, "hetero verifier 0.752"),
    ("hetero.prefix",           0.258,  "icd_kg/results_hetero_aggregate.json", "mean.prefix", 0.002,   "hetero prefix 0.258"),
    ("hetero.embed",            0.689,  "icd_kg/results_hetero_aggregate.json", "mean.embed", 0.002,    "hetero embedding 0.689"),
    # --- Section VI-A: cross-vocabulary (artifacts/icd_kg/results_xvocab.json) ---
    ("xvocab.exact",            0.01,   "icd_kg/results_xvocab.json", "recall@1.exact", 0.005,   "x-vocab exact ~0.01"),
    ("xvocab.embed",            0.91,   "icd_kg/results_xvocab.json", "recall@1.embed", 0.005,   "x-vocab embedding 0.91"),
    ("xvocab.verifier",         0.90,   "icd_kg/results_xvocab.json", "recall@1.verifier", 0.005,"x-vocab verifier 0.90"),
    # --- Section VI-B: BIRD value linking (artifacts/bird/results.json) ---
    ("bird.instances",          1105,   "bird/results.json", "config.instances", 0,             "1,105 value-linking instances"),
    ("bird.exact_drift",        0.00,   "bird/results.json", "recall@1_test.exact.drift", 0.001,"BIRD exact drift recall 0.00"),
    ("bird.verifier_drift",     0.92,   "bird/results.json", "recall@1_test.verifier.drift", 0.005, "BIRD verifier drift recall 0.92"),
    ("bird.verifier_all",       0.93,   "bird/results.json", "recall@1_test.verifier.all", 0.005,    "BIRD verifier overall 0.93"),
    ("bird.auroc",              0.996,  "bird/results.json", "auroc_test.verifier", 0.001,      "BIRD verifier AUROC 0.996"),
    ("bird.ece",                0.001,  "bird/results.json", "calibration_test.ece", 0.001,     "BIRD ensemble ECE 0.001"),
    # --- Section VI-C: BIRD end-to-end EX, full dev n=1534 (artifacts/bird_sql/*_1534.json) ---
    ("ex.baseline_full",        0.454,  "bird_sql/baseline_7b_1534.json", "ex", 0.002,          "full-dev base EX 0.454"),
    ("ex.prune_values_full",    0.477,  "bird_sql/prune_values_7b_1534.json", "ex", 0.002,      "full-dev prune+values EX 0.477"),
    ("ex.drift_baseline",       0.33,   "bird_sql/baseline_7b_1534.json", "ex_drift", 0.005,    "drift base EX 0.33"),
    ("ex.drift_pruneval",       0.42,   "bird_sql/prune_values_7b_1534.json", "ex_drift", 0.005,"drift prune+values EX 0.42 (+9.2)"),
    ("ex.drift_n",              131,    "bird_sql/baseline_7b_1534.json", "drift_n", 0,         "131 drift dev questions"),
    # --- Section VI-D: HybridQA negative result (artifacts/hybridqa/results.json) ---
    ("hqa.lexical",             0.25,   "hybridqa/results.json", "retrieval.lex@1", 0.01,       "HybridQA lexical@1 0.25"),
    ("hqa.verifier",            0.09,   "hybridqa/results.json", "reranking.recall@1.verifier_rerank", 0.005, "HybridQA verifier@1 0.09"),
]


def resolve(obj, key_path):
    cur = obj
    for tok in key_path.split("."):
        if isinstance(cur, list):
            cur = cur[int(tok)]
        elif isinstance(cur, dict):
            if tok not in cur:
                raise KeyError(key_path)
            cur = cur[tok]
        else:
            raise KeyError(key_path)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts",
                    help="directory holding the result JSON tree (default: artifacts)")
    args = ap.parse_args()

    cache = {}
    npass = nfail = nerr = 0
    failures = []
    for cid, paper_val, jfile, kpath, tol, note in CLAIMS:
        path = os.path.join(args.artifacts, jfile)
        try:
            if path not in cache:
                with open(path) as f:
                    cache[path] = json.load(f)
            measured = resolve(cache[path], kpath)
        except (FileNotFoundError, KeyError, IndexError, ValueError) as e:
            nerr += 1
            failures.append((cid, note))
            print(f"  ERROR  {cid:26s} {note}")
            print(f"         could not read {jfile}:{kpath} ({type(e).__name__})")
            continue
        try:
            ok = abs(float(measured) - float(paper_val)) <= tol
        except (TypeError, ValueError):
            ok = (measured == paper_val)
        if ok:
            npass += 1
        else:
            nfail += 1
            failures.append((cid, note))
            print(f"  FAIL   {cid:26s} paper={paper_val}  measured={measured}  (tol {tol})")
            print(f"         {note}  [{jfile}:{kpath}]")

    total = len(CLAIMS)
    print("\n" + "=" * 64)
    print(f"Consistency gate: {npass}/{total} headline numbers match the artifact.")
    if nfail or nerr:
        print(f"  {nfail} drifted, {nerr} unreadable. Drift items:")
        for cid, note in failures:
            print(f"    - {cid}: {note}")
        print("\nA FAIL means a manuscript number no longer matches the committed")
        print("result JSON. Reconcile the paper sentence or re-run the experiment.")
        return 1
    print("All headline numbers trace to a measured result file. No drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
