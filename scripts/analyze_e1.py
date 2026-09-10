"""Offline analysis of the E1 verifier-isolated end-to-end BIRD ablation.

Reads artifacts/e1/*.json (one file per config, aligned per-question records) and reports:
  * EX overall / drift subset per config with paired-bootstrap deltas (vs pv_none and vs pv_exact);
  * hint precision / gold-literal recall per linker (component-level view inside the run);
  * end-to-end selective prediction for the verifier-hinted configs: rank questions that
    received hints by their minimum hint confidence and report EX on the answered fraction.

Run: python3 scripts/analyze_e1.py [artifacts/e1] [out.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

E1 = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/e1")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/e1/analysis.json")
B = 10000


def paired(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    ix = rng.integers(0, n, (B, n))
    d = a[ix].mean(axis=1) - b[ix].mean(axis=1)
    return {"delta": round(float(a.mean() - b.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4), "n": int(n)}


def main():
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(E1.glob("*.json")) if p.stem != "analysis"}
    if not runs:
        print("no runs"); return
    rng = np.random.default_rng(0)
    ref_names = [r for r in ("pv_none", "pv_exact") if r in runs]
    table = {}
    for tag, r in runs.items():
        pq = r["per_question"]
        ok = np.array([q["ok"] for q in pq])
        drift = np.array([q["drift"] for q in pq])
        row = {"ex": r["ex"], "ex_drift": r["ex_drift"], "n": len(pq), "n_drift": int(drift.sum()),
               "exec_error_rate": r["exec_error_rate"], "hints": r["hints"], "vs": {}}
        for ref in ref_names:
            if ref == tag:
                continue
            rok = np.array([q["ok"] for q in runs[ref]["per_question"]])
            if len(rok) != len(ok):
                continue
            row["vs"][ref] = {"all": paired(ok, rok, rng), "drift": paired(ok[drift], rok[drift], rng)}
        # selective prediction over hinted questions (verifier-style linkers carry probabilities)
        hinted = [q for q in pq if q["hint_min_score"] is not None]
        if hinted and "verifier" in tag:
            items = sorted(hinted, key=lambda q: -q["hint_min_score"])
            n = len(items)
            curve = []
            for c in (1.0, 0.9, 0.8, 0.7, 0.5):
                keep = max(1, int(round(n * c)))
                curve.append({"coverage": c, "ex_answered": round(float(np.mean([q["ok"] for q in items[:keep]])), 4),
                              "min_conf": round(items[keep - 1]["hint_min_score"], 4)})
            row["selective_over_hinted"] = {"n_hinted": n, "curve": curve}
        table[tag] = row
    OUT.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"{'config':28s} {'EX':>7s} {'EXdrift':>8s} {'err':>6s} {'hintQ':>6s} {'prec':>6s} {'rec':>6s}  dEX vs none [CI]        dEXdrift vs none [CI]")
    for tag, row in table.items():
        v = row["vs"].get("pv_none", {})
        da = v.get("all", {}); dd = v.get("drift", {})
        print(f"{tag:28s} {row['ex']:7.4f} {str(row['ex_drift']):>8s} {row['exec_error_rate']:6.3f} "
              f"{row['hints']['questions_with_hints']:6d} {row['hints']['hint_precision']:6.3f} {row['hints']['gold_literal_recall']:6.3f}  "
              f"{da.get('delta', float('nan')):+.4f} {da.get('ci95', '')}   {dd.get('delta', float('nan')):+.4f} {dd.get('ci95', '')}")
    for tag, row in table.items():
        if "selective_over_hinted" in row:
            print(tag, "selective over hinted:", json.dumps(row["selective_over_hinted"]))


if __name__ == "__main__":
    main()
