#!/usr/bin/env bash
# Full reproduction of the dependency-light SYNTHETIC core. Every command here is
# CPU-only and regenerates the controlled-study, selective-prediction,
# calibration, and diagnostics results from scratch into a scratch directory,
# then checks that every reported metric matches the shipped results. Embedded
# wall-clock latency fields (the only non-deterministic values) are excluded from
# the comparison, so the shipped manifest is never modified. The heavy real-data
# layer (ICD-10 + Wikidata, BIRD, HybridQA) needs the optional [real] extra,
# public downloads, and an Apple-silicon GPU; its commands are PRINTED at the end
# rather than run. This script does not read or build any manuscript LaTeX/PDF.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
REGEN=${REGEN:-artifacts_regen}

echo "[full 1/6] artifact hygiene"
$PY check_artifact.py

echo "[full 2/6] unit tests"
$PY -m unittest discover -s tests

echo "[full 3/6] regenerate the synthetic tables+KG corpus (60% join drift)"
$PY -m relational_rag_tables.cli synthesize --num-accounts 600 --seed 7 --qa-join-drift 0.6

echo "[full 4/6] controlled study: QA-by-task, discrimination, selective prediction, K-sweep"
$PY -m relational_rag_tables.cli experiments --fixtures fixtures/synthetic --ensemble-k 5 --seed 0 --out "$REGEN/experiments"

echo "[full 5/6] calibration (ECE/Brier, conformal) and diagnostics (threshold, Fellegi-Sunter, composite key)"
$PY -m relational_rag_tables.calibration --fixtures fixtures/synthetic --seed 0 --out "$REGEN/calibration"
$PY -m relational_rag_tables.diagnostics  --fixtures fixtures/synthetic --seed 0 --out "$REGEN/diagnostics"

echo "[full 6/6] verify regenerated metrics reproduce the shipped results (wall-clock latency excluded)"
REGEN="$REGEN" $PY - <<'PYEOF'
import json, os, pathlib, sys

# Wall-clock timing fields are machine-dependent and excluded from the check.
SKIP = ("latency", "_ms", "ms_", "runtime", "wall", "seconds", "elapsed")
def skip(path): return any(s in path.lower() for s in SKIP)

def walk(a, b, path, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            walk(a.get(k), b.get(k), f"{path}.{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs)
    else:
        if a == b or skip(path):
            return
        try:
            if abs(float(a) - float(b)) < 1e-9:
                return
        except (TypeError, ValueError):
            pass
        diffs.append((path, a, b))

regen = pathlib.Path(os.environ["REGEN"])
shipped = pathlib.Path("artifacts")
files = ["experiments/results.json", "calibration/results.json", "diagnostics/results.json"]
bad = []
for f in files:
    d = []
    walk(json.loads((shipped / f).read_text()),
         json.loads((regen / f).read_text()), f, d)
    bad += d
    print(f"  {f}: {'reproduced' if not d else str(len(d)) + ' metric diffs'}")
if bad:
    print("MISMATCH in non-timing metrics:")
    for p, a, b in bad[:20]:
        print(f"    {p}: regen={a!r} shipped={b!r}")
print(f"\nscientific reproduction: {'PASS' if not bad else 'FAIL'} "
      "(all reported metrics match; wall-clock latency fields excluded)")
sys.exit(1 if bad else 0)
PYEOF

rm -rf "$REGEN"

cat <<'NOTE'

FULL SYNTHETIC CORE DONE. Shipped artifacts/ left untouched; scratch regen removed.

The real-data layer is optional and environment-heavy (Python 3.13, Apple-silicon
GPU via MPS + MLX, public benchmark downloads). Install and run separately:

  python3.13 -m venv env && env/bin/pip install -r requirements-real.txt
  env/bin/python -m relational_rag_real.run_icd_kg            # ICD-10 + Wikidata join
  env/bin/python -m relational_rag_real.run_icd_xvocab        # cross-vocabulary join
  env/bin/python -m relational_rag_real.run_icd_hetero        # heterogeneous drift (seeds 0-4)
  env/bin/python -m relational_rag_real.run_icd_ce            # ICD cross-encoder
  env/bin/python -m relational_rag_real.run_bird              # BIRD value linking + ECE
  env/bin/python -m relational_rag_real.run_bird_sql          # end-to-end EX ablations
  env/bin/python -m relational_rag_real.run_hybridqa          # HybridQA negative result
  env/bin/python -m relational_rag_real.finetune_ce           # BIRD/HybridQA cross-encoder

See REPRODUCE.md for the full result-to-command map, seeds, and output files.
NOTE
