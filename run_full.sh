#!/usr/bin/env bash
# Full reproduction of the dependency-light SYNTHETIC core. Every command here
# is deterministic and CPU-only: it regenerates the controlled-study, selective-
# prediction, calibration, and diagnostics results from scratch and verifies they
# match the shipped manifest byte-for-byte. The heavy real-data layer (ICD-10 +
# Wikidata, BIRD, HybridQA) needs the optional [real] extra, public downloads,
# and an Apple-silicon GPU; its commands are PRINTED at the end rather than run.
# This script does not read or build any manuscript LaTeX/PDF files.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}

echo "[full 1/6] artifact hygiene"
$PY check_artifact.py

echo "[full 2/6] unit tests"
$PY -m unittest discover -s tests

echo "[full 3/6] regenerate the synthetic tables+KG corpus (60% join drift)"
$PY -m relational_rag_tables.cli synthesize --num-accounts 600 --seed 7 --qa-join-drift 0.6

echo "[full 4/6] controlled study: QA-by-task, discrimination, selective prediction, K-sweep"
$PY -m relational_rag_tables.cli experiments --fixtures fixtures/synthetic --ensemble-k 5 --seed 0

echo "[full 5/6] calibration (ECE/Brier, conformal) and diagnostics (threshold, Fellegi-Sunter, composite key)"
$PY -m relational_rag_tables.calibration --fixtures fixtures/synthetic --seed 0
$PY -m relational_rag_tables.diagnostics  --fixtures fixtures/synthetic --seed 0

echo "[full 6/6] verify regenerated synthetic results against the shipped manifest"
$PY - <<'PYEOF'
import hashlib, json, pathlib
m = json.load(open("artifacts/manifest.json"))
# Synthetic, deterministic outputs regenerated above.
synth = ("experiments/", "calibration/", "diagnostics/")
checked = {k: v for k, v in m.items()
           if any(s in k for s in synth) and "scalability" not in k}
bad = [k for k, v in checked.items()
       if not pathlib.Path(k).exists()
       or hashlib.sha256(pathlib.Path(k).read_bytes()).hexdigest() != v["sha256"]]
print(f"synthetic regeneration: {len(checked) - len(bad)}/{len(checked)} results match shipped manifest")
if bad:
    print("  mismatched (often a NumPy/scikit-learn version difference):")
    for k in bad:
        print(f"    - {k}")
raise SystemExit(1 if bad else 0)
PYEOF

cat <<'NOTE'

FULL SYNTHETIC CORE DONE.

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
