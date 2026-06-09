#!/usr/bin/env bash
# Fast artifact smoke test. Checks install/import, artifact hygiene, the unit
# tests, the packaged reference pipeline, and stored-result provenance. CPU-only
# and non-destructive (it does not overwrite the shipped result manifests). It
# does not touch the heavy real-data layer or any manuscript LaTeX/PDF files.
# Expected wall-clock: well under a minute.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}

echo "[smoke 1/4] artifact hygiene"
$PY check_artifact.py

echo "[smoke 2/4] unit tests"
$PY -m unittest discover -s tests

echo "[smoke 3/4] reference smoke pipeline (exact-match join; EM must be 1.0)"
$PY -m relational_rag_tables.cli smoke --fixtures fixtures/smoke --out artifacts/smoke

echo "[smoke 4/4] verify shipped stored-result manifest (sha256 + size, read-only)"
$PY - <<'PYEOF'
import hashlib, json, pathlib
m = json.load(open("artifacts/manifest.json"))
bad = [k for k, v in m.items()
       if not pathlib.Path(k).exists()
       or len(pathlib.Path(k).read_bytes()) != v["bytes"]
       or hashlib.sha256(pathlib.Path(k).read_bytes()).hexdigest() != v["sha256"]]
print(f"manifest: {len(m) - len(bad)}/{len(m)} stored results verified")
raise SystemExit(1 if bad else 0)
PYEOF

echo "SMOKE DONE"
