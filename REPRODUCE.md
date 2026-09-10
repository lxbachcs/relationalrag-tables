# Reproduction Guide

This artifact has three parts:

- **`relational_rag_tables/`** — the dependency-light **synthetic core** (no GPU, no
  network, no benchmark data). Reproduces every controlled-study and diagnostics result
  on a CPU in well under a minute. Deps: [`requirements.txt`](requirements.txt).
- **`relational_rag_real/`** — the optional **heavy real-data layer** (network + Apple-silicon
  GPU + local LLM). Reproduces the ICD-10/Wikidata, BIRD, and HybridQA numbers. Deps:
  [`requirements-real.txt`](requirements-real.txt).
- **`artifacts/`** — **stored outputs** for every reported number, with SHA256 + size in
  [`artifacts/manifest.json`](artifacts/manifest.json).

A strict reviewer can verify all numbers from `artifacts/` without rerunning, re-derive the
synthetic half on any laptop, and rerun the real half given the environment below.

## Environments

| Layer | Python | Install | Hardware |
|---|---|---|---|
| Synthetic core | 3.14 (tested; ≥3.10 expected) | `pip install -r requirements.txt` | CPU only |
| Real-data layer | 3.13 (MLX needs ≤3.13) | `pip install -r requirements-real.txt` | Apple-silicon GPU (MPS + MLX) |

Tested synthetic versions: `numpy 2.4.6`, `scikit-learn 1.9.0`. Real-layer lower bounds are
in `requirements-real.txt`; tested versions: `mlx-lm 0.31`, `sentence-transformers 5.x`,
`faiss-cpu 1.14`, `torch` (MPS), `transformers`, `datasets 5.x`.

## Quick start (synthetic core, CPU, < 1 min)

```bash
python3 -m venv env && env/bin/pip install -r requirements.txt
env/bin/python -m unittest discover -s tests                       # 29 tests
env/bin/python -m relational_rag_tables.cli smoke --fixtures fixtures/smoke --out artifacts/smoke
env/bin/python -m relational_rag_tables.cli synthesize --num-accounts 600 --seed 7 --qa-join-drift 0.6
env/bin/python -m relational_rag_tables.cli experiments --fixtures fixtures/synthetic --ensemble-k 5 --seed 0
env/bin/python -m relational_rag_tables.calibration --fixtures fixtures/synthetic --seed 0
env/bin/python -m relational_rag_tables.diagnostics  --fixtures fixtures/synthetic --seed 0
```

## Real-data models and datasets

The artifact contains no persistent trained verifier checkpoint files. The
small MLP/ensemble verifier is trained from the fixtures by the reproduction
scripts. Heavy upstream benchmark corpora are not redistributed here; the
real-data drivers download them from their original public sources.

| Component | Identifier | Source |
|---|---|---|
| Sentence embedding | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | sentence-transformers |
| Local SQL coder (main) | [`Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) (4-bit, MLX) | MLX community |
| Local SQL coder (scale check) | [`Qwen2.5-Coder-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) (4-bit, MLX) | MLX community |
| Cross-encoder | [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | HuggingFace |
| BIRD | text-to-SQL dev (95 DBs) | BIRD benchmark download |
| HybridQA | WikiTables-WithLinks dev | HybridQA repo |
| ICD-10-CM codes | billable codes + descriptions | public ICD-10 mirror |
| Wikidata diseases | P494 (ICD-10), P493 (ICD-9) | Wikidata SPARQL endpoint |

Real-data runs are **seeded but not bit-identical** across hardware (pretrained embeddings +
local LLM decoding). Reported real numbers are single-seed point estimates except the
five-seed heterogeneous mean. Approximate runtimes (Apple-silicon laptop): `run_bird` ~15 s,
`run_icd_*` seconds, `run_hybridqa` ~3 min, `finetune_ce` ~3 min, `run_bird_sql ablation 800`
~20–40 min (800 local-LLM SQL generations), `ablation 1534` ~1 h.

## Result-to-command map

Synthetic (deterministic; output JSON under `artifacts/`):

| Paper item | Command | Output | Status |
|---|---|---|---|
| QA-by-task, discrimination, selective, K-sweep | `cli experiments` | `experiments/results.json` | deterministic |
| ECE, system selective prediction, faithfulness, retrieval noise, cost-aware utility, conformal selective prediction | `calibration` | `calibration/results.json` | deterministic |
| Threshold, compute-matched uncertainty, KG incompleteness, efficiency, gating cost, feature ablation, Fellegi–Sunter baseline, composite-key join | `diagnostics` | `diagnostics/results.json` | deterministic |
| Scalability stress test | `diagnostics.scalability` | `scalability/results.json` | wall-clock (latency varies) |

Real-data (seed shown; output under `artifacts/`):

| Paper item | Command | Output | Seed/status |
|---|---|---|---|
| ICD-10 hierarchy join | `run_icd_kg` | `icd_kg/results.json` | seed 0 |
| ICD OCR-corruption sweep | `run_icd_kg` (vary `corrupt_frac`) | `icd_kg/results_corrupt.json` | seed 0 |
| ICD feature ablation / failure analysis | `run_icd_kg.feature_ablation` / `.failure_analysis` | (stdout) | seed 0 |
| ICD lightweight structural context | `run_icd_kg.structural_context` | `icd_kg/results_struct.json` | seed 0, controlled split |
| ICD consolidated baselines (canon, Jaro–Winkler, TF-IDF) | `run_icd_kg.baselines` | `icd_kg/results_baselines.json` | seed 0 |
| ICD threshold sensitivity, leakage, regime detector, retrieval noise | `run_icd_kg.threshold_sensitivity` / `.leakage` / `.regime_detector` / `.retrieval_noise` | `icd_kg/results_reviewer.json` | seed 0 |
| FAISS ANN scalability (exact vs IVF, 10⁴–10⁶) | `run_icd_kg.ann_scalability` | `icd_kg/results_reviewer.json` | needs faiss-cpu |
| Cross-vocabulary join | `run_icd_xvocab` | `icd_kg/results_xvocab.json` | seed 0 |
| Heterogeneous drift (5-seed) | `run_icd_hetero` seeds 0–4 | `icd_kg/results_hetero_seed{0..4}.json`, `results_hetero_aggregate.json` | 5-seed mean, paired bootstrap |
| ICD cross-encoder | `run_icd_ce` | `icd_kg/results_ce.json` | seed 0 |
| BIRD value linking + ECE + failure | `run_bird` | `bird/results.json` | seed 0 |
| BIRD/HybridQA cross-encoder | `finetune_ce` | `crossencoder/results.json` | single run |
| HybridQA retrieval/rerank | `run_hybridqa` | `hybridqa/results.json` | seed 0 |
| End-to-end EX ablation | `run_bird_sql ablation 800` (then `1534`) | `bird_sql/*.json` | single run |
| Schema-linker comparison | `run_bird_sql slink 800` | `bird_sql/prune_*.json` | single run |

## Provenance

`artifacts/manifest.json` lists the SHA256 and byte size of every stored result JSON.
Transient outputs under `artifacts/smoke/` are intentionally excluded: the smoke run
rewrites them on every invocation, so they are not provenance-tracked (the regeneration
command below filters any path containing `smoke`). Manifest keys are repo-relative
(they include the leading `artifacts/`), so verify with `pathlib.Path(key)` directly.
Regenerate it with:

```bash
python3 -c "import hashlib,json,pathlib;root=pathlib.Path('artifacts');print(json.dumps({str(f):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':len(f.read_bytes())} for f in sorted(root.rglob('*.json')) if f.name!='manifest.json' and 'smoke' not in f.parts}, indent=2))" > artifacts/manifest.json
```

## Revision experiments (ACML journal-track round 1)

All numbers added in the revision are produced by the drivers below and rendered into the
manuscript by `scripts/render_revision.py` (which writes LaTeX rows/macros from the JSON
outputs). The real-data drivers were run on a CUDA server with vLLM 0.7.3 and the official
`Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`; embeddings use `BAAI/bge-small-en-v1.5`.

| Paper item | Command | Output |
|---|---|---|
| 20-seed synthetic study, risk–coverage curves, AURC, validation thresholds (E2) | `python -m relational_rag_tables.multiseed --seeds 20 --workers 20` | `artifacts/multiseed/{seed_*,aggregate}.json` |
| Theorem 1–2 numerical check (E3) | `python -m relational_rag_tables.theory_check` | `artifacts/theory_check.json` |
| Verifier-isolated end-to-end BIRD ablation (E1) | `python -m relational_rag_real.run_e1 --configs all --n 1534`; then `python scripts/analyze_e1.py artifacts/e1` | `artifacts/e1/*.json`, `artifacts/e1/analysis.json` |
| Regime-aware joint matcher, BIRD + HybridQA (E4) | `python -m relational_rag_real.run_e4_joint --seeds 0,1,2` | `artifacts/e4/results.json` |
| Tuned rule cascades vs verifier, heterogeneous ICD (E5) | `python -m relational_rag_real.run_e5_cascade --seeds 0,1,2,3,4` | `artifacts/icd_kg/results_e5_cascade.json` |
| Label protocol, audit samples, label-noise sensitivity (E6) | `python -m relational_rag_real.run_e6_labelnoise` | `artifacts/e6/results.json`, `annotation_sample_*.csv` |
| End-to-end QA on the real ICD-10 + Wikidata table+KG (E7) | `python -m relational_rag_real.run_e7_icd_qa --seeds 0,1,2,3,4` | `artifacts/e7/results.json`, `fixtures/icd_qa/seed_*` |

The Wikidata relations used by E7 (`data/icd_kg/wikidata_icd_relations.json`,
`wikidata_icd_parent_relations.json`) were fetched from the public SPARQL endpoint on
2026-09-09 and are committed so the benchmark is reproducible offline.
