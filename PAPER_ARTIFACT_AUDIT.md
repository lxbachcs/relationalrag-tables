# Reference, Proof, And Artifact Audit

Date: 2026-06-09  
Scope: `references.bib`, compiled `main.pdf` / `supplement.pdf`, LaTeX logs, `README.md`, `REPRODUCE.md`, `artifacts/`, lightweight CLI/tests.

## Executive Summary

No P0 blockers were found. The paper and supplement compile cleanly after the bibliography refresh: no undefined citations, no undefined references, no stale rerun warnings, and no overfull boxes were reported in the final logs.

The largest reviewer-facing risk before this pass was reference polish: `references.bib` had no DOI fields and five abbreviated author lists using `and others`. I applied only confirmed safe fixes: 32 DOI fields were added, all `and others` author lists were expanded, and the Magellan entry was corrected to its full PVLDB title, author list, issue, pages, and DOI.

The artifact package is strong for a reviewer quick check. The synthetic tests pass, the CLI is discoverable, the smoke run succeeds with temporary output, and `artifacts/manifest.json` verifies 36 tracked result JSON files. Heavy BIRD/HybridQA/ICD reruns were not executed, per plan; those claims are assessed from stored outputs, commands, manifests, and documentation.

## P0/P1/P2 Findings

### P0

None found.

### P1

None remaining after safe fixes.

### P2

1. **Residual DOI-free references need optional manual curation, not automatic patching.**  
   Evidence: `references.bib` now has 70 entries and 32 DOI fields, but some venues/arXiv-style entries remain DOI-free. Unsafe DOI candidates were deliberately left out, including `vaswani2017attention` where OpenAlex surfaced a suspicious 2025 DOI, and `vapnik1971rademacher` where the candidate DOI pointed to a later reprint/chapter rather than the 1971 source. This is reviewer-safe: a missing DOI is preferable to an incorrect DOI.

2. **Real-data reproduction is documented but not lightweight-runnable in this audit.**  
   Evidence: `README.md:147-207` describes the real layer as BIRD/HybridQA/ICD scripts needing datasets, pretrained models, embeddings, and a local coder; `REPRODUCE.md:48-55` explicitly states real-data runs are seeded but not bit-identical across hardware. Stored results exist and are manifest-verified, but heavyweight reruns were intentionally skipped.

3. **The generic artifact-auditor probe expects `artifact/`, while this repo uses `artifacts/`.**  
   Evidence: the probe reported "Artifact directory not found" for singular `artifact/`, but `artifacts/manifest.json` exists and verifies. This is a tooling-name mismatch, not a package defect.

## Safe Fixes Applied

### Bibliography Metadata

Confirmed safe changes in `references.bib`:

| Fix type | Count / entries | Evidence |
|---|---:|---|
| DOI fields added | 32 | `references.bib:6`, `references.bib:14`, `references.bib:22`, ..., `references.bib:540` |
| `and others` removed | 5 -> 0 | Expanded `yu2018spider`, `nan2022fetaqa`, `lewis2020rag`, `cheng2023binding`, `wei2022cot`, `ouyang2022instructgpt`, `li2023bird`, `konda2016magellan` where needed |
| Magellan metadata corrected | 1 | `references.bib:419-424` now has the full PVLDB title, authors, volume 9, number 13, pages 1581--1584, DOI `10.14778/3007263.3007314` |
| BIRD provenance clarified | 1 | `references.bib:364-369` has full authors and arXiv DOI `10.48550/arXiv.2305.03111`; no final proceedings DOI was found in this pass |

Representative DOI additions:

| Key | DOI |
|---|---|
| `herzig2020tapas` | `10.18653/v1/2020.acl-main.398` |
| `wang2020ratsql` | `10.18653/v1/2020.acl-main.677` |
| `yu2018spider` | `10.18653/v1/D18-1425` |
| `chen2020hybridqa` | `10.18653/v1/2020.findings-emnlp.91` |
| `nan2022fetaqa` | `10.1162/tacl_a_00446` |
| `li2024codes` | `10.1145/3654930` |
| `pourreza2024dts` | `10.18653/v1/2024.findings-emnlp.481` |
| `li2020ditto` | `10.14778/3421424.3421431` |
| `brunner2021valuenet` | `10.1109/ICDE51399.2021.00220` |
| `fellegi1969theory` | `10.1080/01621459.1969.10501049` |
| `angelopoulos2023conformal` | `10.1561/2200000101` |
| `linacre2022splink` | `10.23889/ijpds.v7i3.1794` |

Metadata sources checked: OpenAlex title/author/DOI records, targeted DBLP metadata for PVLDB Magellan, ACL Anthology DOI patterns for ACL-family papers, and arXiv/OpenReview-style venue records where appropriate. A Crossref batch probe was attempted but became unresponsive and was stopped; no Crossref-only changes were made from that failed run.

## Reference Metadata Audit Table

| Status | Entries |
|---|---|
| Confirmed and patched with DOI | `herzig2020tapas`, `yin2020tabert`, `wang2020ratsql`, `yu2018spider`, `pasupat2015wtq`, `chen2020hybridqa`, `nan2022fetaqa`, `parikh2020totto`, `li2022multispan`, `wang2020mimicsql`, `sun2020sparqa`, `lan2021kbqa`, `eisenschlos2020understanding`, `rajpurkar2016squad`, `williams1992reinforce`, `valiant1984pac`, `wei2022cot`, `li2023resdsql`, `saxena2020embedkgqa`, `li2024codes`, `pourreza2024dts`, `li2023bird`, `li2020ditto`, `konda2016magellan`, `christophides2020ersurvey`, `lin2020bridge`, `brunner2021valuenet`, `efthymiou2017matching`, `sung2020biosyn`, `fellegi1969theory`, `angelopoulos2023conformal`, `linacre2022splink` |
| Confirmed author-list expansion, DOI left absent or already handled | `lewis2020rag`, `cheng2023binding`, `ouyang2022instructgpt` |
| Left DOI-free because authoritative final DOI was not confirmed | `raffel2020t5`, `yao2023react`, `schick2023toolformer`, `chen2020ottqa`, `karpukhin2020dpr`, `lakshminarayanan2017ensembles`, `guo2017calibration`, `liu2022tapex`, `rajkumar2022dailsql`, `chen2023chattable`, `devlin2019bert`, `schulman2017ppo`, `khattab2022demonstrate`, `vrandecic2014wikidata`, `bollacker2008freebase`, `vaswani2017attention`, `wang2022selfconsistency`, `izacard2022atlas`, `wang2024macsql`, `bordes2013transe`, `schulman2015trpo`, `fort2020deepensembles`, `gal2016dropout`, `nan2024tablequery`, `wang2024verifiers`, `vapnik1971rademacher`, `bartlett2002rademacher`, `mudgal2018deepem`, `pourreza2023dinsql`, `deng2020turl`, `cao2024rslsql`, `caferoglu2024esql`, `winkler1990jaro`, `zhu2019josie` |

## Proof And Layout Audit

### Compile Status

Both documents were rebuilt through normal LaTeX/BibTeX:

| Target | Result |
|---|---|
| `bibtex main` | Passed |
| `bibtex supplement` | Passed |
| `pdflatex -interaction=nonstopmode main.tex` twice | Passed; final `main.pdf` has 13 pages |
| `pdflatex -interaction=nonstopmode supplement.tex` twice | Passed; final `supplement.pdf` has 8 pages |

### Log Findings

| Check | Status | Evidence |
|---|---|---|
| Undefined citations | Clean | No `Citation ... undefined` in `main.log`, `supplement.log`, `main.blg`, or `supplement.blg` |
| Undefined references | Clean | No `undefined references` in final logs |
| Rerun warnings | Clean | No `Rerun to get cross-references` in final logs |
| Overfull boxes | Clean | No `Overfull` lines in final logs |
| BibTeX warnings | Clean | `main.blg` and `supplement.blg` contain no warning lines |
| Font warnings | Residual benign | `main.log:1016-1031` font shape declaration warnings; `main.log:1071`, `supplement.log:480`, `supplement.log:484` `OT1/ptm/m/scit` warnings |

### Visual Spot Checks

Rendered with `pdftoppm`:

| Page(s) | Output | Result |
|---|---|---|
| Main page 1 | `/tmp/audit-main-p1-01.png` | Title/abstract/Figure 1 area visually sane |
| Main pages 6-8 | `/tmp/audit-main-tables-06.png` through `/tmp/audit-main-tables-08.png` | Recently resized tables/plots fit; no obvious clipping in inspected pages |
| Main bibliography pages 12-13 | `/tmp/audit-main-bib-12.png`, `/tmp/audit-main-bib-13.png` | Bibliography renders normally |
| Supplement page 8 | `/tmp/audit-supp-bib-8.png` | Supplement bibliography renders normally |

## Artifact Claim Coverage

| Claim / promise | Status | Evidence |
|---|---|---|
| Dependency-light synthetic core exists | Verified | `README.md:11-19`; `requirements.txt` contains only the light synthetic dependencies |
| CLI exposes reproduction commands | Verified | `python3 -m relational_rag_tables.cli --help` succeeded and lists `smoke`, `synthesize`, `benchmark`, `experiments`, etc. |
| Smoke test works without touching tracked artifacts | Verified | `python3 -m relational_rag_tables.cli smoke --fixtures fixtures/smoke --out /tmp/rrt-smoke-audit` returned `smoke passed: EM=1.0` |
| Unit tests pass | Verified | `python3 -m unittest discover -s tests -p 'test_*.py' -v`: 29 tests, OK |
| Stored output manifest is complete for tracked result JSONs | Verified | `artifacts/manifest.json`: 36 entries verified, 0 missing, 0 hash mismatches |
| Smoke JSONs intentionally excluded from manifest | Verified / benign | 5 extra JSONs under `artifacts/smoke/`; `REPRODUCE.md:92-96` shows manifest generation excludes paths containing `smoke` |
| BIRD value-linking headline is backed by stored output | Verified | `artifacts/bird/results.json`: exact drift `0.0`, verifier drift `0.9211`, verifier all `0.9276`, runtime `31.3` seconds |
| BIRD EX full-dev gain is backed by stored output | Verified | `artifacts/bird_sql/baseline_7b_1534.json`: EX `0.4537`, drift `0.3282`; `artifacts/bird_sql/prune_values_7b_1534.json`: EX `0.4765`, drift `0.4198` |
| Cross-vocabulary and heterogeneous ICD artifacts exist | Verified | `artifacts/icd_kg/results_xvocab.json`; `artifacts/icd_kg/results_hetero_aggregate.json`; manifest hashes verified |
| Heavy real-data runs are reproducible from docs alone | Partially verified | `README.md:147-207` and `REPRODUCE.md:48-87` document scripts, seeds, and limitations, but model/dataset-heavy reruns were not performed |
| "Under a minute" synthetic claim | Partially verified | Unit tests and smoke are seconds; full synthetic `experiments` command was not rerun in this pass to avoid unnecessary artifact churn, though docs map it to CPU-only commands |

## Commands Run

Reference/proof:

```sh
python3 /tmp/openalex_probe.py
bibtex main
bibtex supplement
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode supplement.tex
pdflatex -interaction=nonstopmode supplement.tex
pdftoppm -f 1 -l 1 -png -r 150 main.pdf /tmp/audit-main-p1
pdftoppm -f 6 -l 8 -png -r 150 main.pdf /tmp/audit-main-tables
pdftoppm -f 12 -l 13 -png -r 150 main.pdf /tmp/audit-main-bib
pdftoppm -f 8 -l 8 -png -r 150 supplement.pdf /tmp/audit-supp-bib
```

Artifact:

```sh
python3 ~/.codex/skills/artifact-auditor/scripts/artifact_audit_probe.py .
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m relational_rag_tables.cli --help
rm -rf /tmp/rrt-smoke-audit
python3 -m relational_rag_tables.cli smoke --fixtures fixtures/smoke --out /tmp/rrt-smoke-audit
python3 - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('artifacts')
m=json.load(open(root/'manifest.json'))
missing=[]; bad=[]
for rel, meta in m.items():
    p=root/rel
    if not p.exists():
        missing.append(rel); continue
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    size=len(p.read_bytes())
    if digest != meta.get('sha256') or size != meta.get('bytes'):
        bad.append(rel)
print({'ok': len(m)-len(missing)-len(bad), 'missing': missing, 'bad': bad})
PY
```

## Prioritized Fix Checklist

1. Optional: manually curate the remaining DOI-free references that are arXiv/OpenReview/workshop entries, but only when an authoritative final DOI is found.
2. Optional: document in `REPRODUCE.md` that `artifacts/smoke/` is intentionally excluded from `artifacts/manifest.json`; the current manifest command already implies this.
3. Optional: add a one-line note that heavyweight real-data commands were not designed to be bit-identical because embeddings and local LLM decoding vary across hardware; `REPRODUCE.md:53-55` already says this, so this is polish rather than a blocker.
4. Leave the residual font-shape warnings alone unless the camera-ready style permits font package changes; they do not block references, floats, citations, or layout.
