# Self-contained label-protocol audit

This folder is a portable, offline annotation package for independently checking the
deterministic BIRD and ICD label protocols used by *RelationalRAG-Tables*. It contains
the 200 sampled items, separate worksheets for two annotators, the offline context an
annotator needs to check each protocol label (BIRD schemas, gold SQL, cell-value evidence;
ICD-10-CM code validity and a frozen Wikidata reference for every QID in the ICD sample),
validation and agreement code, provenance, tests, and integrity hashes.

The benchmark labels are produced by deterministic rules rather than by human claim
decomposition. Two annotators have independently audited how those rules were applied
to 100 BIRD and 100 ICD items, and all 17 flagged rows have been adjudicated. The final
statistics in `agreement.json` are therefore reportable: BIRD gold/drift error rates
are 0.12/0.00 with Cohen's kappa 1.00 for both judgments; ICD gold/drift-type error
rates are 0.00/0.05, with kappa undefined for the all-accepted gold judgment and 1.00
for drift type.

## Completed audit

The audit described below has been completed once. Its final statistics and the resolved
adjudication decisions are in `results/`; the individual annotator worksheets are not
distributed, and the worksheets in this folder are blank so the audit can be repeated.

## Requirements

- Python 3.10 or newer.
- A spreadsheet application or UTF-8 CSV editor.
- No Python packages, repository checkout, database, or network connection.

All scripts resolve paths relative to this folder, so commands work from any current
working directory.

## Package contents

| File | Role | Editable? |
|---|---|---:|
| `annotation_sample_bird.csv` | Frozen 100-row BIRD sample and protocol labels | No |
| `annotation_sample_icd.csv` | Frozen 100-row ICD sample and protocol labels | No |
| `icd_wikidata_reference.csv` | Offline labels, descriptions, aliases, ICD-10 claims, source revisions, and URLs for all 99 sampled QIDs | No |
| `bird_schema.csv`, `bird_schema.md` | Every table and column (with types and keys) of the 11 BIRD dev databases in the sample | No |
| `bird_column_values.csv` | Per BIRD row: distinct-value count of the protocol column, the raw cell values equal to the protocol literal, and a bounded sample of other values | No |
| `icd_icd10cm_reference.csv` | Per ICD row: every valid ICD-10-CM code under the protocol node's code, with descriptions | No |
| `build_offline_context.py` | Maintainer command that regenerates the three files above and the context columns from the artifact checkout; not needed for annotation | No |
| `bird_annotator_A.csv`, `icd_annotator_A.csv` | Annotator A worksheets | Yes, annotation columns only |
| `bird_annotator_B.csv`, `icd_annotator_B.csv` | Annotator B worksheets | Yes, annotation columns only |
| `compute_agreement.py` | Validates inputs, computes error/agreement statistics, and manages adjudication lists | No |
| `agreement.json` | Generated machine-readable status/results | Generated |
| `adjudicate_bird.csv`, `adjudicate_icd.csv` | Generated joint-resolution worksheets | Resolution columns only |
| `verify_package.py` | Checks hashes, schemas, row alignment, and offline-reference coverage | No |
| `test_compute_agreement.py` | Standard-library unit and package tests | No |
| `refresh_wikidata_reference.py` | Optional online maintenance command; not required for annotation | No |
| `PROVENANCE.md` | Origin, snapshot, scope, and claim-boundary record | No |
| `SOURCE_MANIFEST.sha256` | Integrity hashes for immutable package files | No |

## Quick start

Before distributing or annotating, verify the package:

```bash
python3 verify_package.py
python3 -m unittest -v test_compute_agreement.py
```

A valid untouched package reports 100 rows for each sample and annotator worksheet,
99/99 ICD QIDs covered by the offline reference, schemas for all 11 BIRD databases,
column-value and ICD-10-CM reference rows aligned with the samples, no external Python
dependencies, and matching source hashes.

## Annotation protocol

Annotators work independently. Annotator A edits only files ending `_annotator_A.csv`;
annotator B edits only files ending `_annotator_B.csv`. Do not sort rows, rename columns,
or change any frozen field. The validator rejects such changes.

### BIRD worksheet

Each row is one `col = 'value'` literal from a gold SQL query. The CSV embeds the
question, its BIRD evidence sentence, and the offline context needed to judge the
protocol label without the database:

- `gold_sql` and `question_id`: the BIRD gold query the literal was taken from.
- `sql_tables`: the tables named in that query's `FROM`/`JOIN` clauses.
- `sql_column_ref` and `sql_column_table`: the column reference as written in the SQL
  (for example `T1.nationality`) and the table it resolves to after alias resolution.
  If this differs from `table`, the protocol attributed the literal to a different
  table than the one the query filters.
- `same_name_columns`: every `table.column` in the database schema whose column name
  equals `column`, each marked `present` or `absent` according to whether a cell equal
  to `protocol_gold` (case- and whitespace-insensitive) exists in it. The protocol picks
  the first `present` table in schema order, which is what makes wrong attribution
  possible when several tables share a column name.
- `matching_cell_values` and `column_distinct_count`: the raw cell value(s) in
  `table.column` that equal `protocol_gold`, and how many distinct non-null values that
  column holds (capped at 4000). `bird_column_values.csv` adds a sample of the other
  values. Full schemas are in `bird_schema.md`.

- `annotator_gold`: enter the database value that the question plus evidence refers to
  for the given column. Copy `protocol_gold` exactly when it is correct, enter the
  corrected value otherwise, or enter `NONE` if the question does not refer to that
  column. Use `gold_sql`, `sql_column_table`, and `same_name_columns` to decide whether
  the literal belongs to `table.column` at all: if the query filters a different table
  that also holds the value, the protocol's table attribution is wrong and `NONE` is
  the expected entry.
- `annotator_drift`: enter `1` when the selected value does not appear in the question
  text under a case-insensitive, whitespace-normalized substring check; otherwise enter
  `0`. Judge drift from the question, not from the evidence field.
- `note`: optional explanation; use `UNSURE: ...` when the item cannot be resolved.

The frozen protocol defines `protocol_gold` as the literal in the gold SQL, confirmed
present in the referenced column, and `protocol_drift` as whether the normalized gold
value is absent from the normalized question. "Present in the referenced column" is
witnessed offline by `matching_cell_values`; the SQLite databases themselves are not
shipped (they are the public BIRD dev_20240627 release).

### ICD worksheet

Each row contains a constructed clinical record, its billable ICD code and description,
the Wikidata node selected by the protocol, and offline code-table context:

- `record_code_in_icd10cm` and `record_code_icd10cm_desc`: whether `table_code` is a
  valid ICD-10-CM code in the artifact's code table and, if so, its official
  description. A valid code whose description matches `table_desc` has no corruption;
  a valid code whose description differs from `table_desc` was produced by corruption
  that happened to land on another real code.
- `record_category_code` and `record_category_title`: the first three characters of the
  record code and the ICD-10-CM category title for them (empty when no such category
  exists, which itself indicates corruption or an ICD-9 code).
- `nearest_valid_icd10cm`: up to five valid ICD-10-CM codes within edit distance 2 of
  the record code, with `d=` distance and description, closest first and description
  matches first. For ICD-9 records this list is noise and can be ignored.
- `node_wikidata_icd9_claim`: the node's ICD-9 code(s) on Wikidata; an `xvocab` record
  is expected to carry exactly one of these codes.
- `node_all_icd10_codes` and `n_valid_codes_under_node`: every ICD-10 code the KG
  holds for the node, and how many valid ICD-10-CM codes start with any of them. A node
  with several codes (27 of the 100 rows) may have had its record constructed from a
  code other than `protocol_gold_code`, which reports only the node's last-listed code.
  The codes are listed with descriptions in `icd_icd10cm_reference.csv`, tagged by
  `node_code`; `desc_equals_record_desc` flags the one whose description equals
  `table_desc`.

- Find `protocol_gold_qid` in `icd_wikidata_reference.csv`. The local reference contains
  the node label, description, aliases, Wikidata ICD-10 claims, source revision, and URL;
  no web lookup is required. ICD-9-CM descriptions are not shipped; the only ICD-9 facts
  available offline are the Wikidata claims above.
- `annotator_gold_qid`: copy `protocol_gold_qid` when it is the correct disease/category
  for the record; enter a corrected QID if one is evident from the supplied reference;
  or enter `NONE` if no node should match.
- `annotator_drift_type`: enter exactly one of `hier`, `corrupt`, `xvocab`, or `none`.
  `hier` means the record code is a descendant/specific code of the node's category;
  `corrupt` means the record code contains OCR/typographical corruption; `xvocab` means
  the record uses ICD-9 while the node uses ICD-10; `none` means no such drift applies.
- `note`: optional explanation; use `UNSURE: ...` when needed.

The protocol defines `protocol_gold_qid` as the Wikidata node whose ICD-10 code is the
category of the record's billable code; the drift type is assigned by construction. The
context columns let an annotator check the construction: `hier` records should be valid
codes under the node with a matching description, `corrupt` records should be invalid
or mis-described codes with a valid near neighbour under one of the node's codes, and `xvocab` records
should equal the node's ICD-9 claim.

## Validate progress and compute results

At any point, validate without creating or changing outputs:

```bash
python3 compute_agreement.py --validate-only
```

After both annotators have completed their files, run:

```bash
python3 compute_agreement.py
```

The command writes `agreement.json` and two adjudication CSVs. It reports:

- completed rows per annotator and paired completed rows;
- each annotator's protocol-gold and drift-label/type error rates;
- Cohen's kappa on the binary “protocol gold is correct” judgment;
- Cohen's kappa on the drift label/type; and
- the number of rows requiring joint adjudication.

Kappa is `null` when the data are empty or degenerate (for example, both annotators give
the same category on every row), because kappa is undefined in that case.

## Adjudication

The generated adjudication files include every paired row where annotators disagree or
where either annotator challenges a protocol label. Jointly fill:

- BIRD: `resolved_gold`, `resolved_drift`, and optionally `resolution_note`;
- ICD: `resolved_qid`, `resolved_type`, and optionally `resolution_note`.

Re-run `python3 compute_agreement.py`. Existing resolution fields are preserved when
the adjudication list is regenerated. Adjudicated error rates appear only when every
required resolution field is complete.

## Optional context rebuild

`build_offline_context.py` is a maintainer command. It needs the artifact checkout with
the BIRD dev databases and the ICD-10-CM code table, re-derives the 200 sampled items
from the same seeds as the experiment driver, refuses to write if any frozen field
differs, and regenerates the schema, column-value, and ICD-10-CM files plus the context
columns while preserving existing annotations:

```bash
python3 build_offline_context.py --artifact-root /path/to/relationalrag-tables
```

Regenerate `SOURCE_MANIFEST.sha256` afterwards.

## Optional snapshot refresh

Annotation is fully offline. Only maintainers who intentionally want newer Wikidata
metadata should run:

```bash
python3 refresh_wikidata_reference.py
```

This command requires internet access and rewrites the reference snapshot. Because the
source revisions and hashes then change, review the diff and regenerate
`SOURCE_MANIFEST.sha256` before redistribution. The sampled QIDs and protocol labels
are never changed by this command.
