# Completed audit results

`agreement.json` is the final output of `compute_agreement.py` after two annotators
independently completed all 200 items from the package alone and every adjudication row
was jointly resolved (2026-09-11). `resolved_bird.csv` and `resolved_icd.csv` list the
19 items that went to adjudication with the protocol label, the resolved label, and the
resolution note. The individual annotator worksheets are not distributed; the worksheets
in the parent folder are blank so that the audit can be repeated.

| | BIRD | ICD |
|---|---|---|
| Paired items | 100 | 100 |
| Cohen's kappa, protocol gold correct | 0.811 | undefined (every node accepted by both) |
| Cohen's kappa, drift label / type | 1.0 | 1.0 |
| Adjudicated protocol error, gold / drift | 0.12 / 0.00 | 0.00 / 0.05 |

The 12 BIRD gold errors are table-attribution errors: the literal and drift flag were
right, but the extractor took the first same-named column in schema order instead of the
alias-resolved table the gold SQL filters. The 5 ICD drift-type errors are records labelled
`corrupt` whose OCR substitution was a no-op, leaving a valid descendant code (`hier`).
