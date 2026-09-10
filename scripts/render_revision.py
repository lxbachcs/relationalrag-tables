"""Render the revision's new tables/figures as LaTeX snippets from the harness JSON outputs.

Reads the mirrored server outputs (artifacts_server/artifacts) and writes generated/*.tex.
Nothing in the manuscript's new tables is typed by hand: every number is emitted here.

Run: python3 render_revision.py [artifacts_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ART = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/Users/xuanbachle/Desktop/05_Notes/2026/0-Under-submission/ICDE-2026-relational-rag-tables/artifacts_server/artifacts")
OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(exist_ok=True)


def J(rel):
    return json.loads((ART / rel).read_text())


def pm(mean, sd, d=3):
    """Text-mode cell: 0.862$\\pm$0.016 (used inside tabular rows)."""
    return f"\\msd{{{mean:.{d}f}}}{{{sd:.{d}f}}}"


def pmm(mean, sd, d=3):
    """Mode-safe macro body: works in text and inside math."""
    return f"\\msd{{{mean:.{d}f}}}{{{sd:.{d}f}}}"


def w(name, text):
    (OUT / name).write_text(text, encoding="utf-8")
    print("wrote", OUT / name)


# ------------------------------------------------------------------ E2: multi-seed
agg = J("multiseed/aggregate.json")
S = agg["scalars"]

def sc(key):
    return S[key]["mean"], S[key]["sd"]

rows = []
for name, key in (("Exact-match", "exact_match"), ("Learned", "learned"), ("Deep ensemble", "ensemble")):
    vals = [pm(*sc(f"matrix/verifier_discrimination/{key}/{m}")) for m in ("accuracy", "precision", "recall", "f1", "auroc", "brier")]
    rows.append(f"{name} & " + " & ".join(vals) + " \\\\")
w("tab_disc_rows.tex", "\n".join(rows) + "\n")

rows = []
for name, key in (("Exact match", "exact_match"), ("Learned", "learned"), ("\\textbf{Ensemble}", "ensemble")):
    e, es = sc(f"ece/{key}/ece"); m, ms = sc(f"ece/{key}/mce"); a, as_ = sc(f"risk_coverage_pairs/{key}/aurc")
    rows.append(f"Synthetic & {name} & {pm(e, es)} & {pm(m, ms)} & {pm(a, as_)} \\\\")
w("tab_cal_rows.tex", "\n".join(rows) + "\n")

cov = ["1.0", "0.9", "0.8", "0.7", "0.5"]
acc = [sc(f"risk_coverage_pairs/ensemble/accuracy_at/{c}") for c in cov]
sysacc = [sc(f"risk_coverage_system/accuracy_at/{c}") for c in cov]
w("tab_sel_rows.tex",
  "Pairs (ensemble) & " + " & ".join(pm(a, s) for a, s in acc) + " \\\\\n" +
  "System (answered) & " + " & ".join(pm(a, s) for a, s in sysacc) + " \\\\\n")

qa = {}
for key in ("exact_match", "no_verifier", "learned", "ensemble", "oracle"):
    qa[key] = {"em": sc(f"matrix/qa_by_verifier/{key}/em"),
               "abst": S[f"matrix/qa_by_verifier/{key}/abstentions"]["mean"],
               "task": {t: S[f"matrix/qa_by_verifier/{key}/by_task/{t}/em"]["mean"]
                        for t in ("single_table_lookup", "table_kg_join", "multi_hop_traverse", "bad_join_backtrack", "abstain")}}
rows = []
for name, key in (("Exact-match", "exact_match"), ("No verifier", "no_verifier"), ("Learned", "learned"), ("Deep ensemble", "ensemble"), ("Oracle (UB)", "oracle")):
    t = qa[key]["task"]
    rows.append(f"{name} & {t['single_table_lookup']:.2f} & {t['table_kg_join']:.2f} & {t['multi_hop_traverse']:.2f} & {t['bad_join_backtrack']:.2f} & {t['abstain']:.2f} & {pm(*qa[key]['em'])} & {qa[key]['abst']:.0f} \\\\")
w("tab_qa_rows.tex", "\n".join(rows) + "\n")

# thresholds / risk targets
tv = sc("validation_thresholds/tau_v/tau_v_selected")
f1s = sc("validation_thresholds/tau_v/test_at_selected/f1"); f15 = sc("validation_thresholds/tau_v/test_at_0.5/f1")
rt = agg["risk_targets"]
rows = [f"{a} & \\msd{{{rt[a]['test_coverage']['mean']:.3f}}}{{{rt[a]['test_coverage']['sd']:.3f}}} & \\msd{{{rt[a]['test_risk']['mean']:.3f}}}{{{rt[a]['test_risk']['sd']:.3f}}} & {rt[a]['risk_held_frac']*20:.0f}/20 \\\\"
        for a in ("0.02", "0.05", "0.1")]
w("tab_risktargets_rows.tex", "\n".join(rows) + "\n")
w("macros_e2.tex", "\n".join([
    f"\\newcommand{{\\tauSel}}{{\\msd{{{tv[0]:.2f}}}{{{tv[1]:.2f}}}}}",
    f"\\newcommand{{\\fOneSel}}{{\\msd{{{f1s[0]:.3f}}}{{{f1s[1]:.3f}}}}}",
    f"\\newcommand{{\\fOneFixed}}{{\\msd{{{f15[0]:.3f}}}{{{f15[1]:.3f}}}}}",
    f"\\newcommand{{\\emEns}}{{{pmm(*sc('matrix/qa_by_verifier/ensemble/em'))}}}",
    f"\\newcommand{{\\emExact}}{{{pmm(*sc('matrix/qa_by_verifier/exact_match/em'))}}}",
    f"\\newcommand{{\\emLearned}}{{{pmm(*sc('matrix/qa_by_verifier/learned/em'))}}}",
    f"\\newcommand{{\\emNoVer}}{{{pmm(*sc('matrix/qa_by_verifier/no_verifier/em'))}}}",
    f"\\newcommand{{\\aurocEns}}{{{pmm(*sc('matrix/verifier_discrimination/ensemble/auroc'))}}}",
    f"\\newcommand{{\\aurocLearned}}{{{pmm(*sc('matrix/verifier_discrimination/learned/auroc'))}}}",
    f"\\newcommand{{\\eceEns}}{{{pmm(*sc('ece/ensemble/ece'))}}}",
    f"\\newcommand{{\\eceLearned}}{{{pmm(*sc('ece/learned/ece'))}}}",
    f"\\newcommand{{\\aurcEns}}{{{pmm(*sc('risk_coverage_pairs/ensemble/aurc'))}}}",
    f"\\newcommand{{\\aurcLearned}}{{{pmm(*sc('risk_coverage_pairs/learned/aurc'))}}}",
    f"\\newcommand{{\\aurcExact}}{{{pmm(*sc('risk_coverage_pairs/exact_match/aurc'))}}}",
    f"\\newcommand{{\\aurcSys}}{{{pmm(*sc('risk_coverage_system/aurc'))}}}",
    f"\\newcommand{{\\utilFiveBest}}{{{pmm(*sc('cost_aware_utility/lambda_5/best_utility'))}}}",
    f"\\newcommand{{\\utilFiveFull}}{{{pmm(*sc('cost_aware_utility/lambda_5/full_utility'))}}}",
    f"\\newcommand{{\\covFiveBest}}{{{pmm(*sc('cost_aware_utility/lambda_5/best_coverage'), 2)}}}",
]) + "\n")

# risk-coverage figure coordinates (mean curve + 95% CI band across seeds)
curves = agg["curves_mean"]
def coords(fam, key="risk_mean"):
    return " ".join(f"({p['coverage']},{100 * (1 - p[key]):.2f})" for p in curves[fam])
def band(fam):
    lo = " ".join(f"({p['coverage']},{100 * (1 - p['risk_ci95'][1]):.2f})" for p in curves[fam])
    hi = " ".join(f"({p['coverage']},{100 * (1 - p['risk_ci95'][0]):.2f})" for p in reversed(curves[fam]))
    return lo + " " + hi
fig = r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=0.78\columnwidth, height=4.3cm,
  xlabel={Coverage answered}, ylabel={Accuracy on answered (\%)},
  xmin=0.05, xmax=1.0, ymin=55, ymax=101, x dir=reverse,
  xmajorgrids, ymajorgrids, grid style={dashed,gray!25},
  axis line style={grayA}, tick style={grayA},
  tick label style={font=\scriptsize}, label style={font=\scriptsize},
  legend style={font=\tiny, draw=none, fill=none, at={(0.97,0.04)}, anchor=south east, cells={anchor=west}},
]
\addplot[fill=blueA!15, draw=none, forget plot] coordinates {BAND_ENS} -- cycle;
\addplot[fill=greenA!15, draw=none, forget plot] coordinates {BAND_SYS} -- cycle;
\addplot[fill=grayA!20, draw=none, forget plot] coordinates {BAND_EXACT} -- cycle;
\addplot[blueA, line width=0.9pt, no marks] coordinates {ENS};
\addplot[orangeA, line width=0.9pt, no marks, dashed] coordinates {LEARNED};
\addplot[grayA, line width=0.9pt, no marks] coordinates {EXACT};
\addplot[greenA, line width=0.9pt, no marks] coordinates {SYS};
\legend{Ensemble (pairs), Single verifier (pairs), Exact match (pairs), Ensemble (system)}
\end{axis}
\end{tikzpicture}
\caption{Full risk--coverage curves (1\% coverage steps), mean over $20$ seeds with 95\% CI bands, for held-out join pairs and for answered QA questions (system level). AURC: exact \aurcExact, single \aurcLearned, ensemble \aurcEns\ (pairs); \aurcSys\ (system).}
\label{fig:riskcoverage}
\end{figure}
"""
fig = (fig.replace("BAND_ENS", band("pairs/ensemble")).replace("BAND_SYS", band("system/ensemble")).replace("BAND_EXACT", band("pairs/exact_match"))
       .replace("ENS", coords("pairs/ensemble")).replace("LEARNED", coords("pairs/learned")).replace("EXACT", coords("pairs/exact_match")).replace("SYS", coords("system/ensemble")))
w("fig_riskcoverage.tex", fig)

# ------------------------------------------------------------------ E3: theory check
th = J("theory_check.json")["aggregate"]
b = J("theory_check.json")["bird"]
t1, t2 = th["theorem1"], th["theorem2"]
w("tab_theory_rows.tex", "\n".join([
    f"Synthetic ($20$ seeds) & {pm(t1['pi_plus']['mean'], t1['pi_plus']['sd'])} & {pm(t1['alpha_fpr']['mean'], t1['alpha_fpr']['sd'])} & {pm(t1['beta_fnr']['mean'], t1['beta_fnr']['sd'])} & {pm(t1['lambda_star']['mean'], t1['lambda_star']['sd'], 2)} & {th['theorem1_qa_sign_agreement']['lambda_1']*20:.0f}/20 \\\\",
    f"BIRD value linking & {b['pi_plus']:.3f} & {b['alpha_fpr']:.4f} & {b['beta_fnr']:.3f} & {b['lambda_star']:.3f} & --- \\\\",
]) + "\n")
w("macros_e3.tex", "\n".join([
    f"\\newcommand{{\\thDelta}}{{{pmm(t2['delta']['mean'], t2['delta']['sd'])}}}",
    f"\\newcommand{{\\thR}}{{{pmm(t2['r_drifted']['mean'], t2['r_drifted']['sd'])}}}",
    f"\\newcommand{{\\thRzero}}{{{t2['r0_clean']['mean']:.4f}}}",
    f"\\newcommand{{\\thPred}}{{{pmm(t2['predicted_gain']['mean'], t2['predicted_gain']['sd'])}}}",
    f"\\newcommand{{\\thObs}}{{{pmm(t2['observed_join_em_gain']['mean'], t2['observed_join_em_gain']['sd'])}}}",
    f"\\newcommand{{\\thErr}}{{{t2['abs_error']['mean']:.3f}}}",
    f"\\newcommand{{\\thRatioObs}}{{{pmm(th['wrong_rate_ratio']['observed']['mean'], th['wrong_rate_ratio']['observed']['sd'], 2)}}}",
    f"\\newcommand{{\\thAlpha}}{{{t1['alpha_fpr']['mean']:.2f}}}",
    f"\\newcommand{{\\thLamStar}}{{{t1['lambda_star']['mean']:.2f}}}",
]) + "\n")

# ------------------------------------------------------------------ E1: BIRD linkers
e1 = J("e1/analysis.json")
order = [("Baseline (no pruning, no values)", "baseline"), ("none (pruning + values)", "pv_none"), ("exact", "pv_exact"),
         ("fuzzy", "pv_fuzzy"), ("embedding", "pv_embed"), ("learned verifier, $\\tau{=}0.5$", "pv_verifier"),
         ("learned verifier, $\\tau{=}0.8$", "pv_verifier_t80"), ("learned verifier, $\\tau{=}0.9$", "pv_verifier_t90"),
         ("oracle gold literals", "pv_oracle"), ("oracle, drifted literals only", "pv_oracle_driftonly"),
         ("verifier $\\tau{=}0.8$, drifted only", "pv_verifier_t80_driftonly")]
def dd(row, ref, key):
    v = row["vs"].get(ref, {}).get(key)
    if not v:
        return "---"
    return f"{100*v['delta']:+.1f} [{100*v['ci95'][0]:+.1f},{100*v['ci95'][1]:+.1f}]"
rows = []
for name, tag in order:
    r = e1[tag]
    prec = "---" if r["hints"]["questions_with_hints"] == 0 else f"{r['hints']['hint_precision']:.2f}"
    bold = tag == "pv_oracle"
    da, ddr = dd(r, "pv_none", "all"), dd(r, "pv_none", "drift")
    if bold:
        da, ddr = f"\\textbf{{{da}}}", f"\\textbf{{{ddr}}}"
    rows.append(f"{name} & {r['ex']:.3f} & {r['ex_drift']:.3f} & {r['hints']['questions_with_hints']} & {prec} & {da} & {ddr} \\\\")
w("tab_e1_rows.tex", "\n".join(rows) + "\n")
sel = e1["pv_verifier"]["selective_over_hinted"]["curve"]
w("macros_e1.tex", "\n".join([
    f"\\newcommand{{\\eOneOracleAll}}{{{100*e1['pv_oracle']['vs']['pv_none']['all']['delta']:+.1f}}}",
    f"\\newcommand{{\\eOneOracleDrift}}{{{100*e1['pv_oracle']['vs']['pv_none']['drift']['delta']:+.1f}}}",
    f"\\newcommand{{\\eOneSelFull}}{{{sel[0]['ex_answered']:.3f}}}",
    f"\\newcommand{{\\eOneSelHalf}}{{{sel[-1]['ex_answered']:.3f}}}",
    f"\\newcommand{{\\eOneBaseEx}}{{{e1['baseline']['ex']:.3f}}}",
    f"\\newcommand{{\\eOneBaseDrift}}{{{e1['baseline']['ex_drift']:.3f}}}",
    f"\\newcommand{{\\eOnePvEx}}{{{e1['pv_none']['ex']:.3f}}}",
    f"\\newcommand{{\\eOnePvDrift}}{{{e1['pv_none']['ex_drift']:.3f}}}",
]) + "\n")

# ------------------------------------------------------------------ E7: ICD end-to-end
e7 = J("e7/results.json")
import statistics
def util(g, lam):
    u = [p["gates"][g]["em"] - lam * p["gates"][g]["wrong_rate"] for p in e7["per_seed"]]
    return statistics.mean(u)
rows = []
for name, g in (("Exact code match", "exact_match"), ("Prefix / hierarchy rule", "prefix_rule"), ("Embedding cosine ($\\tau{=}0.62$)", "embedding"),
                ("No verifier", "no_verifier"), ("\\textbf{Learned verifier}", "verifier_mlp"), ("Oracle (UB)", "oracle")):
    a = e7["aggregate"][g]
    wr = f"{a['wrong_rate']['mean']:.3f}"
    if g == "verifier_mlp":
        wr = f"\\textbf{{{wr}}}"
    rows.append(f"{name} & {pm(a['em']['mean'], a['em']['sd'])} & {wr} & {a['false_abstain_rate']['mean']:.3f} & {util(g,1):.3f} & {util(g,2):.3f} & {util(g,5):.3f} \\\\")
w("tab_e7_rows.tex", "\n".join(rows) + "\n")
m0 = e7["per_seed"][0]["meta"]
rc = e7["system_risk_coverage_mlp"]
w("macros_e7.tex", "\n".join([
    f"\\newcommand{{\\eSevenNq}}{{{m0['n_questions']}}}",
    f"\\newcommand{{\\eSevenNtriples}}{{{m0['n_triples']:,}}}",
    f"\\newcommand{{\\eSevenWrongVer}}{{{100*e7['aggregate']['verifier_mlp']['wrong_rate']['mean']:.1f}}}",
    f"\\newcommand{{\\eSevenWrongNo}}{{{100*e7['aggregate']['no_verifier']['wrong_rate']['mean']:.1f}}}",
    f"\\newcommand{{\\eSevenWrongEmb}}{{{100*e7['aggregate']['embedding']['wrong_rate']['mean']:.1f}}}",
    f"\\newcommand{{\\eSevenEmVer}}{{{e7['aggregate']['verifier_mlp']['em']['mean']:.3f}}}",
    f"\\newcommand{{\\eSevenEmNo}}{{{e7['aggregate']['no_verifier']['em']['mean']:.3f}}}",
    f"\\newcommand{{\\eSevenUtilVer}}{{{util('verifier_mlp',5):.3f}}}",
    f"\\newcommand{{\\eSevenUtilNo}}{{{util('no_verifier',5):.3f}}}",
    f"\\newcommand{{\\eSevenAnsAcc}}{{{100*statistics.mean(r['accuracy_at']['1.0'] for r in rc):.1f}}}",
    f"\\newcommand{{\\eSevenAurc}}{{{statistics.mean(r['aurc'] for r in rc):.3f}}}",
]) + "\n")

# ------------------------------------------------------------------ E5: cascades
e5 = J("icd_kg/results_e5_cascade.json")["aggregate_recall@1"]
sig = J("icd_kg/results_e5_cascade.json")["verifier_vs_aggregate"]
rows = []
for name, k in (("Tuned cascade prefix$\\to$fuzzy$\\to$embedding", "cascade_prefix_fuzzy_embed"), ("Logistic fusion (same features)", "logistic"),
                ("Learned verifier (MLP ensemble)", "verifier"), ("Learned verifier + pool-relative features", "verifier_rel")):
    rows.append(f"{name} & {pm(e5[k]['mean'], e5[k]['sd'])} \\\\")
w("tab_e5_rows.tex", "\n".join(rows) + "\n")
w("macros_e5.tex", "\n".join([
    f"\\newcommand{{\\eFiveCascade}}{{{pmm(e5['cascade_prefix_fuzzy_embed']['mean'], e5['cascade_prefix_fuzzy_embed']['sd'])}}}",
    f"\\newcommand{{\\eFiveLogit}}{{{pmm(e5['logistic']['mean'], e5['logistic']['sd'])}}}",
    f"\\newcommand{{\\eFiveVer}}{{{pmm(e5['verifier']['mean'], e5['verifier']['sd'])}}}",
    f"\\newcommand{{\\eFiveVerRel}}{{{pmm(e5['verifier_rel']['mean'], e5['verifier_rel']['sd'])}}}",
    f"\\newcommand{{\\eFiveCascadeDelta}}{{{sig['cascade_prefix_fuzzy_embed']['mean_delta']:+.3f}}}",
]) + "\n")

# ------------------------------------------------------------------ E6: label noise
e6 = J("e6/results.json")
rows = []
for rho in ("0.00", "0.05", "0.10", "0.20"):
    bb = e6["bird"]["by_rho"][f"rho_{rho}"]
    ii = e6["icd"]["aggregate"][f"rho_{rho}"]
    rows.append(f"{rho} & {bb['recall@1_all']:.3f} / {bb['recall@1_drift']:.3f} & {100*bb['gate_at_0.5']['fp_rate']:.2f}\\% / {100*bb['gate_at_0.5']['fn_rate']:.1f}\\% & "
                f"{pm(ii['recall@1']['mean'], ii['recall@1']['sd'])} & {100*ii['fp_rate']['mean']:.2f}\\% / {100*ii['fn_rate']['mean']:.1f}\\% \\\\")
w("tab_e6_rows.tex", "\n".join(rows) + "\n")
print("done")
