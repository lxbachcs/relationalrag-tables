"""Split-conformal selective prediction at the *system*
(QA-answer) level, not only on pairwise join classification.

We evaluate conformalized selective prediction for the
planner's *answer-level* abstentions (defer the whole QA answer), and how it compares
with plain entropy-thresholding for coverage--risk control. We rank each answered
question by the uncertainty of its committed join (the same signal the planner already
exposes), split the answered questions into a calibration and a disjoint test half,
and for each target answer-error ``alpha`` accept the largest low-uncertainty superlevel
set whose calibration error is at most ``alpha``. We then report the test coverage and
realized answer error. The entropy-thresholding baseline instead fixes a single
uncertainty cut (independent of any target) and reports the (coverage, risk) point it
lands on -- it ranks but does not *control* risk to a target.

Run: ``python3 -m relational_rag_tables.selective_prediction --fixtures fixtures/synthetic``
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import datagen
from .index import build_entries
from .ingest import load_corpus
from .plan_execute import PlannerConfig, build_triple_index, exact_match, run_question
from .uncertainty import EnsembleVerifier

RESULTS = Path(__file__).resolve().parents[1] / "artifacts" / "calibration" / "results_conformal_qa.json"


def _answered_items(fixtures, seed: int, k: int):
    """(uncertainty, correct) over every ANSWERED (non-abstained) question, where the
    uncertainty is the ensemble entropy of the question's committed join."""
    corpus = load_corpus(fixtures)
    entries = build_entries(corpus)
    tau_v = float(corpus.config["tau_v"]); top_k = int(corpus.config["top_k"])
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    idx = build_triple_index(corpus)
    cfg = PlannerConfig(tau_v=tau_v, top_k=top_k, use_verifier=True)
    items = []
    for q in corpus.questions:
        r = run_question(corpus, entries, q, ens, cfg, idx, None)
        if r.status != "completed":
            continue
        u = 0.0
        for e in r.trace:
            if e.operation.type == "join" and e.accepted:
                u = ens.uncertainty(str(e.operation.args["lhs_value"]), str(e.operation.args["rhs"]))
        items.append((float(u), 1 if exact_match(r.prediction, r.gold) else 0))
    return items


def conformal_qa_selective(fixtures, seed: int = 0, k: int = 5,
                           alphas=(0.05, 0.10, 0.15, 0.20),
                           entropy_cuts=(0.10, 0.30, 0.50, 0.70)) -> dict:
    items = _answered_items(fixtures, seed, k)
    rng = random.Random(seed)
    order = list(range(len(items))); rng.shuffle(order)
    half = len(order) // 2
    cal = [items[i] for i in order[:half]]
    test = [items[i] for i in order[half:]]

    def select_conformal(alpha):
        # accept low-uncertainty pairs first (super-level set on confidence = -uncertainty)
        cal_sorted = sorted(cal, key=lambda t: t[0])
        err = n = 0; thr = -1.0  # accept nothing until cal error clears alpha
        for u, ok in cal_sorted:
            err += (1 - ok); n += 1
            if err / n <= alpha:
                thr = u
        acc = [(1 - ok) for u, ok in test if u <= thr]
        cov = len(acc) / len(test) if test else 0.0
        risk = (sum(acc) / len(acc)) if acc else 0.0
        return {"alpha": alpha, "u_threshold": round(thr, 4), "coverage": round(cov, 4),
                "test_risk": round(risk, 4), "risk_held": bool(not acc or risk <= alpha + 1e-9)}

    def entropy_point(cut):
        acc = [(1 - ok) for u, ok in test if u <= cut]
        cov = len(acc) / len(test) if test else 0.0
        risk = (sum(acc) / len(acc)) if acc else 0.0
        return {"u_cut": cut, "coverage": round(cov, 4), "test_risk": round(risk, 4)}

    full_risk = round(sum(1 - ok for _, ok in test) / len(test), 4) if test else 0.0
    out = {
        "n_answered": len(items), "n_cal": len(cal), "n_test": len(test),
        "full_coverage_risk": full_risk,
        "conformal": [select_conformal(a) for a in alphas],
        "entropy_threshold": [entropy_point(c) for c in entropy_cuts],
    }
    print("CONFORMAL_QA", json.dumps({"full_risk": full_risk,
          "conformal": [(r["alpha"], r["coverage"], r["test_risk"]) for r in out["conformal"]],
          "entropy": [(r["u_cut"], r["coverage"], r["test_risk"]) for r in out["entropy_threshold"]]}))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", RESULTS)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="fixtures/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    conformal_qa_selective(args.fixtures, seed=args.seed, k=args.k)
