"""Experiment E3: numerical check of Theorems 1-2 against measured verifier error rates.

Theorem 1 (abstention utility) is conditional on the verifier's false-positive rate alpha
and false-negative rate beta and the match prevalence pi+. We measure these on the
held-out pair split of every multi-seed corpus, derive the critical penalty
lambda* = pi+ beta / ((1 - pi+)(1 - alpha)) above which gating must help, and check the
sign of the prediction against the QA-level utility of the ungated planner versus the
verifier-gated planner (Corollary: gated wrong-join rate = (1 - pi+) alpha vs (1 - pi+)).

Theorem 2 (regime separation) predicts the join-recall gain delta * r - (1 - delta)(1 - r0)
from the drift fraction delta, the verifier's recall r on drifted gold joins and r0 on
clean gold joins. We measure r, r0, delta directly on each corpus's gold join pairs and
compare the predicted gain with the observed EM gap on the join task family.

Run after ``multiseed``:  ``python -m relational_rag_tables.theory_check``
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import json
import math
from pathlib import Path

from . import datagen
from .index import normalize_identifier
from .ingest import load_corpus
from .uncertainty import EnsembleVerifier


def _stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"mean": round(m, 4), "sd": round(sd, 4), "n": n}


def check_seed(seed_json: Path, fixtures_root: Path, k: int, lambdas=(1, 2, 5)) -> dict:
    res = json.loads(seed_json.read_text())
    seed = res["seed"]
    fx = fixtures_root / f"seed_{seed}"
    corpus = load_corpus(fx)
    tau_v = float(corpus.config["tau_v"])
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fx), k=k, seed=seed)

    # --- Theorem 1 inputs on held-out pairs
    conf = res["matrix"]["verifier_discrimination"]["ensemble"]["confusion"]
    n = sum(conf.values())
    pos = conf["tp"] + conf["fn"]
    neg = conf["fp"] + conf["tn"]
    alpha = conf["fp"] / neg if neg else 0.0
    beta = conf["fn"] / pos if pos else 0.0
    pi = pos / n
    lam_star = pi * beta / ((1 - pi) * (1 - alpha)) if (1 - pi) * (1 - alpha) > 0 else float("inf")
    pair_level = {}
    for lam in lambdas:
        u_no = pi - lam * (1 - pi)
        u_gate = pi * (1 - beta) - lam * (1 - pi) * alpha
        pair_level[f"lambda_{lam}"] = {"u_ungated": round(u_no, 4), "u_gated": round(u_gate, 4),
                                       "gating_helps_predicted": bool(lam > lam_star)}
    # --- Theorem 1 / Corollary check at QA level: ungated planner vs gated planner
    qa = res["matrix"]["qa_by_verifier"]
    nq = sum(t["total"] for t in qa["ensemble"]["by_task"].values())
    def rates(row):
        em = row["em"]
        abst = row["abstentions"] / nq
        return em, abst, max(0.0, 1 - em - abst)
    em_no, ab_no, wr_no = rates(qa["no_verifier"])
    em_ge, ab_ge, wr_ge = rates(qa["ensemble"])
    qa_level = {"ungated": {"em": em_no, "abstain": round(ab_no, 4), "wrong": round(wr_no, 4)},
                "gated": {"em": em_ge, "abstain": round(ab_ge, 4), "wrong": round(wr_ge, 4)},
                "wrong_rate_ratio_observed": round(wr_ge / wr_no, 4) if wr_no else None,
                "wrong_rate_ratio_predicted_alpha": round(alpha, 4)}
    for lam in lambdas:
        u_no = em_no - lam * wr_no
        u_ge = em_ge - lam * wr_ge
        qa_level[f"lambda_{lam}"] = {"u_ungated": round(u_no, 4), "u_gated": round(u_ge, 4),
                                     "gating_helps_observed": bool(u_ge > u_no),
                                     "gating_helps_predicted": bool(lam > lam_star),
                                     "sign_agrees": bool((u_ge > u_no) == (lam > lam_star))}

    # --- Theorem 2: r, r0, delta on the corpus's gold join pairs
    gold_pos = [(lhs, rhs) for (lhs, rhs), lab in corpus.join_labels.items() if lab]
    def drifted(lhs, rhs):
        a, b = normalize_identifier(lhs), normalize_identifier(rhs)
        return not (a == b or a in b or b in a)
    drift_pairs = [(l, r) for l, r in gold_pos if drifted(l, r)]
    clean_pairs = [(l, r) for l, r in gold_pos if not drifted(l, r)]
    r = sum(ens.score(l, rr) >= tau_v for l, rr in drift_pairs) / len(drift_pairs) if drift_pairs else None
    r0 = sum(ens.score(l, rr) >= tau_v for l, rr in clean_pairs) / len(clean_pairs) if clean_pairs else None
    delta = len(drift_pairs) / len(gold_pos) if gold_pos else None
    predicted_gain = delta * r - (1 - delta) * (1 - r0) if None not in (r, r0, delta) else None
    join_em = {v: qa[v]["by_task"].get("table_kg_join", {}).get("em") for v in ("exact_match", "ensemble", "learned")}
    observed_gain = join_em["ensemble"] - join_em["exact_match"] if None not in join_em.values() else None
    theorem2 = {"n_gold_pairs": len(gold_pos), "delta": round(delta, 4), "r_drifted": round(r, 4), "r0_clean": round(r0, 4),
                "exact_recall_predicted": round(1 - delta, 4), "exact_join_em_observed": join_em["exact_match"],
                "predicted_gain": round(predicted_gain, 4), "observed_join_em_gain": round(observed_gain, 4),
                "abs_error": round(abs(predicted_gain - observed_gain), 4),
                "condition_r_gt_bound": bool(r > (1 - r0) * (1 - delta) / delta) if delta else None}
    return {"seed": seed, "theorem1": {"n_pairs": n, "pi_plus": round(pi, 4), "alpha_fpr": round(alpha, 4),
                                       "beta_fnr": round(beta, 4), "lambda_star": round(lam_star, 4),
                                       "pair_level": pair_level, "qa_level": qa_level},
            "theorem2": theorem2}


def bird_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    f = d.get("failure_test")
    if not f:
        return None
    n = f["tp"] + f["fp"] + f["tn"] + f["fn"]
    pi = (f["tp"] + f["fn"]) / n
    alpha, beta = f["fp_rate"], f["fn_rate"]
    lam_star = pi * beta / ((1 - pi) * (1 - alpha))
    return {"dataset": "BIRD value linking (held-out pairs)", "n_pairs": n, "pi_plus": round(pi, 4),
            "alpha_fpr": alpha, "beta_fnr": beta, "lambda_star": round(lam_star, 4),
            "note": "candidate pools are large, so pi+ is small and lambda* is tiny: gating helps for any lambda >= 1"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiseed", default="artifacts/multiseed")
    ap.add_argument("--fixtures-root", default="fixtures/multiseed")
    ap.add_argument("--ensemble-k", type=int, default=5)
    ap.add_argument("--out", default="artifacts/theory_check.json")
    a = ap.parse_args(argv)
    seeds = sorted(Path(a.multiseed).glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    per = [check_seed(p, Path(a.fixtures_root), a.ensemble_k) for p in seeds]
    t1 = [p["theorem1"] for p in per]
    t2 = [p["theorem2"] for p in per]
    agg = {
        "theorem1": {k: _stats([x[k] for x in t1]) for k in ("pi_plus", "alpha_fpr", "beta_fnr", "lambda_star")},
        "theorem1_qa_sign_agreement": {f"lambda_{l}": round(sum(x["qa_level"][f"lambda_{l}"]["sign_agrees"] for x in t1) / len(t1), 4)
                                       for l in (1, 2, 5)},
        "theorem1_qa_gating_helps_observed": {f"lambda_{l}": round(sum(x["qa_level"][f"lambda_{l}"]["gating_helps_observed"] for x in t1) / len(t1), 4)
                                              for l in (1, 2, 5)},
        "wrong_rate_ratio": {"observed": _stats([x["qa_level"]["wrong_rate_ratio_observed"] for x in t1 if x["qa_level"]["wrong_rate_ratio_observed"] is not None]),
                             "predicted_alpha": _stats([x["alpha_fpr"] for x in t1])},
        "theorem2": {k: _stats([x[k] for x in t2]) for k in ("delta", "r_drifted", "r0_clean", "predicted_gain", "observed_join_em_gain", "abs_error")},
        "theorem2_condition_holds_frac": round(sum(bool(x["condition_r_gt_bound"]) for x in t2) / len(t2), 4),
    }
    out = {"n_seeds": len(per), "aggregate": agg, "bird": bird_row(Path("artifacts/bird/results.json")), "per_seed": per}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))
    print("bird:", json.dumps(out["bird"]))
    return out


if __name__ == "__main__":
    main()
