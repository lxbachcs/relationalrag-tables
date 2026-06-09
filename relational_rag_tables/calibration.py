"""Calibration and system-level robustness analyses for the synthetic study.

These are the reviewer-requested diagnostics the controlled study did not
originally report:

* **ECE** --- expected calibration error of the verifier / ensemble on the
  held-out graded-corruption split, beyond the Brier score already reported.
* **End-to-end selective prediction** --- a risk-coverage curve over QA *answers*
  (system level), ranking each answered question by the uncertainty of its
  committed join, rather than over pairwise joins.
* **Faithfulness** --- the fraction of answers grounded in retrieved evidence.
* **Retrieval-noise robustness** --- dropping the gold candidate from the pool
  with probability ``p`` (so retrieval surfaces a plausible distractor but misses
  the gold) shows that a no-verifier planner returns a *wrong* answer while the
  verifier abstains. This is the regime idealized retrieval hides.

Run:  ``python3 -m relational_rag_tables.calibration --fixtures fixtures/synthetic``
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import datagen
from .benchmark import evaluate_verifier
from .index import build_entries
from .ingest import load_corpus
from .plan_execute import PlannerConfig, build_triple_index, exact_match, run_question
from .uncertainty import EnsembleVerifier
from .verify import DeterministicJoinVerifier, ExactMatchVerifier

_JOIN_TASKS = {"table_kg_join", "bad_join_backtrack", "multi_hop_traverse"}


def _ece(scores, labels, n_bins: int = 10):
    """Expected/maximum calibration error on confidence ``max(p, 1-p)``."""
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(scores, labels):
        conf = max(p, 1.0 - p)
        pred = 1 if p >= 0.5 else 0
        b = min(n_bins - 1, int(conf * n_bins))
        bins[b].append((conf, 1 if pred == y else 0))
    n = len(scores)
    ece = mce = 0.0
    rel = []
    for b in bins:
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(a for _, a in b) / len(b)
        gap = abs(acc - conf)
        ece += gap * len(b) / n
        mce = max(mce, gap)
        rel.append({"conf": round(conf, 4), "acc": round(acc, 4), "n": len(b)})
    return round(ece, 4), round(mce, 4), rel


def calibration(fixtures, seed: int = 0, k: int = 5) -> dict:
    corpus = load_corpus(fixtures)
    tau_v = float(corpus.config["tau_v"])
    train = datagen.load_verifier_pairs(fixtures)
    ev = datagen.load_verifier_eval(fixtures)
    ens = EnsembleVerifier.train(train, k=k, seed=seed)
    out = {}
    for name, v in (("learned", ens.members[0]), ("ensemble", ens), ("exact_match", ExactMatchVerifier())):
        scores = [v.score(a, b) for a, b, _ in ev]
        labels = [1 if m else 0 for _, _, m in ev]
        ece, mce, rel = _ece(scores, labels)
        m = evaluate_verifier(v, ev, tau_v=tau_v)
        out[name] = {"ece": ece, "mce": mce, "brier": m["brier"], "auroc": m["auroc"], "reliability": rel}
    print("CALIBRATION", json.dumps({k2: {x: out[k2][x] for x in ("ece", "mce", "brier", "auroc")} for k2 in out}))
    return out


def end_to_end_selective(fixtures, seed: int = 0, k: int = 5, coverages=(1.0, 0.9, 0.8, 0.7, 0.5)) -> dict:
    corpus = load_corpus(fixtures)
    entries = build_entries(corpus)
    tau_v = float(corpus.config["tau_v"]); top_k = int(corpus.config["top_k"])
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    idx = build_triple_index(corpus)
    cfg = PlannerConfig(tau_v=tau_v, top_k=top_k, use_verifier=True)
    items = []  # (uncertainty, correct) over ANSWERED (non-abstained) questions
    for q in corpus.questions:
        r = run_question(corpus, entries, q, ens, cfg, idx, None)
        if r.status != "completed":
            continue
        u = 0.0
        for e in r.trace:
            if e.operation.type == "join" and e.accepted:
                u = ens.uncertainty(str(e.operation.args["lhs_value"]), str(e.operation.args["rhs"]))
        items.append((u, 1 if exact_match(r.prediction, r.gold) else 0))
    items.sort(key=lambda t: t[0])
    n = len(items)
    curve = []
    for c in coverages:
        keep = max(1, int(round(n * c)))
        curve.append({"coverage": round(keep / n, 4),
                      "accuracy": round(sum(a for _, a in items[:keep]) / keep, 4)})
    print("E2E_SELECTIVE", json.dumps(curve), f"(n_answered={n})")
    return {"n_answered": n, "risk_coverage": curve}


def cost_aware_utility(fixtures, seed: int = 0, k: int = 5, lambdas=(1, 2, 5),
                       coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> dict:
    """Deployment trade-off: rank answered QA questions by committed-join uncertainty and
    defer the most-uncertain tail. With a wrong answer penalized by lambda and an
    abstention scored 0, the utility-maximizing coverage falls as wrong answers cost more,
    quantifying when abstention pays off."""
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
        items.append((u, 1 if exact_match(r.prediction, r.gold) else 0))
    items.sort(key=lambda t: t[0])
    n = len(items)
    out = {}
    for lam in lambdas:
        best_u, best_c, curve = -1e9, None, []
        for c in coverages:
            keep = max(1, int(round(n * c)))
            correct = sum(a for _, a in items[:keep]); wrong = keep - correct
            util = (correct - lam * wrong) / n
            curve.append([round(keep / n, 2), round(util, 4)])
            if util > best_u:
                best_u, best_c = util, round(keep / n, 2)
        out[f"lambda_{lam}"] = {"best_coverage": best_c, "best_utility": round(best_u, 4), "curve": curve}
    print("COST_UTILITY", json.dumps({k2: {"best_cov": out[k2]["best_coverage"], "best_util": out[k2]["best_utility"]} for k2 in out}))
    return {"n_answered": n, "by_lambda": out}


def faithfulness(fixtures, seed: int = 0, k: int = 5) -> dict:
    from .evaluate import evaluate_results
    from .plan_execute import run_all
    corpus = load_corpus(fixtures)
    entries = build_entries(corpus)
    tau_v = float(corpus.config["tau_v"]); top_k = int(corpus.config["top_k"])
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    cfg = PlannerConfig(tau_v=tau_v, top_k=top_k, use_verifier=True)
    res = run_all(corpus, entries, verifier=ens, cfg=cfg)
    pq = evaluate_results(corpus, res)["per_question"]
    faithful = sum(1 for q in pq if q["faithful"]) / len(pq)
    answered = [q for q in pq if q["status"] == "completed"]
    faithful_ans = sum(1 for q in answered if q["faithful"]) / len(answered) if answered else 0.0
    out = {"faithful_all": round(faithful, 4), "faithful_answered": round(faithful_ans, 4)}
    print("FAITHFULNESS", json.dumps(out))
    return out


def retrieval_noise(fixtures, seed: int = 0, k: int = 5, ps=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """Drop the gold candidate with prob ``p`` on questions that also carry a real
    distractor (bad-join backtrack). Reports EM / wrong-answer / abstention rates
    for the no-verifier ablation vs the ensemble."""
    corpus = load_corpus(fixtures)
    entries = build_entries(corpus)
    tau_v = float(corpus.config["tau_v"]); top_k = int(corpus.config["top_k"])
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    oracle = DeterministicJoinVerifier(corpus.join_labels)
    idx = build_triple_index(corpus)
    join_qs = [q for q in corpus.questions
               if q.task == "bad_join_backtrack" and q.gold_join and q.answer != "ABSTAIN"]
    out = {}
    for p in ps:
        rng = random.Random(1000 + int(p * 100))
        excl = {q.id: {q.gold_join["rhs"]} for q in join_qs if rng.random() < p}
        rows = {}
        for name, use_ver, ver in (("no_verifier", False, oracle), ("ensemble", True, ens)):
            cfg = PlannerConfig(tau_v=tau_v, top_k=top_k, use_verifier=use_ver)
            c = w = ab = 0
            for q in join_qs:
                r = run_question(corpus, entries, q, ver, cfg, idx, None, exclude=excl.get(q.id))
                if r.status == "abstained":
                    ab += 1
                elif exact_match(r.prediction, r.gold):
                    c += 1
                else:
                    w += 1
            nq = len(join_qs)
            rows[name] = {"em": round(c / nq, 4), "wrong": round(w / nq, 4), "abstain": round(ab / nq, 4)}
        out[f"{p:.2f}"] = rows
    print("RETRIEVAL_NOISE", json.dumps(out), f"(n_backtrack={len(join_qs)})")
    return {"n_backtrack": len(join_qs), "by_drop_prob": out}


def conformal_selective(fixtures, seed: int = 0, k: int = 5,
                        alphas=(0.01, 0.02, 0.05, 0.10)) -> dict:
    """Risk-controlled (split-conformal) selective prediction versus the deep ensemble.

    A reviewer asked whether conformal or risk-controlling selective classification is
    preferable to deep-ensemble abstention. We find the two are complementary rather
    than competing. Conformal selection supplies a target on the error of accepted
    joins that the ensemble's heuristic entropy threshold does not, and the ensemble
    supplies a better-ranked confidence score to conformalize. On a held-out
    calibration split we select, for each target risk ``alpha``, the largest accepted
    set whose calibration error is at most ``alpha`` (scanning from the most confident
    pair so the accepted set is a confidence super-level set), then report the achieved
    coverage and error on a disjoint test split. We conformalize both a single member
    and the full ensemble to isolate the value of the ensemble score."""
    ev = datagen.load_verifier_eval(fixtures)
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)

    def conf_err(v):
        s, e = [], []
        for a, b, m in ev:
            p = v.score(a, b)
            y = 1 if m else 0
            s.append(max(p, 1.0 - p))
            e.append(0 if (1 if p >= 0.5 else 0) == y else 1)
        return s, e

    rng = random.Random(seed)
    idx = list(range(len(ev)))
    rng.shuffle(idx)
    half = len(idx) // 2
    cal_idx, test_idx = idx[:half], idx[half:]

    def select(s, e, alpha):
        cal = sorted(((s[i], e[i]) for i in cal_idx), key=lambda t: -t[0])
        err = n = 0
        thr = 1.01  # accept nothing until the calibration error clears alpha
        for c, er in cal:
            err += er
            n += 1
            if err / n <= alpha:
                thr = c
        test = [(s[i], e[i]) for i in test_idx]
        acc = [er for c, er in test if c >= thr]
        cov = len(acc) / len(test)
        risk = (sum(acc) / len(acc)) if acc else 0.0
        return {"alpha": alpha, "threshold": round(thr, 4), "coverage": round(cov, 4),
                "test_risk": round(risk, 4), "risk_held": bool(not acc or risk <= alpha)}

    out = {}
    for name, v in (("single_conformal", ens.members[0]), ("ensemble_conformal", ens)):
        s, e = conf_err(v)
        out[name] = [select(s, e, a) for a in alphas]
    print("CONFORMAL", json.dumps({n2: [(r["alpha"], r["coverage"], r["test_risk"]) for r in out[n2]] for n2 in out}))
    return {"n_cal": len(cal_idx), "n_test": len(test_idx), "by_method": out}


def recalibration(fixtures, seed: int = 0, k: int = 5) -> dict:
    """Reviewer Q5: can recalibrating a single model close the ensemble's calibration gap
    at one-fifth the compute? We split the eval pairs into a calibration and a test half,
    fit temperature scaling and isotonic regression on the single verifier using only the
    calibration half, and compare test-half ECE against the raw single model and the deep
    ensemble. If a recalibrated single model matches the ensemble's ECE, the ensemble's
    calibration advantage is recoverable without its compute."""
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    ev = datagen.load_verifier_eval(fixtures)
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    single = ens.members[0]
    y = np.array([1 if m else 0 for *_, m in ev])
    s1 = np.array([single.score(a, b) for a, b, _ in ev])
    se = np.array([ens.score(a, b) for a, b, _ in ev])
    rng = random.Random(seed); idx = list(range(len(ev))); rng.shuffle(idx)
    half = len(idx) // 2; cal, test = np.array(idx[:half]), np.array(idx[half:])
    eps = 1e-6
    logit = np.log(np.clip(s1, eps, 1 - eps) / (1 - np.clip(s1, eps, 1 - eps)))

    def nll(T, ix):
        p = np.clip(1 / (1 + np.exp(-logit[ix] / T)), eps, 1 - eps)
        return -np.mean(y[ix] * np.log(p) + (1 - y[ix]) * np.log(1 - p))

    Ts = np.linspace(0.5, 3.0, 51)
    T = float(Ts[int(np.argmin([nll(t, cal) for t in Ts]))])
    p_temp = 1 / (1 + np.exp(-logit / T))
    iso = IsotonicRegression(out_of_bounds="clip").fit(s1[cal], y[cal])
    p_iso = iso.predict(s1)

    def ece_of(p):
        return _ece(p[test].tolist(), y[test].tolist())[0]

    out = {"single_raw": ece_of(s1), "single_temp_scaled": ece_of(p_temp),
           "single_isotonic": ece_of(p_iso), "ensemble": ece_of(se),
           "temperature": round(T, 3), "n_test": int(len(test))}
    print("RECALIBRATION", json.dumps(out))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="fixtures/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/calibration")
    a = ap.parse_args(argv)
    res = {
        "calibration": calibration(a.fixtures, seed=a.seed),
        "end_to_end_selective": end_to_end_selective(a.fixtures, seed=a.seed),
        "conformal_selective": conformal_selective(a.fixtures, seed=a.seed),
        "recalibration": recalibration(a.fixtures, seed=a.seed),
        "cost_aware_utility": cost_aware_utility(a.fixtures, seed=a.seed),
        "faithfulness": faithfulness(a.fixtures, seed=a.seed),
        "retrieval_noise": retrieval_noise(a.fixtures, seed=a.seed),
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
