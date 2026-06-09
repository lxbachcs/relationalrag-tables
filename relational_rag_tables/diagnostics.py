"""Reviewer-requested diagnostics on the seeded synthetic corpus.

* **Threshold sensitivity** --- precision/recall/F1 of the verifier as the
  acceptance threshold sweeps, showing stability bands around $\\tau_v{=}0.5$.
* **Compute-matched uncertainty** --- the $K$-member deep ensemble against two
  single-model alternatives at one-fifth the compute (single-model entropy and a
  score-margin heuristic), as a selective-prediction comparison per cost.
* **KG incompleteness** --- removing a fraction of graph edges and showing that
  the system degrades gracefully into abstentions rather than wrong answers.
* **Efficiency** --- stage-wise P50/P95 latency and index size.
* **Where gating hurts** --- the false-negative cost of the gate on clean joins,
  the price the verifier pays for safety.

Run:  ``python3 -m relational_rag_tables.diagnostics --fixtures fixtures/synthetic``
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from . import datagen
from .benchmark import evaluate_verifier
from .index import build_entries
from .ingest import load_corpus
from .plan_execute import PlannerConfig, build_triple_index, exact_match, run_question
from .uncertainty import EnsembleVerifier
from .verify import DeterministicJoinVerifier


def _bent(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def _pct(xs, q):
    return round(float(np.percentile(xs, q)), 3)


def threshold_sweep(fixtures, seed=0, k=5, taus=(0.3, 0.4, 0.5, 0.6, 0.7)):
    ev = datagen.load_verifier_eval(fixtures)
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    rows = {}
    for t in taus:
        m = evaluate_verifier(ens, ev, tau_v=t)
        rows[f"{t:.1f}"] = {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"]}
    print("THRESHOLD_SWEEP", json.dumps(rows))
    return rows


def cheap_uncertainty(fixtures, seed=0, k=5, coverages=(1.0, 0.9, 0.8, 0.7, 0.5)):
    corpus = load_corpus(fixtures)
    tau_v = float(corpus.config["tau_v"])
    ev = datagen.load_verifier_eval(fixtures)
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    single = ens.members[0]

    def curve(score_fn, unc_fn):
        items = []
        for a, b, m in ev:
            pred = 1 if score_fn(a, b) >= tau_v else 0
            items.append((unc_fn(a, b), int(pred == (1 if m else 0))))
        items.sort(key=lambda t: t[0])
        n = len(items)
        return [round(sum(c for _, c in items[: max(1, int(round(n * cv)))]) / max(1, int(round(n * cv))), 4)
                for cv in coverages]

    out = {
        "ensemble_K5_5x": curve(ens.score, ens.uncertainty),
        "single_entropy_1x": curve(single.score, lambda a, b: _bent(single.score(a, b))),
        "single_margin_1x": curve(single.score, lambda a, b: -abs(single.score(a, b) - 0.5)),
    }
    print("CHEAP_UNCERTAINTY", json.dumps({m: dict(zip([str(c) for c in coverages], v)) for m, v in out.items()}))
    return {"coverages": list(coverages), "curves": out}


def kg_incompleteness(fixtures, seed=0, k=5, ps=(0.0, 0.1, 0.2, 0.3)):
    corpus = load_corpus(fixtures)
    entries = build_entries(corpus)
    tau_v = float(corpus.config["tau_v"]); top_k = int(corpus.config["top_k"])
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    join_qs = [q for q in corpus.questions if q.task in {"table_kg_join", "multi_hop_traverse"}]
    out = {}
    for p in ps:
        rng = random.Random(2000 + int(p * 100))
        idx = {}
        for t in corpus.triples:
            if rng.random() >= p:
                idx.setdefault((t.subject, t.relation), t.object)
        cfg = PlannerConfig(tau_v=tau_v, top_k=top_k, use_verifier=True)
        c = w = ab = 0
        for q in join_qs:
            r = run_question(corpus, entries, q, ens, cfg, idx, None)
            if r.status == "abstained":
                ab += 1
            elif exact_match(r.prediction, r.gold):
                c += 1
            else:
                w += 1
        nq = len(join_qs)
        out[f"{p:.1f}"] = {"em": round(c / nq, 4), "wrong": round(w / nq, 4), "abstain": round(ab / nq, 4)}
    print("KG_INCOMPLETE", json.dumps(out), f"(n_join={len(join_qs)})")
    return {"n_join": len(join_qs), "by_drop": out}


def efficiency(fixtures, seed=0, k=5):
    corpus = load_corpus(fixtures)
    entries = build_entries(corpus)
    tau_v = float(corpus.config["tau_v"]); top_k = int(corpus.config["top_k"])
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    idx = build_triple_index(corpus)
    cfg = PlannerConfig(tau_v=tau_v, top_k=top_k, use_verifier=True)
    ret, exe, tot = [], [], []
    for q in corpus.questions:
        r = run_question(corpus, entries, q, ens, cfg, idx, None)
        ret.append(r.latency_ms["retrieval"]); exe.append(r.latency_ms["execution"]); tot.append(r.latency_ms["total"])
    out = {"index_entries": len(entries), "n": len(tot),
           "retrieval_ms": {"p50": _pct(ret, 50), "p95": _pct(ret, 95)},
           "execution_ms": {"p50": _pct(exe, 50), "p95": _pct(exe, 95)},
           "total_ms": {"p50": _pct(tot, 50), "p95": _pct(tot, 95)}}
    print("EFFICIENCY", json.dumps(out))
    return out


def gating_hurts(fixtures, seed=0, k=5):
    corpus = load_corpus(fixtures)
    entries = build_entries(corpus)
    tau_v = float(corpus.config["tau_v"]); top_k = int(corpus.config["top_k"])
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    oracle = DeterministicJoinVerifier(corpus.join_labels)
    idx = build_triple_index(corpus)
    join_qs = [q for q in corpus.questions if q.task == "table_kg_join"]
    em = {"no_verifier": 0, "ensemble": 0}; hurt = 0
    for q in join_qs:
        rn = run_question(corpus, entries, q, oracle, PlannerConfig(tau_v=tau_v, top_k=top_k, use_verifier=False), idx, None)
        re = run_question(corpus, entries, q, ens, PlannerConfig(tau_v=tau_v, top_k=top_k, use_verifier=True), idx, None)
        ok_n = exact_match(rn.prediction, rn.gold); ok_e = exact_match(re.prediction, re.gold)
        em["no_verifier"] += ok_n; em["ensemble"] += ok_e
        if ok_n and not ok_e:
            hurt += 1
    nq = len(join_qs)
    out = {"n_join": nq, "em_no_verifier": round(em["no_verifier"] / nq, 4),
           "em_ensemble": round(em["ensemble"] / nq, 4), "gating_hurts_cases": hurt,
           "gating_hurts_rate": round(hurt / nq, 4)}
    print("GATING_HURTS", json.dumps(out))
    return out


def _auroc_np(p, y):
    import numpy as np
    p = np.asarray(p, float); y = np.asarray(y)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(p, kind="mergesort"); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def feature_ablation(fixtures, seed=0):
    """Train the verifier on feature subsets to show the learned gain is not a single
    obvious rule. The generator preserves the numeric key, so digit agreement is the
    dominant synthetic signal; on messier real data the full feature set matters more."""
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .features import join_features
    tr = datagen.load_verifier_pairs(fixtures); ev = datagen.load_verifier_eval(fixtures)
    Xtr = np.array([join_features(a, b) for a, b, _ in tr], "float32"); ytr = np.array([1 if m else 0 for *_, m in tr])
    Xev = np.array([join_features(a, b) for a, b, _ in ev], "float32"); yev = np.array([1 if m else 0 for *_, m in ev])
    # indices into FEATURE_NAMES: 2 digit_suffix, 3 digit_full, 5 edit_ratio, 7 prefix, 8 embed_cosine
    subsets = {"digit_only": [2, 3], "edit_only": [5], "prefix_only": [7], "embed_only": [8], "full": list(range(10))}
    out = {}
    for name, idx in subsets.items():
        clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=600, random_state=seed))
        clf.fit(Xtr[:, idx], ytr)
        p = clf.predict_proba(Xev[:, idx])[:, 1]
        out[name] = {"acc": round(float(((p >= 0.5).astype(int) == yev).mean()), 4), "auroc": round(_auroc_np(p, yev), 4)}
    print("FEATURE_ABLATION", json.dumps(out))
    return out


def fellegi_sunter(fixtures, seed=0):
    """Classical Fellegi-Sunter probabilistic record linkage as a training-light
    baseline, requested by reviewers who wanted a strong non-neural matcher in the
    same harness. We binarize each join feature into a field-agreement indicator,
    estimate the agreement probabilities under a match (``m``) and a non-match
    (``u``) from the labelled training pairs with Laplace smoothing, and score a
    pair by the summed log-likelihood ratio. We rank the same evaluation pairs and
    compare AUROC against the learned MLP verifier on the identical features."""
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .features import join_features, FEATURE_NAMES
    tr = datagen.load_verifier_pairs(fixtures); ev = datagen.load_verifier_eval(fixtures)
    Xtr = np.array([join_features(a, b) for a, b, _ in tr], "float64"); ytr = np.array([1 if m else 0 for *_, m in tr])
    Xev = np.array([join_features(a, b) for a, b, _ in ev], "float64"); yev = np.array([1 if m else 0 for *_, m in ev])
    A, Aev = (Xtr >= 0.5).astype(int), (Xev >= 0.5).astype(int)
    pos, neg = A[ytr == 1], A[ytr == 0]
    m = (pos.sum(0) + 1) / (len(pos) + 2)   # P(agree | match)
    u = (neg.sum(0) + 1) / (len(neg) + 2)   # P(agree | non-match)
    w_agree, w_disagree = np.log(m / u), np.log((1 - m) / (1 - u))
    fs = (Aev * w_agree + (1 - Aev) * w_disagree).sum(1)
    clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=600, random_state=seed))
    clf.fit(Xtr, ytr); p = clf.predict_proba(Xev)[:, 1]
    out = {"auroc_fs": round(_auroc_np(fs, yev), 4),
           "acc_fs": round(float(((fs > 0).astype(int) == yev).mean()), 4),
           "auroc_mlp": round(_auroc_np(p, yev), 4),
           "top_fields": [FEATURE_NAMES[i] for i in np.argsort(-np.abs(w_agree))[:3].tolist()]}
    print("FELLEGI_SUNTER", json.dumps(out))
    return out


def composite_key(fixtures, seed=0, k=5, n=400):
    """Composite (multi-attribute) key joins with AND/OR composition and conflicting
    components, requested by reviewers. The verifier is pairwise, but a composite key
    over two attributes is scored by combining the per-attribute match probabilities,
    by a probabilistic AND (product, for keys where both components must agree) or a
    probabilistic OR (``1-(1-s1)(1-s2)``, for alternate keys where either component
    identifies). We evaluate positives (both attributes drift yet agree), conflicts /
    hard negatives (one attribute agrees, one does not), and easy negatives (neither
    agrees). AND accepts above ``tau_v^2``, OR above ``tau_v``. The two rules encode
    opposite key semantics: AND rejects a conflict, OR treats one matching component as
    sufficient, so the composition operator is a modeling choice, not a fixed recipe."""
    import numpy as np
    from .datagen import _corrupt
    from .verify import ExactMatchVerifier
    ens = EnsembleVerifier.train(datagen.load_verifier_pairs(fixtures), k=k, seed=seed)
    exact = ExactMatchVerifier()
    rng = random.Random(seed)

    def attr(base_idx, agree):
        base = f"ACC{2000 + base_idx}"
        src = base if agree else f"ACC{2000 + (base_idx + 537) % n}"
        return f"ACC-{2000 + base_idx}", f"account:{_corrupt(src, rng.choice([1, 2, 3]), rng)}"

    rows = []  # (s_prod, s_or, s_single, exact_and, label, kind)
    for kind, a1ok, a2ok, label in (("pos", True, True, 1), ("conflict", True, False, 0), ("easyneg", False, False, 0)):
        for i in range(n):
            l1, r1 = attr(i, a1ok); l2, r2 = attr(i, a2ok)
            s1, s2 = ens.score(l1, r1), ens.score(l2, r2)
            e_and = 1.0 if (exact.score(l1, r1) >= 0.5 and exact.score(l2, r2) >= 0.5) else 0.0
            rows.append((s1 * s2, 1.0 - (1.0 - s1) * (1.0 - s2), s1, e_and, label, kind))
    y = np.array([r[4] for r in rows])

    def rate(kind, sel):
        sub = [r for r in rows if r[5] == kind]
        return round(sum(1 for r in sub if sel(r)) / len(sub), 4)

    out = {"auroc_product": round(_auroc_np([r[0] for r in rows], y), 4),
           "auroc_or": round(_auroc_np([r[1] for r in rows], y), 4),
           "auroc_single_attr": round(_auroc_np([r[2] for r in rows], y), 4),
           "recall_pos_product": rate("pos", lambda r: r[0] >= 0.25),
           "recall_pos_or": rate("pos", lambda r: r[1] >= 0.5),
           "recall_pos_exact": rate("pos", lambda r: r[3] >= 0.5),
           "conflict_accept_product": rate("conflict", lambda r: r[0] >= 0.25),
           "conflict_accept_or": rate("conflict", lambda r: r[1] >= 0.5),
           "conflict_accept_exact": rate("conflict", lambda r: r[3] >= 0.5)}
    print("COMPOSITE_KEY", json.dumps(out))
    return out


def scalability(sizes=(600, 1200, 2400, 4800, 9600), seed=7, drift=0.6, sample_q=150):
    """Index build time, index size, and per-question latency as the corpus grows.
    The pure-Python retriever is a linear scan, so latency rises with corpus size,
    which is precisely the regime where approximate search / sharding takes over."""
    import shutil
    import tempfile
    import time
    from .datagen import GenConfig
    from .index import build_entries
    from .ingest import load_corpus
    from .verify import DeterministicJoinVerifier
    out = []
    for n in sizes:
        d = tempfile.mkdtemp()
        try:
            datagen.write_corpus(datagen.generate(GenConfig(num_accounts=n, seed=seed, qa_join_drift=drift)), d)
            corpus = load_corpus(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        t0 = time.perf_counter(); entries = build_entries(corpus); build_s = time.perf_counter() - t0
        idx = build_triple_index(corpus)
        ver = DeterministicJoinVerifier(corpus.join_labels)
        cfg = PlannerConfig(tau_v=0.5, top_k=8, use_verifier=True)
        lat = [run_question(corpus, entries, q, ver, cfg, idx, None).latency_ms["total"]
               for q in corpus.questions[:sample_q]]
        row = {"accounts": n, "questions": len(corpus.questions), "index_entries": len(entries),
               "build_s": round(build_s, 2), "lat_p50_ms": _pct(lat, 50), "lat_p95_ms": _pct(lat, 95)}
        out.append(row); print("SCALE", json.dumps(row))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="fixtures/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/diagnostics")
    a = ap.parse_args(argv)
    res = {"threshold_sweep": threshold_sweep(a.fixtures, seed=a.seed),
           "cheap_uncertainty": cheap_uncertainty(a.fixtures, seed=a.seed),
           "kg_incompleteness": kg_incompleteness(a.fixtures, seed=a.seed),
           "efficiency": efficiency(a.fixtures, seed=a.seed),
           "gating_hurts": gating_hurts(a.fixtures, seed=a.seed),
           "feature_ablation": feature_ablation(a.fixtures, seed=a.seed),
           "fellegi_sunter": fellegi_sunter(a.fixtures, seed=a.seed),
           "composite_key": composite_key(a.fixtures, seed=a.seed)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
