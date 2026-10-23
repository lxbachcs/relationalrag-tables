"""Experiment E5: hand-designed rule cascades vs the learned verifier on heterogeneous ICD drift.

Reviewer 2 notes that simple rules tie or beat the verifier in single-regime settings. The
question for the heterogeneous column (hierarchy + OCR + cross-vocabulary drift mixed) is
whether a practitioner could match the learned fusion by *combining* those rules by hand.
We build the strongest fixed combinations we can, tune their thresholds and order on the
TRAIN split only, and compare on the test split with a paired bootstrap:

  prefix->fuzzy->embed   fixed-order cascade: prefix if it fires, else fuzzy if above a
                         tuned threshold, else embedding
  best-order cascade     the cascade order and thresholds that maximise train recall@1
  max-normalised         argmax over the max of per-rule scores after min-max scaling
  sum-normalised         argmax over the sum of scaled rule scores (equal-weight fusion)
  logistic (linear)      logistic regression on the same six features (a linear learned
                         fusion; isolates non-linearity of the MLP)
  verifier (MLP ens.)    the paper's learned verifier

Run: ``.venv/bin/python -m relational_rag_real.run_e5_cascade --seeds 0,1,2,3,4``
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

from . import icd_kg as K
from .embed import embed
from .run_icd_hetero import build
from .run_icd_kg import features

RESULTS = Path(__file__).resolve().parents[1] / "artifacts" / "icd_kg" / "results_e5_cascade.json"
# feature layout of run_icd_kg.features: [exact, prefix, common_prefix, edit, cos, desc_jaccard]
RULES = {"prefix": lambda F: F[:, 1] + 0.01 * F[:, 2], "fuzzy": lambda F: F[:, 3], "embed": lambda F: F[:, 4]}


def _scale(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _cascade(F, order, thr):
    """Return the chosen index: first rule in ``order`` whose best score clears its threshold."""
    for rule in order:
        s = RULES[rule](F)
        j = int(np.argmax(s))
        if s[j] >= thr[rule]:
            return j
    return int(np.argmax(RULES[order[-1]](F)))


def _fusion(F, kind):
    S = np.stack([_scale(RULES[r](F)) for r in ("prefix", "fuzzy", "embed")], axis=1)
    return int(np.argmax(S.max(axis=1) if kind == "max" else S.sum(axis=1)))


def run_seed(seed: int, test_frac: float = 0.4) -> dict:
    recs, kg = build(seed=seed)
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    ne = embed([f"{n['label']} {n['code']}" for n in kg])
    re_ = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)
    rng = random.Random(seed)
    idx = list(range(len(recs)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    Fall = {i: np.array([features(recs[i], re_[i], kg[j], ne[j]) for j in range(len(kg))], "float32")
            for i in idx}
    gold = {i: qi[recs[i].gold_qid] for i in idx}

    # learned models on the train split (same hard-negative protocol as run_icd_hetero)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for i in tr:
        gi = gold[i]
        X.append(Fall[i][gi]); y.append(1)
        for j in [k for k in np.argsort(-(ne @ re_[i]))[:8] if k != gi][:4] + rng.sample(range(len(kg)), 4):
            X.append(Fall[i][j]); y.append(0)
    X, y = np.array(X, "float32"), np.array(y)
    members = []
    rn = np.random.default_rng(seed)
    for m in range(5):
        b = rn.integers(0, len(X), len(X))
        c = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=[(32, 16), (24,), (40, 20), (16, 16), (48, 24)][m], alpha=1e-3, max_iter=600, random_state=m))
        c.fit(X[b], y[b]); members.append(c)
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X, y)

    # Repair candidate (diagnosis: the pairwise MLP never sees pool-relative information --
    # e.g. "does ANY candidate satisfy the prefix rule" -- which is exactly what a cascade
    # exploits). ``verifier_rel`` adds, per candidate, its score minus the pool maximum and an
    # is-pool-argmax flag for prefix / edit / cosine / description-Jaccard, and trains on the
    # top-4 negatives under BOTH the embedding and the edit rule plus 4 random negatives.
    def rel_feats(F):
        cols = [1, 3, 4, 5]
        mx = F[:, cols].max(axis=0)
        rel = F[:, cols] - mx
        ismax = (F[:, cols] >= mx - 1e-9).astype("float32")
        return np.concatenate([F, rel, ismax], axis=1)
    Xr, yr = [], []
    for i in tr:
        gi = gold[i]
        R = rel_feats(Fall[i])
        Xr.append(R[gi]); yr.append(1)
        negs = [k for k in np.argsort(-Fall[i][:, 4])[:6] if k != gi][:4]
        negs += [k for k in np.argsort(-Fall[i][:, 3])[:6] if k != gi and k not in negs][:4]
        negs += rng.sample([k for k in range(len(kg)) if k != gi], 4)
        for j in negs:
            Xr.append(R[j]); yr.append(0)
    Xr, yr = np.array(Xr, "float32"), np.array(yr)
    members_rel = []
    for m in range(5):
        b = rn.integers(0, len(Xr), len(Xr))
        c = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=[(32, 16), (24,), (40, 20), (16, 16), (48, 24)][m], alpha=1e-3, max_iter=600, random_state=m))
        c.fit(Xr[b], yr[b]); members_rel.append(c)

    # tune cascades on TRAIN only
    grid = {"prefix": [0.5, 1.0], "fuzzy": [0.5, 0.6, 0.7, 0.8, 0.9], "embed": [0.0, 0.5, 0.6, 0.7, 0.8]}
    def train_acc(order, thr):
        return np.mean([_cascade(Fall[i], order, thr) == gold[i] for i in tr])
    fixed_order = ("prefix", "fuzzy", "embed")
    best_fixed = max(({"prefix": p, "fuzzy": f, "embed": e} for p in grid["prefix"] for f in grid["fuzzy"] for e in grid["embed"]),
                     key=lambda t: train_acc(fixed_order, t))
    best_any = max(((order, {"prefix": p, "fuzzy": f, "embed": e}) for order in itertools.permutations(RULES)
                    for p in grid["prefix"] for f in grid["fuzzy"] for e in grid["embed"]),
                   key=lambda t: train_acc(*t))

    methods = {
        "prefix": lambda F: int(np.argmax(RULES["prefix"](F))),
        "fuzzy": lambda F: int(np.argmax(RULES["fuzzy"](F))),
        "embed": lambda F: int(np.argmax(RULES["embed"](F))),
        "cascade_prefix_fuzzy_embed": lambda F: _cascade(F, fixed_order, best_fixed),
        "cascade_best_order": lambda F: _cascade(F, best_any[0], best_any[1]),
        "fusion_max": lambda F: _fusion(F, "max"),
        "fusion_sum": lambda F: _fusion(F, "sum"),
        "logistic": lambda F: int(np.argmax(logit.predict_proba(F)[:, 1])),
        "verifier": lambda F: int(np.argmax(np.mean([m.predict_proba(F)[:, 1] for m in members], axis=0))),
        "verifier_rel": lambda F: int(np.argmax(np.mean([m.predict_proba(rel_feats(F))[:, 1] for m in members_rel], axis=0))),
    }
    ok = {name: np.array([int(fn(Fall[i]) == gold[i]) for i in te]) for name, fn in methods.items()}
    dtypes = np.array([getattr(recs[i], "dtype", "hier") for i in te])
    rb = np.random.default_rng(seed)
    n = len(te)
    def boot(a, b, B=10000):
        ix = rb.integers(0, n, (B, n))
        d = a[ix].mean(axis=1) - b[ix].mean(axis=1)
        return {"delta": round(float(a.mean() - b.mean()), 4), "p_one_sided": round(float((d <= 0).mean()), 4),
                "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    return {"seed": seed, "n_test": n, "tuned": {"fixed_order_thresholds": best_fixed,
                                                  "best_order": list(best_any[0]), "best_order_thresholds": best_any[1]},
            "recall@1": {k: round(float(v.mean()), 4) for k, v in ok.items()},
            "recall@1_by_dtype": {t: {k: round(float(ok[k][dtypes == t].mean()), 4) for k in ok} for t in sorted(set(dtypes))},
            "verifier_vs": {k: boot(ok["verifier"], ok[k]) for k in ok if k != "verifier"}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args(argv)
    per = [run_seed(int(s)) for s in a.seeds.split(",")]
    names = list(per[0]["recall@1"])
    agg = {k: {"mean": round(float(np.mean([p["recall@1"][k] for p in per])), 4),
               "sd": round(float(np.std([p["recall@1"][k] for p in per], ddof=1)), 4) if len(per) > 1 else 0.0}
           for k in names}
    sig = {k: {"mean_delta": round(float(np.mean([p["verifier_vs"][k]["delta"] for p in per])), 4),
               "n_seeds_sig": int(sum(p["verifier_vs"][k]["p_one_sided"] < 0.05 for p in per))}
           for k in names if k != "verifier"}
    out = {"per_seed": per, "aggregate_recall@1": agg, "verifier_vs_aggregate": sig}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("E5 recall@1 (mean±sd over seeds):", json.dumps(agg))
    print("E5 verifier vs:", json.dumps(sig))
    return out


if __name__ == "__main__":
    main()
