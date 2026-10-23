"""Experiment E4: one regime-aware matcher for both the identifier regime (BIRD value linking)
and the lexical regime (HybridQA evidence selection).

The paper's cascades fail because two separately trained matchers produce scores on
incompatible scales. Instead of routing between them we train ONE verifier on the pooled
training pairs of both datasets with a feature space that is computable in both regimes:

  base (8)      embedding cosine, lexical coverage, Jaccard, best-token edit ratio,
                char-3-gram Jaccard, containment, is-code, length
  pool-relative (10)  for five of these: value minus the pool maximum, and an is-pool-argmax flag
  regime (6)    the mean surface features of the candidate pool (token count, digit and
                alphabetic fractions, mean token length, code punctuation, single-token flag)

Models compared (5-member MLP ensembles, 3 seeds): joint-base (base + regime), joint-rel
(all 24), and single-regime models with the same 24 features; baselines are lexical
(coverage on HybridQA, best-token edit ratio on BIRD) and dense cosine. We report recall@1
on BIRD test (all / drift) and on HybridQA dev, plus pair-level ECE of the joint model per
regime (the cross-regime calibration question the paper left open).

Run: ``.venv/bin/python -m relational_rag_real.run_e4_joint --seeds 0,1,2``
"""

from __future__ import annotations

import argparse
import json
import random
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from . import bird as B
from . import hybridqa as H
from . import run_bird as RB
from . import run_hybridqa as RH
from .embed import embed
from .run_icd_kg import _regime_feats

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "e4"
REL_COLS = [0, 1, 2, 3, 4]  # cos, coverage, jaccard, best_tok, cjac


# ------------------------------------------------------------------------ features
def _char3(s):
    s = s.replace(" ", "")
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def _jac(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def base_feats(qn, qtok, qvec, cand, cvec):
    cn = B.normalize(cand)[:300]
    ct = set(cn.split())
    cos = float(qvec @ cvec)
    inter = len(qtok & ct)
    cov = inter / len(qtok) if qtok else 0.0
    jac = inter / len(qtok | ct) if (qtok or ct) else 0.0
    short = cn[:60]
    best_tok = max((SequenceMatcher(None, short, t).ratio() for t in qtok), default=0.0) if len(cn) <= 60 else cov
    cjac = _jac(_char3(cn[:120]), set().union(*[_char3(t) for t in qtok]) if qtok else set())
    contained = 1.0 if cn and cn in qn else 0.0
    is_code = 1.0 if (len(cn) <= 6 and " " not in cn) else 0.0
    return [cos, cov, jac, best_tok, cjac, contained, is_code, min(len(cn), 2000) / 2000.0]


def pool_feats(F, cands):
    """F: (n, 8) base features of one candidate pool -> (n, 24) with pool-relative + regime."""
    F = np.asarray(F, "float32")
    mx = F[:, REL_COLS].max(axis=0)
    rel = F[:, REL_COLS] - mx
    ismax = (F[:, REL_COLS] >= mx - 1e-9).astype("float32")
    reg = np.array([_regime_feats(c) for c in cands], "float32").mean(axis=0, keepdims=True)
    reg = np.repeat(reg, len(F), axis=0)
    return np.concatenate([F, rel, ismax, reg], axis=1)


# ------------------------------------------------------------------------ BIRD pools
def bird_pools(seed, test_frac=0.35):
    inst = B.value_link_instances(B.questions(), max_instances=10000)
    rng = random.Random(seed)
    rng.shuffle(inst)
    pools = []
    q_texts, c_texts = [], set()
    for i in inst:
        qtok = set(B.normalize(i["question"]).split())
        i["_cands"] = RB.cap_candidates(i, qtok, seed=seed)
        q_texts.append(f"{i['question']} {i['evidence']}")
        c_texts.update(i["_cands"])
    qv = embed(q_texts, is_query=True, batch_size=512)
    c_list = sorted(c_texts)
    cv = dict(zip(c_list, embed(c_list, batch_size=512)))
    for k, i in enumerate(inst):
        qn = B.normalize(i["question"] + " " + i["evidence"])
        qtok = set(B.normalize(i["question"]).split())
        F = np.array([base_feats(qn, qtok, qv[k], c, cv[c]) for c in i["_cands"]], "float32")
        gold = np.array([1 if B.normalize(c) == B.normalize(i["gold"]) else 0 for c in i["_cands"]])
        pools.append({"F": F, "cands": i["_cands"], "y": gold, "drift": i["drift"]})
    cut = int(len(pools) * (1 - test_frac))
    return pools[:cut], pools[cut:]


# --------------------------------------------------------------------- HybridQA pools
def hqa_pools(questions, topk=20):
    pools = []
    for q, units, gold, qvec, qtok, E in RH._records(questions):
        dense = E @ qvec
        lex = np.array([RH._lexical(qtok, u.text)[0] for u in units])
        pool = sorted(set(np.argsort(-dense)[:topk]).union(np.argsort(-lex)[:topk]))
        y = np.array([1 if units[i].uid in gold else 0 for i in pool])
        if y.sum() == 0:
            continue
        qn = B.normalize(q["question"])
        F = np.array([base_feats(qn, qtok, qvec, units[i].text, E[i]) for i in pool], "float32")
        pools.append({"F": F, "cands": [units[i].text for i in pool], "y": y, "drift": False,
                      "lex": lex[pool], "dense": dense[pool]})
    return pools


# ------------------------------------------------------------------------- models
def stack(pools, rel):
    X, y = [], []
    for p in pools:
        Fx = pool_feats(p["F"], p["cands"]) if rel else np.concatenate(
            [p["F"], np.repeat(np.array([_regime_feats(c) for c in p["cands"]], "float32").mean(axis=0, keepdims=True), len(p["F"]), axis=0)], axis=1)
        X.append(Fx); y.append(p["y"])
    return np.concatenate(X), np.concatenate(y)


def train(X, y, k=5, seed=0):
    return RB.train_ensemble(X, y, k=k, seed=seed)


def recall1(pools, scorer, drift_only=False):
    hit = n = 0
    for p in pools:
        if drift_only and not p["drift"]:
            continue
        n += 1
        s = scorer(p)
        hit += int(p["y"][int(np.argmax(s))] == 1)
    return round(hit / n, 4) if n else None, n


def ece(pools, scorer, bins=10):
    s, y = [], []
    for p in pools:
        s.extend(scorer(p).tolist()); y.extend(p["y"].tolist())
    s, y = np.asarray(s), np.asarray(y)
    conf = np.maximum(s, 1 - s); pred = (s >= 0.5).astype(int)
    e = 0.0
    for b in range(bins):
        m = (conf >= b / bins) & (conf < (b + 1) / bins) if b < bins - 1 else conf >= b / bins
        if m.any():
            e += abs((pred[m] == y[m]).mean() - conf[m].mean()) * m.mean()
    return round(float(e), 4)


def run_seed(seed, n_htrain=2000, n_hdev=600):
    t0 = time.time()
    b_tr, b_te = bird_pools(seed)
    rng = random.Random(seed)
    htrain = rng.sample(H.questions("train"), n_htrain)
    hdev = rng.sample(H.questions("dev"), n_hdev)
    h_tr, h_te = hqa_pools(htrain), hqa_pools(hdev)
    print(f"[e4 seed {seed}] pools: bird {len(b_tr)}/{len(b_te)}  hqa {len(h_tr)}/{len(h_te)}", flush=True)

    models = {}
    for name, tr, rel in (("joint_base", b_tr + h_tr, False), ("joint_rel", b_tr + h_tr, True),
                          ("bird_only_rel", b_tr, True), ("hqa_only_rel", h_tr, True)):
        X, y = stack(tr, rel)
        models[name] = (train(X, y, seed=seed), rel)

    def mk(name):
        members, rel = models[name]
        def sc(p):
            Fx = pool_feats(p["F"], p["cands"]) if rel else stack([p], False)[0]
            return RB._proba(members, Fx)
        return sc
    scorers = {n: mk(n) for n in models}
    scorers["lexical"] = lambda p: p["lex"] if "lex" in p else p["F"][:, 3]  # coverage on HQA, edit ratio on BIRD
    scorers["dense"] = lambda p: p["F"][:, 0]
    scorers["containment"] = lambda p: p["F"][:, 5] + 1e-3 * p["F"][:, 7]

    res = {"seed": seed, "n": {"bird_test": len(b_te), "bird_drift": sum(p["drift"] for p in b_te), "hqa_dev": len(h_te)}, "recall@1": {}, "ece": {}}
    for n, sc in scorers.items():
        res["recall@1"][n] = {"bird_all": recall1(b_te, sc)[0], "bird_drift": recall1(b_te, sc, True)[0], "hqa": recall1(h_te, sc)[0]}
        if n in models:
            res["ece"][n] = {"bird": ece(b_te, sc), "hqa": ece(h_te, sc)}
    res["runtime_s"] = round(time.time() - t0, 1)
    for n, v in res["recall@1"].items():
        print(f"[e4 seed {seed}] {n:14s} BIRD {v['bird_all']} / drift {v['bird_drift']}  HQA {v['hqa']}  ECE {res['ece'].get(n, '')}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seed_{seed}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    per = [run_seed(int(s)) for s in a.seeds.split(",")]
    names = list(per[0]["recall@1"])
    agg = {n: {m: {"mean": round(float(np.mean([p["recall@1"][n][m] for p in per])), 4),
                   "sd": round(float(np.std([p["recall@1"][n][m] for p in per], ddof=1)), 4) if len(per) > 1 else 0.0}
               for m in ("bird_all", "bird_drift", "hqa")} for n in names}
    agg_ece = {n: {d: round(float(np.mean([p["ece"][n][d] for p in per])), 4) for d in ("bird", "hqa")} for n in per[0]["ece"]}
    out = {"n_seeds": len(per), "aggregate_recall@1": agg, "aggregate_ece": agg_ece, "per_seed": per}
    (OUT / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("E4 aggregate:", json.dumps(agg))
    print("E4 ECE:", json.dumps(agg_ece))
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
