"""Real HybridQA study: dense retrieval, a learned evidence verifier, and a deep
ensemble for selective prediction — all on the downloaded corpus.

The verifier scores a (question, unit) pair using ONLY question+unit features (no
answer leakage): embedding cosine, lexical overlap/coverage, unit kind, and length.
Distant supervision (answer string occurs in the unit) provides the labels. We
report: retrieval recall@k (dense vs lexical), verifier reranking (recall@1 / MRR),
held-out evidence-classification AUROC vs lexical/dense baselines, and a
deep-ensemble selective-prediction curve. Subset sizes are configurable; numbers
scale to the full splits.

Run: ``env/bin/python -m relational_rag_real.run_hybridqa``
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from . import hybridqa as H
from .embed import embed

RESULTS = Path(__file__).resolve().parents[1] / "artifacts" / "hybridqa" / "results.json"

_units_cache: dict[str, list] = {}
_uemb_cache: dict[str, np.ndarray] = {}
_qvec_cache: dict[str, np.ndarray] = {}


def _units(table_id: str):
    if table_id not in _units_cache:
        _units_cache[table_id] = H.table_units(table_id)
    return _units_cache[table_id]


def _ensure_units_embedded(table_ids) -> None:
    """Batch-embed all units of not-yet-cached tables in one flat pass."""
    todo, flat, spans = [], [], []
    for tid in dict.fromkeys(table_ids):
        if tid in _uemb_cache:
            continue
        try:
            units = _units(tid)
        except FileNotFoundError:
            _uemb_cache[tid] = np.zeros((0, 384), "float32")
            continue
        spans.append((tid, len(units)))
        flat.extend(u.text for u in units)
    if not flat:
        return
    E = embed(flat, batch_size=512)
    i = 0
    for tid, n in spans:
        _uemb_cache[tid] = E[i : i + n]
        i += n


def _ensure_questions_embedded(questions) -> None:
    todo = [q for q in questions if q["question_id"] not in _qvec_cache]
    if not todo:
        return
    V = embed([q["question"] for q in todo], is_query=True, batch_size=512)
    for q, v in zip(todo, V):
        _qvec_cache[q["question_id"]] = v


def _qtokens(text: str) -> set[str]:
    return set(H.normalize(text).split())


def _lexical(qtok: set[str], text: str) -> tuple[float, float]:
    t = set(H.normalize(text).split())
    if not t or not qtok:
        return 0.0, 0.0
    inter = len(qtok & t)
    return inter / len(qtok), inter / len(qtok | t)  # coverage, jaccard


def _features(qvec, qtok, unit, uvec) -> list[float]:
    cos = float(qvec @ uvec)
    cov, jac = _lexical(qtok, unit.text)
    return [cos, cov, jac, 1.0 if unit.kind == "cell" else 0.0,
            1.0 if unit.kind == "passage" else 0.0, min(len(unit.text), 2000) / 2000.0]


def _records(questions):
    """Yield (q, units, gold_uids, qvec, qtok, unit_emb) from caches."""
    _ensure_units_embedded([q["table_id"] for q in questions])
    _ensure_questions_embedded(questions)
    for q in questions:
        try:
            units = _units(q["table_id"])
        except FileNotFoundError:
            continue
        gold = {u.uid for u in H.answer_bearing_units(q, units)}
        if not gold or not units:
            continue
        yield q, units, gold, _qvec_cache[q["question_id"]], _qtokens(q["question"]), _uemb_cache[q["table_id"]]


def _rrf(dense, lex, c: float = 60.0) -> np.ndarray:
    """Reciprocal-rank fusion of two score vectors into a hybrid score."""
    dr = np.empty(len(dense)); dr[np.argsort(-dense)] = np.arange(len(dense))
    lr = np.empty(len(lex)); lr[np.argsort(-lex)] = np.arange(len(lex))
    return 1.0 / (c + dr + 1) + 1.0 / (c + lr + 1)


def eval_retrieval(questions, ks=(1, 5, 10, 20)) -> dict:
    names = ("dense", "lex", "hybrid")
    hits = {f"{nm}@{k}": 0 for nm in names for k in ks}
    n = 0
    for q, units, gold, qvec, qtok, E in _records(questions):
        n += 1
        dense = E @ qvec
        lex = np.array([_lexical(qtok, u.text)[0] for u in units])
        scored = {"dense": dense, "lex": lex, "hybrid": _rrf(dense, lex)}
        for nm, scores in scored.items():
            ranked = [units[i].uid for i in np.argsort(-scores)]
            for k in ks:
                if gold & set(ranked[:k]):
                    hits[f"{nm}@{k}"] += 1
    return {"n": n, **{key: round(v / n, 4) for key, v in hits.items()}}


def build_pairs(questions, *, neg_per_q: int = 8, seed: int = 0):
    rng = random.Random(seed)
    X, y = [], []
    for q, units, gold, qvec, qtok, E in _records(questions):
        gold_idx = [i for i, u in enumerate(units) if u.uid in gold]
        neg_idx = [i for i, u in enumerate(units) if u.uid not in gold]
        neg_idx.sort(key=lambda i: -_lexical(qtok, units[i].text)[0])  # hard negs first
        chosen = neg_idx[: neg_per_q // 2]
        if neg_idx:
            chosen += rng.sample(neg_idx, min(len(neg_idx), neg_per_q))
        for i in gold_idx:
            X.append(_features(qvec, qtok, units[i], E[i])); y.append(1)
        for i in dict.fromkeys(chosen):
            X.append(_features(qvec, qtok, units[i], E[i])); y.append(0)
    return np.array(X, "float32"), np.array(y)


def train_ensemble(X, y, k: int = 5, seed: int = 0):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    archs = [(48, 24), (32,), (64, 32), (24, 24), (40, 20)]
    members = []
    rng = np.random.default_rng(seed)
    for m in range(k):
        idx = rng.integers(0, len(X), len(X))  # bootstrap
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=archs[m % len(archs)], alpha=1e-3,
                                          max_iter=800, random_state=seed + m))
        clf.fit(X[idx], y[idx])
        members.append(clf)
    return members


def _proba(members, X) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


def _auroc(scores, labels) -> float:
    s = np.asarray(scores, float); y = np.asarray(labels)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return round(float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)), 4)


def eval_verifier(members, dev_q, *, seed: int = 0) -> dict:
    Xte, yte = build_pairs(dev_q, seed=seed + 1)
    vscore = _proba(members, Xte)
    out = {
        "n_pairs": int(len(yte)), "positives": int(yte.sum()),
        "auroc_verifier": _auroc(vscore, yte),
        "auroc_dense_cos": _auroc(Xte[:, 0], yte),
        "auroc_lexical": _auroc(Xte[:, 1], yte),
    }
    pred = (vscore >= 0.5).astype(int)
    correct = (pred == yte)
    order = np.argsort(-np.abs(vscore - 0.5))  # most confident first
    out["selective"] = [
        {"coverage": c, "accuracy": round(float(correct[order[: max(1, int(len(order) * c))]].mean()), 4)}
        for c in (1.0, 0.9, 0.8, 0.7, 0.5)
    ]
    return out


def eval_reranking(members, dev_q, *, topk: int = 20) -> dict:
    """Rerank a HYBRID candidate pool (dense∪lexical top-k) with each scorer; report
    recall@1 and MRR. The verifier reranks the same pool the baselines see."""
    r1 = {"lexical": 0, "dense": 0, "hybrid": 0, "verifier_rerank": 0,
          "cascade_cheap": 0, "cascade_ver": 0}
    mrr = {"hybrid": 0.0, "verifier_rerank": 0.0}
    n = 0
    lex_thr = 0.1   # "lexical found something" gate (cheap-primary)
    ver_thr = 0.5   # verifier-confidence gate (verifier-primary)
    for q, units, gold, qvec, qtok, E in _records(dev_q):
        n += 1
        dense = E @ qvec
        lex = np.array([_lexical(qtok, u.text)[0] for u in units])
        hybrid = _rrf(dense, lex)
        pool = sorted(set(np.argsort(-dense)[:topk]).union(np.argsort(-lex)[:topk]))
        def top1(scores_over_pool):
            return units[pool[int(np.argmax(scores_over_pool))]].uid in gold
        if top1([dense[i] for i in pool]): r1["dense"] += 1
        if top1([lex[i] for i in pool]): r1["lexical"] += 1
        if top1([hybrid[i] for i in pool]): r1["hybrid"] += 1
        feats = np.array([_features(qvec, qtok, units[i], E[i]) for i in pool], "float32")
        vs = _proba(members, feats)
        ver_order = [pool[i] for i in np.argsort(-vs)]
        if units[ver_order[0]].uid in gold: r1["verifier_rerank"] += 1
        lex_pool = [lex[i] for i in pool]
        lex_pick = pool[int(np.argmax(lex_pool))]
        # cheap-primary: trust lexical when it fires; else escalate to verifier
        casc_cheap = lex_pick if max(lex_pool) >= lex_thr else ver_order[0]
        if units[casc_cheap].uid in gold: r1["cascade_cheap"] += 1
        # verifier-primary: trust verifier; fall back to lexical when unconfident
        casc_ver = ver_order[0] if vs.max() >= ver_thr else lex_pick
        if units[casc_ver].uid in gold: r1["cascade_ver"] += 1
        for scorer, order in (("hybrid", [pool[i] for i in np.argsort(-hybrid[pool])]),
                              ("verifier_rerank", ver_order)):
            for rank, i in enumerate(order, 1):
                if units[i].uid in gold:
                    mrr[scorer] += 1 / rank; break
    return {"n": n,
            "recall@1": {kk: round(v / n, 4) for kk, v in r1.items()},
            "mrr": {kk: round(v / n, 4) for kk, v in mrr.items()}}


def main(n_train: int = 2000, n_dev: int = 600, k: int = 5, seed: int = 0) -> dict:
    t0 = time.time()
    rng = random.Random(seed)
    train = rng.sample(H.questions("train"), n_train)
    dev = rng.sample(H.questions("dev"), n_dev)

    retrieval = eval_retrieval(dev)
    print("retrieval:", json.dumps(retrieval))

    Xtr, ytr = build_pairs(train, seed=seed)
    print(f"train pairs: {len(ytr)} ({int(ytr.sum())} pos)")
    members = train_ensemble(Xtr, ytr, k=k, seed=seed)

    verifier = eval_verifier(members, dev, seed=seed)
    print("verifier AUROC:", {kk: verifier[kk] for kk in ("auroc_verifier", "auroc_dense_cos", "auroc_lexical")})
    rerank = eval_reranking(members, dev)
    print("rerank recall@1:", rerank["recall@1"])

    results = {"config": {"n_train": n_train, "n_dev": n_dev, "k": k, "seed": seed,
                          "embed_model": "BAAI/bge-small-en-v1.5", "runtime_s": round(time.time() - t0, 1)},
               "retrieval": retrieval, "verifier": verifier, "reranking": rerank}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", RESULTS, "in", results["config"]["runtime_s"], "s")
    return results


if __name__ == "__main__":
    main()
