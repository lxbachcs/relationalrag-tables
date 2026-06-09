"""Real BIRD value-linking study: link a question entity to the correct database
value, where exact/substring matching fails on drifted values (question "high
school" -> value 'HS'; "Special Education" -> 'SPECON').

This is the real-data analogue of the synthetic identifier-drift study. We score
each candidate value of the relevant column against the question with: exact match
(substring), fuzzy match (best edit ratio), a learned verifier (MLP over
identifier+embedding features), and a deep ensemble. We report recall@1 (pick the
gold value) overall and on the DRIFT subset, plus held-out AUROC and selective
prediction. Run: ``env/bin/python -m relational_rag_real.run_bird``.
"""

from __future__ import annotations

import json
import random
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from . import bird as B
from .embed import embed

RESULTS = Path(__file__).resolve().parents[1] / "artifacts" / "bird" / "results.json"

_emb: dict[str, np.ndarray] = {}


def _char3(s: str) -> set:
    s = s.replace(" ", "")
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def cap_candidates(inst: dict, qtok: set, cap: int = 80, seed: int = 0) -> list[str]:
    """Keep the gold value + lexically-closest + random candidates (for tractability)."""
    rng = random.Random(seed)
    cands = inst["candidates"]
    gold = inst["gold"]
    others = [c for c in cands if B.normalize(c) != B.normalize(gold)]
    others.sort(key=lambda c: -_jaccard(_char3(B.normalize(c)), set().union(*[_char3(t) for t in qtok]) if qtok else set()))
    keep = others[: cap // 2]
    if len(others) > cap // 2:
        keep += rng.sample(others, min(len(others) - cap // 2, cap // 2))
    keep = list(dict.fromkeys(keep))[:cap]
    pool = [gold] + keep
    rng.shuffle(pool)  # gold must NOT be at a fixed index (argmax ties would leak it)
    return pool


def _vec(text: str) -> np.ndarray:
    if text not in _emb:
        _emb[text] = embed([text])[0]
    return _emb[text]


def _embed_all(texts) -> None:
    todo = [t for t in dict.fromkeys(texts) if t not in _emb]
    if todo:
        V = embed(todo, batch_size=512)
        for t, v in zip(todo, V):
            _emb[t] = v


def features(qn: str, evn: str, qtok: set, qvec, value: str, vvec) -> list[float]:
    vn = B.normalize(value)
    cos = float(qvec @ vvec)
    contained = 1.0 if vn and vn in qn else 0.0
    contained_ev = 1.0 if vn and vn in evn else 0.0
    best_tok = max((SequenceMatcher(None, vn, t).ratio() for t in qtok), default=0.0)
    cjac = _jaccard(_char3(vn), set().union(*[_char3(t) for t in qtok]) if qtok else set())
    vtok = set(vn.split())
    tok_overlap = len(vtok & qtok) / len(vtok) if vtok else 0.0
    is_code = 1.0 if (len(vn) <= 6 and " " not in vn) else 0.0
    return [cos, contained, contained_ev, best_tok, cjac, tok_overlap, is_code, min(len(vn), 40) / 40.0]


def featurize(instances):
    # Embed all questions(+evidence) and candidate values in two batched passes.
    q_text = {id(i): f"{i['question']} {i['evidence']}" for i in instances}
    cand_text = {}
    for i in instances:
        i["_cands"] = cap_candidates(i, set(B.normalize(i["question"]).split()))
        for c in i["_cands"]:
            cand_text[(i["column"], c)] = f"{i['column']}: {c}"
    _embed_all(list(q_text.values()) + list(cand_text.values()))
    rows = []
    for i in instances:
        qn, evn = B.normalize(i["question"]), B.normalize(i["evidence"])
        qtok = set(qn.split())
        qvec = _vec(q_text[id(i)])
        per = []
        for c in i["_cands"]:
            vvec = _vec(cand_text[(i["column"], c)])
            per.append((c, features(qn, evn, qtok, qvec, c, vvec)))
        rows.append((i, per))
    return rows


def train_ensemble(X, y, k: int = 5, seed: int = 0):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    archs = [(48, 24), (32,), (64, 32), (24, 24), (40, 20)]
    members = []
    rng = np.random.default_rng(seed)
    for m in range(k):
        idx = rng.integers(0, len(X), len(X))
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=archs[m % len(archs)], alpha=1e-3,
                                          max_iter=800, random_state=seed + m))
        clf.fit(X[idx], y[idx])
        members.append(clf)
    return members


def _proba(members, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


def _auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return round(float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)), 4)


def _confusion(s, y, tau=0.5):
    """Pair-classification false positives/negatives at threshold tau on held-out pairs."""
    s = np.asarray(s, float); y = np.asarray(y); p = (s >= tau).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fn_rate": round(fn / npos, 4) if npos else None,
            "fp_rate": round(fp / nneg, 4) if nneg else None}


def _ece(s, y, n_bins=10):
    """Expected and maximum calibration error on confidence max(p, 1-p), plus Brier."""
    s = np.asarray(s, float); y = np.asarray(y)
    conf = np.maximum(s, 1.0 - s); pred = (s >= 0.5).astype(int)
    ece = mce = 0.0
    for b in range(n_bins):
        m = (conf >= b / n_bins) & (conf < (b + 1) / n_bins) if b < n_bins - 1 else (conf >= b / n_bins)
        if not m.any():
            continue
        acc = float((pred[m] == y[m]).mean()); c = float(conf[m].mean()); gap = abs(acc - c)
        ece += gap * m.mean(); mce = max(mce, gap)
    brier = float(((s - y) ** 2).mean())
    return round(ece, 4), round(mce, 4), round(brier, 4)


def pairs_from_rows(rows):
    X, y, gold_flag = [], [], []
    for inst, per in rows:
        gn = B.normalize(inst["gold"])
        for c, feat in per:
            X.append(feat); lab = 1 if B.normalize(c) == gn else 0
            y.append(lab); gold_flag.append(lab)
    return np.array(X, "float32"), np.array(y)


def recall_at_1(rows, scorer, *, drift_only=False, threshold=None):
    """recall@1; if ``threshold`` is set, the scorer abstains (miss) when its best
    score is below it — the honest behavior of a string/fuzzy-match *tool* that
    returns nothing rather than guessing among a small pool."""
    hit = n = 0
    for inst, per in rows:
        if drift_only and not inst["drift"]:
            continue
        n += 1
        gn = B.normalize(inst["gold"])
        scores = scorer(np.array([f for _, f in per], "float32"))
        j = int(np.argmax(scores))
        if threshold is not None and scores[j] < threshold:
            continue  # tool finds no match -> miss
        if B.normalize(per[j][0]) == gn:
            hit += 1
    return round(hit / n, 4) if n else 0.0, n


def cascade_recall_at_1(rows, vscorer, *, drift_only=False, exact_col=1, exact_thr=0.5):
    """Exact->learned cascade: trust exact containment when it fires; else escalate
    to the learned verifier. Should never lose to exact and should recover drift."""
    hit = n = 0
    for inst, per in rows:
        if drift_only and not inst["drift"]:
            continue
        n += 1
        F = np.array([f for _, f in per], "float32")
        exact = F[:, exact_col]
        j = int(np.argmax(exact)) if exact.max() >= exact_thr else int(np.argmax(vscorer(F)))
        if B.normalize(per[j][0]) == B.normalize(inst["gold"]):
            hit += 1
    return round(hit / n, 4) if n else 0.0


def main(max_q: int = 1534, k: int = 5, seed: int = 0, test_frac: float = 0.35) -> dict:
    t0 = time.time()
    inst = B.value_link_instances(B.questions(), max_instances=max_q * 3)
    rng = random.Random(seed); rng.shuffle(inst)
    rows = featurize(inst)
    cut = int(len(rows) * (1 - test_frac))
    train_rows, test_rows = rows[:cut], rows[cut:]

    Xtr, ytr = pairs_from_rows(train_rows)
    members = train_ensemble(Xtr, ytr, k=k, seed=seed)

    # scorers over a feature matrix -> per-candidate score
    EXACT = lambda F: F[:, 1]          # containment (exact substring match)
    FUZZY = lambda F: F[:, 3]          # best edit ratio
    EMB = lambda F: F[:, 0]            # embedding cosine alone
    VER = lambda F: _proba(members, F)

    def block(rows):
        out = {}
        for name, sc, thr in (("exact", EXACT, 0.5), ("fuzzy", FUZZY, 0.6), ("embed", EMB, None), ("verifier", VER, None)):
            out[name] = {"all": recall_at_1(rows, sc, threshold=thr)[0],
                         "drift": recall_at_1(rows, sc, drift_only=True, threshold=thr)[0]}
        return out

    def cascade_ver(rows, *, drift_only=False, ver_thr=0.5, exact_col=1):
        """Verifier-primary: trust the verifier; fall back to exact only when the
        verifier's top score is below ver_thr (uncertain)."""
        hit = n = 0
        for inst, per in rows:
            if drift_only and not inst["drift"]:
                continue
            n += 1
            F = np.array([f for _, f in per], "float32")
            vs = VER(F); j = int(np.argmax(vs))
            if vs[j] < ver_thr and F[:, exact_col].max() >= 0.5:
                j = int(np.argmax(F[:, exact_col]))
            if B.normalize(per[j][0]) == B.normalize(inst["gold"]):
                hit += 1
        return round(hit / n, 4) if n else 0.0

    recall = block(test_rows)
    recall["cascade_cheap"] = {"all": cascade_recall_at_1(test_rows, VER),
                               "drift": cascade_recall_at_1(test_rows, VER, drift_only=True)}
    recall["cascade_ver"] = {"all": cascade_ver(test_rows), "drift": cascade_ver(test_rows, drift_only=True)}

    Xte, yte = pairs_from_rows(test_rows)
    vscore = _proba(members, Xte)
    pred = (vscore >= 0.5).astype(int); correct = (pred == yte)
    order = np.argsort(-np.abs(vscore - 0.5))
    selective = [{"coverage": c, "accuracy": round(float(correct[order[: max(1, int(len(order) * c))]].mean()), 4)}
                 for c in (1.0, 0.9, 0.8, 0.7, 0.5)]

    n_drift_test = sum(1 for i, _ in test_rows if i["drift"])
    results = {
        "config": {"instances": len(inst), "train_rows": len(train_rows), "test_rows": len(test_rows),
                   "test_drift_instances": n_drift_test, "k": k, "seed": seed,
                   "embed_model": "BAAI/bge-small-en-v1.5", "runtime_s": round(time.time() - t0, 1)},
        "recall@1_test": recall,
        "auroc_test": {"verifier": _auroc(vscore, yte), "embed_cos": _auroc(Xte[:, 0], yte),
                       "fuzzy": _auroc(Xte[:, 3], yte), "exact": _auroc(Xte[:, 1], yte)},
        "calibration_test": dict(zip(("ece", "mce", "brier"), _ece(vscore, yte))),
        "selective_prediction": selective,
        "failure_test": _confusion(vscore, yte),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("recall@1 (test):", json.dumps(results["recall@1_test"]))
    print("auroc (test):", json.dumps(results["auroc_test"]))
    print(f"wrote {RESULTS} in {results['config']['runtime_s']}s "
          f"({len(test_rows)} test instances, {n_drift_test} drift)")
    return results


if __name__ == "__main__":
    main()
