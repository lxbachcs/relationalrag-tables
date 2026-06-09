"""Real table+KG join experiment on the ICD-10-CM + Wikidata benchmark.

Match each table record (specific billable code + description) to its Wikidata KG node
(category code + disease label) among all KG nodes. Baselines: exact code match, fuzzy
edit, a hand-crafted prefix/hierarchy rule, and embedding similarity; vs a learned
verifier over all features. Reports recall@1 overall and on the drift subset, with
paired-bootstrap significance. Run: ``env/bin/python -m relational_rag_real.run_icd_kg``.
"""

from __future__ import annotations

import json
import random
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from . import icd_kg as K
from .embed import embed

RESULTS = Path(__file__).resolve().parents[1] / "artifacts" / "icd_kg" / "results.json"


def _char3(s: str) -> set:
    s = s.lower().replace(" ", "")
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def _jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def features(rec, rvec, node, nvec) -> list[float]:
    tc, K_ = rec.table_code, node["norm"]
    exact = 1.0 if tc == K_ else 0.0
    prefix = 1.0 if (tc.startswith(K_) or K_.startswith(tc)) else 0.0
    cp = _common_prefix(tc, K_) / max(len(tc), len(K_), 1)
    edit = SequenceMatcher(None, tc, K_).ratio()
    cos = float(rvec @ nvec)
    djac = _jac(_char3(rec.table_desc), _char3(node["label"]))
    return [exact, prefix, cp, edit, cos, djac]


def _auroc(scores, labels) -> float:
    s = np.asarray(scores, float); y = np.asarray(labels)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return round(float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)), 4)


def main(seed: int = 0, test_frac: float = 0.4, k: int = 5, corrupt_frac: float = 0.0,
         desc_noise: float = 0.0) -> dict:
    recs, kg = K.build(seed=seed, corrupt_frac=corrupt_frac)
    if desc_noise > 0:  # degrade description/entity-linking quality (header noise)
        nrng = random.Random(seed + 99)
        def _noise(t):
            w = [x for x in t.split() if nrng.random() > desc_noise]
            return " ".join(w) or t
        for r in recs:
            r.table_desc = _noise(r.table_desc)
        for nd in kg:
            nd["label"] = _noise(nd["label"])
    node_text = [f"{n['label']} {n['code']}" for n in kg]
    node_emb = embed(node_text)
    qid_idx = {n["qid"]: i for i, n in enumerate(kg)}
    rec_emb = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)

    rng = random.Random(seed)
    idx = list(range(len(recs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    train_i, test_i = idx[:cut], idx[cut:]

    # learned verifier: train on (record, node) pairs with sampled hard negatives
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = [], []
    for i in train_i:
        r = recs[i]; gi = qid_idx[r.gold_qid]
        X.append(features(r, rec_emb[i], kg[gi], node_emb[gi])); y.append(1)
        sims = node_emb @ rec_emb[i]
        hard = [j for j in np.argsort(-sims)[:8] if j != gi][:4]
        rand = rng.sample(range(len(kg)), 4)
        for j in dict.fromkeys(hard + rand):
            X.append(features(r, rec_emb[i], kg[j], node_emb[j])); y.append(0)
    X, y = np.array(X, "float32"), np.array(y)
    members = []
    rngnp = np.random.default_rng(seed)
    archs = [(32, 16), (24,), (40, 20), (16, 16), (48, 24)]
    for m in range(k):
        b = rngnp.integers(0, len(X), len(X))
        clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=archs[m % len(archs)],
                            alpha=1e-3, max_iter=600, random_state=seed + m))
        clf.fit(X[b], y[b]); members.append(clf)

    def proba(F):
        return np.mean([m.predict_proba(F)[:, 1] for m in members], axis=0)

    # score every test record against all KG nodes with each method; recall@1
    methods = ["exact", "fuzzy", "prefix", "embed", "verifier"]
    hit = {m: [0, 0] for m in methods}     # [all_correct, drift_correct]
    per_q = {m: [] for m in methods}        # correctness for bootstrap (test order)
    drift_flags = []
    for i in test_i:
        r = recs[i]; gi = qid_idx[r.gold_qid]
        F = np.array([features(r, rec_emb[i], kg[j], node_emb[j]) for j in range(len(kg))], "float32")
        scores = {"exact": F[:, 0], "fuzzy": F[:, 3], "prefix": F[:, 1] + 0.01 * F[:, 2],
                  "embed": F[:, 4], "verifier": proba(F)}
        drift_flags.append(r.drift)
        for m in methods:
            top = int(np.argmax(scores[m]))
            ok = int(top == gi)
            hit[m][0] += ok; hit[m][1] += ok and r.drift
            per_q[m].append(ok)

    nt = len(test_i); ndrift = sum(drift_flags)
    out = {"r1": {m: {"all": round(hit[m][0] / nt, 4),
                      "drift": round(hit[m][1] / ndrift, 4) if ndrift else None} for m in methods}}

    # paired bootstrap: verifier vs exact, and verifier vs prefix (overall recall@1)
    ve = np.array(per_q["verifier"]); ex = np.array(per_q["exact"]); pf = np.array(per_q["prefix"])
    rb = np.random.default_rng(0)
    def boot(a, b):
        d = [(a[ix].mean() - b[ix].mean()) for ix in (rb.integers(0, nt, nt) for _ in range(10000))]
        d = np.array(d); return round(float(a.mean() - b.mean()), 4), round(float((d <= 0).mean()), 4)
    out["sig"] = {"verifier_vs_exact": boot(ve, ex), "verifier_vs_prefix": boot(ve, pf)}
    out["meta"] = {"kg_nodes": len(kg), "records": len(recs), "test": nt, "test_drift": ndrift,
                   "auroc_verifier_test": None}
    out["meta"]["corrupt_frac"] = corrupt_frac
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS.parent / ("results_corrupt.json" if corrupt_frac else "results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[corrupt_frac={corrupt_frac}] recall@1 (test, all / drift):")
    for m in methods:
        print("  %-9s %.3f / %s" % (m, out["r1"][m]["all"], out["r1"][m]["drift"]))
    print("sig verifier vs exact:", out["sig"]["verifier_vs_exact"], "| vs prefix:", out["sig"]["verifier_vs_prefix"])
    return out


def feature_ablation(seed: int = 0, test_frac: float = 0.4, k: int = 5, corrupt_frac: float = 0.0) -> dict:
    """Train the verifier ensemble on feature subsets of the 6-dim ICD feature vector
    [exact, prefix, common-prefix, edit, embed-cos, desc-jaccard] to show the learned
    gain is a fusion, not one obvious rule. Reports recall@1 (all / drift)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    recs, kg = K.build(seed=seed, corrupt_frac=corrupt_frac)
    node_emb = embed([f"{n['label']} {n['code']}" for n in kg])
    qid_idx = {n["qid"]: i for i, n in enumerate(kg)}
    rec_emb = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)
    rng = random.Random(seed); idx = list(range(len(recs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac)); train_i, test_i = idx[:cut], idx[cut:]

    Xtr, ytr = [], []
    for i in train_i:
        r = recs[i]; gi = qid_idx[r.gold_qid]
        Xtr.append(features(r, rec_emb[i], kg[gi], node_emb[gi])); ytr.append(1)
        hard = [j for j in np.argsort(-(node_emb @ rec_emb[i]))[:8] if j != gi][:4]
        for j in dict.fromkeys(hard + rng.sample(range(len(kg)), 4)):
            Xtr.append(features(r, rec_emb[i], kg[j], node_emb[j])); ytr.append(0)
    Xtr, ytr = np.array(Xtr, "float32"), np.array(ytr)
    Ftest = {i: np.array([features(recs[i], rec_emb[i], kg[j], node_emb[j]) for j in range(len(kg))], "float32") for i in test_i}
    archs = [(32, 16), (24,), (40, 20), (16, 16), (48, 24)]
    subsets = {"prefix_only": [1, 2], "edit_only": [3], "embed_only": [4], "code_struct": [0, 1, 2, 3], "full": [0, 1, 2, 3, 4, 5]}
    out = {}
    for name, cols in subsets.items():
        members = []; rn = np.random.default_rng(seed)
        for m in range(k):
            b = rn.integers(0, len(Xtr), len(Xtr))
            c = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=archs[m], alpha=1e-3, max_iter=600, random_state=seed + m))
            c.fit(Xtr[b][:, cols], ytr[b]); members.append(c)
        hit = [0, 0]; nd = 0
        for i in test_i:
            gi = qid_idx[recs[i].gold_qid]
            s = np.mean([mm.predict_proba(Ftest[i][:, cols])[:, 1] for mm in members], axis=0)
            ok = int(int(np.argmax(s)) == gi); hit[0] += ok; hit[1] += ok and recs[i].drift; nd += recs[i].drift
        out[name] = {"all": round(hit[0] / len(test_i), 4), "drift": round(hit[1] / nd, 4) if nd else None}
    print("ICD_FEATURE_ABLATION:", json.dumps(out))
    return out


def failure_analysis(seed: int = 0, test_frac: float = 0.4, k: int = 5) -> dict:
    """Characterize verifier errors on the ICD held-out split: recall@1 misses, how
    many are sibling-code confusions vs unrelated, how many commit a wrong join
    (score >= tau_v) vs safely abstain, and the false-negative rate (gold scored low)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    recs, kg = K.build(seed=seed)
    node_emb = embed([f"{n['label']} {n['code']}" for n in kg])
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    rec_emb = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)
    rng = random.Random(seed); idx = list(range(len(recs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac)); tr, te = idx[:cut], idx[cut:]
    X, y = [], []
    for i in tr:
        r = recs[i]; gi = qi[r.gold_qid]
        X.append(features(r, rec_emb[i], kg[gi], node_emb[gi])); y.append(1)
        hard = [j for j in np.argsort(-(node_emb @ rec_emb[i]))[:8] if j != gi][:4]
        for j in dict.fromkeys(hard + rng.sample(range(len(kg)), 4)):
            X.append(features(r, rec_emb[i], kg[j], node_emb[j])); y.append(0)
    X, y = np.array(X, "float32"), np.array(y)
    members = []; rn = np.random.default_rng(seed)
    archs = [(32, 16), (24,), (40, 20), (16, 16), (48, 24)]
    for m in range(k):
        b = rn.integers(0, len(X), len(X))
        c = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=archs[m], alpha=1e-3, max_iter=600, random_state=seed + m))
        c.fit(X[b], y[b]); members.append(c)
    tau = 0.5; miss = sibling = commit_wrong = abstain_on_miss = fn = 0
    for i in te:
        gi = qi[recs[i].gold_qid]
        F = np.array([features(recs[i], rec_emb[i], kg[j], node_emb[j]) for j in range(len(kg))], "float32")
        s = np.mean([mm.predict_proba(F)[:, 1] for mm in members], axis=0)
        if s[gi] < tau:
            fn += 1
        top = int(np.argmax(s))
        if top != gi:
            miss += 1
            if kg[top]["norm"][:3] == kg[gi]["norm"][:3]:
                sibling += 1
            if s[top] >= tau:
                commit_wrong += 1
            else:
                abstain_on_miss += 1
    n = len(te)
    out = {"n_test": n, "recall@1_misses": miss, "miss_rate": round(miss / n, 4),
           "miss_sibling_code": sibling, "miss_committed_wrong": commit_wrong,
           "miss_safe_abstain": abstain_on_miss, "gold_rejected_fn": fn, "fn_rate": round(fn / n, 4)}
    print("ICD_FAILURE_ANALYSIS:", json.dumps(out))
    return out


def _struct_features(rec, node) -> list[float]:
    """Lightweight ICD-hierarchy structural context (O(1) string ops, no extra I/O):
    shared 3-char category (the 1-hop parent in the ICD tree), shared chapter letter,
    code-length specificity agreement, and a sibling indicator (same parent, different
    code). These are the type/hierarchy priors a pairwise feature map ignores."""
    tc, K_ = rec.table_code, node["norm"]
    cat = 1.0 if tc[:3] == K_[:3] else 0.0
    chap = 1.0 if tc[:1] == K_[:1] else 0.0
    depth = 1.0 - abs(len(tc) - len(K_)) / max(len(tc), len(K_), 1)
    sib = 1.0 if (tc[:3] == K_[:3] and tc != K_) else 0.0
    return [cat, chap, depth, sib]


def _ece_simple(ps, ls, n_bins: int = 10) -> float:
    ps = np.asarray(ps, float); ls = np.asarray(ls)
    conf = np.maximum(ps, 1 - ps); acc = ((ps >= 0.5).astype(int) == ls).astype(float)
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        if m.sum():
            e += abs(acc[m].mean() - conf[m].mean()) * m.sum() / len(ps)
    return round(float(e), 4)


def structural_context(seed: int = 0, test_frac: float = 0.4, k: int = 5) -> dict:
    """Reviewer Q1: does lightweight structural context beyond the pairwise features help?
    We augment the verifier with ICD-hierarchy context and compare base vs base+structural
    on a controlled split (identical training pairs and ensemble seeds, only the feature
    map differs), reporting recall@1 (all / drift), held-out pair AUROC, ECE, the
    sibling-code confusion count the failure analysis flagged, and per-pair feature cost."""
    import time
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    recs, kg = K.build(seed=seed)
    node_emb = embed([f"{n['label']} {n['code']}" for n in kg])
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    rec_emb = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)
    rng = random.Random(seed); idx = list(range(len(recs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac)); tr, te = idx[:cut], idx[cut:]
    archs = [(32, 16), (24,), (40, 20), (16, 16), (48, 24)]

    def feat(i, j, struct):
        base = features(recs[i], rec_emb[i], kg[j], node_emb[j])
        return base + _struct_features(recs[i], kg[j]) if struct else base

    # fixed (record, node, label) training pairs so the two variants differ only in features
    rngn = random.Random(seed); train_pairs = []
    for i in tr:
        gi = qi[recs[i].gold_qid]; train_pairs.append((i, gi, 1))
        hard = [j for j in np.argsort(-(node_emb @ rec_emb[i]))[:8] if j != gi][:4]
        for j in dict.fromkeys(hard + rngn.sample(range(len(kg)), 4)):
            train_pairs.append((i, j, 0))

    def run(struct):
        Xtr = np.array([feat(i, j, struct) for i, j, _ in train_pairs], "float32")
        ytr = np.array([l for _, _, l in train_pairs])
        members = []; rn = np.random.default_rng(seed)
        for m in range(k):
            b = rn.integers(0, len(Xtr), len(Xtr))
            c = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=archs[m], alpha=1e-3, max_iter=600, random_state=seed + m))
            c.fit(Xtr[b], ytr[b]); members.append(c)
        hit_all = hit_drift = ndrift = sib = 0; ps, ls = [], []
        for i in te:
            gi = qi[recs[i].gold_qid]
            F = np.array([feat(i, j, struct) for j in range(len(kg))], "float32")
            s = np.mean([mm.predict_proba(F)[:, 1] for mm in members], axis=0)
            top = int(np.argmax(s)); ok = int(top == gi); hit_all += ok
            if recs[i].drift:
                ndrift += 1; hit_drift += ok
            if top != gi and kg[top]["norm"][:3] == kg[gi]["norm"][:3]:
                sib += 1
            ps.append(float(s[gi])); ls.append(1)
            for j in [j for j in np.argsort(-s)[:6] if j != gi][:1]:
                ps.append(float(s[j])); ls.append(0)
        nt = len(te)
        t0 = time.perf_counter()
        for _ in range(2000):
            feat(te[0], 0, struct)
        us = (time.perf_counter() - t0) / 2000 * 1e6
        return {"recall@1_all": round(hit_all / nt, 4),
                "recall@1_drift": round(hit_drift / ndrift, 4) if ndrift else None,
                "auroc": _auroc(ps, ls), "ece": _ece_simple(ps, ls),
                "sibling_confusions": sib, "feat_us_per_pair": round(us, 2)}

    base, struct = run(False), run(True)
    out = {"base": base, "base_plus_struct": struct,
           "delta_recall_all": round(struct["recall@1_all"] - base["recall@1_all"], 4),
           "delta_recall_drift": round((struct["recall@1_drift"] or 0) - (base["recall@1_drift"] or 0), 4),
           "delta_ece": round(struct["ece"] - base["ece"], 4),
           "delta_sibling": struct["sibling_confusions"] - base["sibling_confusions"]}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    (RESULTS.parent / "results_struct.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("ICD_STRUCTURAL_CONTEXT:", json.dumps(out))
    return out


def _jaro(s1, s2) -> float:
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2:
        return 0.0
    md = max(max(l1, l2) // 2 - 1, 0)
    m1 = [False] * l1; m2 = [False] * l2; matches = 0
    for i in range(l1):
        for j in range(max(0, i - md), min(i + md + 1, l2)):
            if not m2[j] and s1[i] == s2[j]:
                m1[i] = m2[j] = True; matches += 1; break
    if not matches:
        return 0.0
    t = k = 0
    for i in range(l1):
        if m1[i]:
            while not m2[k]:
                k += 1
            if s1[i] != s2[k]:
                t += 1
            k += 1
    t /= 2
    return (matches / l1 + matches / l2 + (matches - t) / matches) / 3


def _jaro_winkler(s1, s2, p: float = 0.1) -> float:
    j = _jaro(s1, s2); pre = 0
    for a, b in zip(s1, s2):
        if a != b or pre == 4:
            break
        pre += 1
    return j + pre * p * (1 - j)


def _canon(code: str) -> str:
    """Domain-aware ICD code canonicalization: strip dots/spaces/hyphens, uppercase."""
    return code.upper().replace(".", "").replace(" ", "").replace("-", "")


def _train_ensemble(recs, kg, rec_emb, node_emb, qi, train_i, seed, k):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = random.Random(seed)
    X, y = [], []
    for i in train_i:
        r = recs[i]; gi = qi[r.gold_qid]
        X.append(features(r, rec_emb[i], kg[gi], node_emb[gi])); y.append(1)
        hard = [j for j in np.argsort(-(node_emb @ rec_emb[i]))[:8] if j != gi][:4]
        for j in dict.fromkeys(hard + rng.sample(range(len(kg)), 4)):
            X.append(features(r, rec_emb[i], kg[j], node_emb[j])); y.append(0)
    X, y = np.array(X, "float32"), np.array(y)
    members = []; rn = np.random.default_rng(seed)
    archs = [(32, 16), (24,), (40, 20), (16, 16), (48, 24)]
    for m in range(k):
        b = rn.integers(0, len(X), len(X))
        c = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=archs[m % len(archs)], alpha=1e-3, max_iter=600, random_state=seed + m))
        c.fit(X[b], y[b]); members.append(c)
    return members


def baselines(seed: int = 0, test_frac: float = 0.4, k: int = 5) -> dict:
    """Consolidated baseline comparison on the ICD-10 hierarchy split, adding the
    practical normalization and fuzzy matchers reviewers requested: domain-aware code
    canonicalization (strip dots and spacing), Jaro-Winkler string similarity, and a
    character-TF-IDF cosine over code and description, alongside exact, edit-fuzzy, the
    prefix-hierarchy rule, embedding, and the learned verifier. We report recall@1 on a
    held-out split (all / drift). Canonicalization and Jaro-Winkler test whether `exact`
    and `fuzzy` are merely underpowered by formatting, isolating genuine hierarchy drift."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    recs, kg = K.build(seed=seed)
    node_text = [f"{n['label']} {n['code']}" for n in kg]
    rec_text = [f"{r.table_desc} {r.table_code}" for r in recs]
    node_emb = embed(node_text)
    rec_emb = embed(rec_text, is_query=True)
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    node_canon = [_canon(n["norm"]) for n in kg]
    tfv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(node_text + rec_text)
    node_tf = tfv.transform(node_text)
    rng = random.Random(seed); idx = list(range(len(recs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac)); tr, te = idx[:cut], idx[cut:]
    members = _train_ensemble(recs, kg, rec_emb, node_emb, qi, tr, seed, k)

    def proba(F):
        return np.mean([m.predict_proba(F)[:, 1] for m in members], axis=0)

    methods = ["exact", "canon", "fuzzy", "jaro_winkler", "tfidf", "prefix", "embed", "verifier"]
    hit = {m: [0, 0] for m in methods}; ndrift = 0
    for i in te:
        r = recs[i]; gi = qi[r.gold_qid]; tcn = _canon(r.table_code)
        F = np.array([features(r, rec_emb[i], kg[j], node_emb[j]) for j in range(len(kg))], "float32")
        tf_sim = (tfv.transform([rec_text[i]]) @ node_tf.T).toarray().ravel()
        jw = np.array([_jaro_winkler(tcn, node_canon[j]) for j in range(len(kg))])
        canon = np.array([1.0 if tcn == node_canon[j] else 0.0 for j in range(len(kg))])
        scores = {"exact": F[:, 0], "canon": canon, "fuzzy": F[:, 3], "jaro_winkler": jw,
                  "tfidf": tf_sim, "prefix": F[:, 1] + 0.01 * F[:, 2], "embed": F[:, 4], "verifier": proba(F)}
        if r.drift:
            ndrift += 1
        for m in methods:
            ok = int(int(np.argmax(scores[m])) == gi)
            hit[m][0] += ok; hit[m][1] += ok and r.drift
    nt = len(te)
    out = {m: {"all": round(hit[m][0] / nt, 4), "drift": round(hit[m][1] / ndrift, 4) if ndrift else None} for m in methods}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    (RESULTS.parent / "results_baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("ICD_BASELINES:", json.dumps(out))
    return out


def threshold_sensitivity(seed: int = 0, test_frac: float = 0.4, k: int = 5,
                          taus=(0.3, 0.4, 0.5, 0.6, 0.7)) -> dict:
    """Reviewer Q8: planner-threshold stability on real ICD data. We score the gold pair
    and sampled hard negatives for each held-out record and sweep the acceptance
    threshold tau_v, reporting precision, recall, and F1 of the accept decision."""
    recs, kg = K.build(seed=seed)
    node_emb = embed([f"{n['label']} {n['code']}" for n in kg])
    rec_emb = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    rng = random.Random(seed); idx = list(range(len(recs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac)); tr, te = idx[:cut], idx[cut:]
    members = _train_ensemble(recs, kg, rec_emb, node_emb, qi, tr, seed, k)
    def proba(F):
        return np.mean([m.predict_proba(F)[:, 1] for m in members], axis=0)
    ps, ys = [], []
    for i in te:
        gi = qi[recs[i].gold_qid]
        sims = node_emb @ rec_emb[i]
        cand = [gi] + [j for j in np.argsort(-sims)[:6] if j != gi][:3]
        F = np.array([features(recs[i], rec_emb[i], kg[j], node_emb[j]) for j in cand], "float32")
        for s, j in zip(proba(F), cand):
            ps.append(float(s)); ys.append(int(j == gi))
    ps, ys = np.array(ps), np.array(ys)
    out = {}
    for t in taus:
        pred = (ps >= t).astype(int)
        tp = int(((pred == 1) & (ys == 1)).sum()); fp = int(((pred == 1) & (ys == 0)).sum())
        fn = int(((pred == 0) & (ys == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"{t}"] = {"prec": round(prec, 3), "rec": round(rec, 3), "f1": round(f1, 3)}
    print("ICD_THRESHOLD:", json.dumps(out))
    return out


def leakage(seed: int = 0, test_frac: float = 0.4) -> dict:
    """Reviewer Q3: quantify train/test overlap in the ICD split. We report how many
    test records share their 3-character ICD category, and their exact gold node, with a
    training record, since the verifier could in principle exploit category-level signal."""
    recs, _ = K.build(seed=seed)
    rng = random.Random(seed); idx = list(range(len(recs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac)); tr, te = idx[:cut], idx[cut:]
    tr_cat = {recs[i].table_code[:3] for i in tr}
    tr_node = {recs[i].gold_qid for i in tr}
    tr_code = {recs[i].table_code for i in tr}
    cat = sum(recs[i].table_code[:3] in tr_cat for i in te)
    node = sum(recs[i].gold_qid in tr_node for i in te)
    code = sum(recs[i].table_code in tr_code for i in te)
    n = len(te)
    out = {"n_test": n, "share_category_with_train": round(cat / n, 3),
           "share_gold_node_with_train": round(node / n, 3),
           "share_exact_code_with_train": round(code / n, 3)}
    print("ICD_LEAKAGE:", json.dumps(out))
    return out


def _regime_feats(s: str):
    toks = s.split(); chars = s.replace(" ", "")
    nc = max(len(chars), 1)
    return [len(toks),
            sum(c.isdigit() for c in chars) / nc,
            sum(c.isalpha() for c in chars) / nc,
            len(chars) / max(len(toks), 1),
            1.0 if any(c in ".-/" for c in s) else 0.0,
            1.0 if len(toks) == 1 else 0.0]


def retrieval_noise(seed: int = 0, test_frac: float = 0.4, k: int = 5, tau: float = 0.5,
                    ps=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """Reviewer concern: retrieval noise on the real ICD corpus, not only synthetic. We drop
    the gold KG node from the candidate pool with probability p and report, for the
    no-verifier planner (commit the top-scored node) and the verifier (commit only if the top
    score is at least tau, else abstain), the wrong-answer and abstention rates. As p grows the
    verifier converts unrecoverable joins into abstentions while the no-verifier planner returns
    silent wrong answers."""
    recs, kg = K.build(seed=seed)
    node_emb = embed([f"{n['label']} {n['code']}" for n in kg])
    rec_emb = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    rng = random.Random(seed); idx = list(range(len(recs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac)); tr, te = idx[:cut], idx[cut:]
    members = _train_ensemble(recs, kg, rec_emb, node_emb, qi, tr, seed, k)
    def proba(F):
        return np.mean([m.predict_proba(F)[:, 1] for m in members], axis=0)
    out = {}
    for p in ps:
        rngp = random.Random(1000 + int(p * 100)); nvw = vw = va = 0
        for i in te:
            gi = qi[recs[i].gold_qid]
            drop = rngp.random() < p
            cand = [j for j in range(len(kg)) if not (drop and j == gi)]
            s = proba(np.array([features(recs[i], rec_emb[i], kg[j], node_emb[j]) for j in cand], "float32"))
            top = int(np.argmax(s)); top_j = cand[top]
            if top_j != gi:
                nvw += 1
            if s[top] >= tau:
                vw += int(top_j != gi)
            else:
                va += 1
        n = len(te)
        out[f"{p:.2f}"] = {"nv_wrong": round(nvw / n, 3), "ver_wrong": round(vw / n, 3), "ver_abstain": round(va / n, 3)}
    print("ICD_RETRIEVAL_NOISE:", json.dumps(out))
    return out


def ann_scalability(sizes=(10000, 100000, 1000000), d: int = 384, nq: int = 64, seed: int = 0) -> dict:
    """Reviewer concern: empirically demonstrate enterprise-scale retrieval rather than only
    asserting it. We build a FAISS exact (flat inner-product) and an ANN (IVF) index over N
    random unit d-vectors and measure mean per-query latency. Exact search grows linearly with
    N while ANN stays roughly flat; because the verifier scores only the few retrieved
    candidates, its per-join cost is constant in N and retrieval, not verification, is the
    scaling bottleneck."""
    import faiss
    import time
    rng = np.random.default_rng(seed); out = []
    for N in sizes:
        X = rng.standard_normal((N, d)).astype("float32"); faiss.normalize_L2(X)
        Q = X[rng.integers(0, N, nq)].copy()
        flat = faiss.IndexFlatIP(d); flat.add(X)
        nlist = max(1, int(4 * (N ** 0.5)))
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(X); ivf.add(X); ivf.nprobe = 16
        t0 = time.perf_counter(); flat.search(Q, 10); flat_ms = (time.perf_counter() - t0) / nq * 1000
        t0 = time.perf_counter(); ivf.search(Q, 10); ann_ms = (time.perf_counter() - t0) / nq * 1000
        row = {"N": N, "exact_ms": round(flat_ms, 3), "ann_ms": round(ann_ms, 3)}
        out.append(row); print("ANN_SCALE", json.dumps(row))
        del X, flat, ivf
    return {"dim": d, "by_size": out}


def regime_detector(seed: int = 0) -> dict:
    """Reviewer Q7: a lightweight per-instance regime detector. The cascade failed because
    it routed on the verifier's miscalibrated confidence (an output); a detector that routes
    on cheap input features should not. We classify identifier queries (route to the learned
    verifier) versus natural-language spans (route to lexical matching) using six surface
    features (token count, digit/alpha fraction, mean token length, code punctuation,
    single-token), with the ICD codes as identifiers and their descriptions as NL spans.
    We report five-fold cross-validated accuracy of a logistic detector."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    recs, _ = K.build(seed=seed)
    ids = [r.table_code for r in recs]
    spans = [r.table_desc for r in recs]
    X = np.array([_regime_feats(s) for s in ids + spans], "float64")
    y = np.array([1] * len(ids) + [0] * len(spans))
    acc = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5).mean()
    # majority-class baseline is 0.5 (balanced)
    out = {"n": int(len(y)), "detector_cv_accuracy": round(float(acc), 4), "majority_baseline": 0.5}
    print("ICD_REGIME_DETECTOR:", json.dumps(out))
    return out


if __name__ == "__main__":
    main()
