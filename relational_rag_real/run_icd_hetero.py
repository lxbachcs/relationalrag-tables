"""Heterogeneous-drift table+KG join: one realistic benchmark mixing hierarchy drift,
OCR corruption, AND cross-vocabulary (ICD-9->ICD-10) records over a single ICD-10 disease
KG. No fixed rule is robust across all three (prefix dies on corruption + cross-vocab;
embedding is weak on hierarchy where descriptions differ). Tests whether a learned
verifier that adapts per-instance BEATS every fixed string/rule/semantic baseline ---
the honest case for a learned matcher.

Run: ``env/bin/python -m relational_rag_real.run_icd_hetero``.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import icd_kg as K
from .embed import embed
from .run_icd_kg import features

RESULTS = Path(__file__).resolve().parents[1] / "artifacts" / "icd_kg" / "results_hetero.json"


def _icd9_map():
    d = json.loads((K.DATA / "wikidata_icd9_icd10.json").read_text())["results"]["bindings"]
    return {b["item"]["value"].split("/")[-1]: re.sub(r"[^A-Za-z0-9]", "", b["icd9"]["value"]).upper() for b in d}


def build(seed: int = 0):
    kg = K.load_kg(); icd = K.load_icd10(); desc_of = dict(icd)
    by_cat = defaultdict(list)
    for c, _ in icd:
        by_cat[c[:3]].append(c)
    icd9 = _icd9_map()
    rng = random.Random(seed)
    recs = []
    for n in kg:
        Kc, q = n["norm"], n["qid"]
        children = [c for c in by_cat.get(Kc[:3], []) if c != Kc and c.startswith(Kc) and len(c) > len(Kc)]
        opts = (["hier", "corrupt"] if children else []) + (["xvocab"] if q in icd9 else [])
        if not opts:
            continue
        t = rng.choice(opts)
        if t == "xvocab":
            recs.append(K.Record(icd9[q], n["label"], q, True, corrupted=False))
            recs[-1].dtype = "xvocab"
        else:
            ch = rng.choice(children)
            corrupt = t == "corrupt"
            code = K.corrupt_code(ch, rng, rng.choice([1, 2])) if corrupt else ch
            r = K.Record(code, desc_of.get(ch, n["label"]), q, True, corrupted=corrupt)
            r.dtype = t
            recs.append(r)
    return recs, kg


def main(seed: int = 0, test_frac: float = 0.4) -> dict:
    recs, kg = build(seed=seed)
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    ne = embed([f"{n['label']} {n['code']}" for n in kg])
    re_ = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)

    rng = random.Random(seed); idx = list(range(len(recs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac)); tr, te = idx[:cut], idx[cut:]

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for i in tr:
        r = recs[i]; gi = qi[r.gold_qid]
        X.append(features(r, re_[i], kg[gi], ne[gi])); y.append(1)
        for j in [k for k in np.argsort(-(ne @ re_[i]))[:8] if k != gi][:4] + rng.sample(range(len(kg)), 4):
            X.append(features(r, re_[i], kg[j], ne[j])); y.append(0)
    X, y = np.array(X, "float32"), np.array(y)
    members = []; rn = np.random.default_rng(seed)
    for m in range(5):
        b = rn.integers(0, len(X), len(X))
        c = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=[(32, 16), (24,), (40, 20), (16, 16), (48, 24)][m], alpha=1e-3, max_iter=600, random_state=m))
        c.fit(X[b], y[b]); members.append(c)

    scorers = {"exact": lambda F: F[:, 0], "fuzzy": lambda F: F[:, 3],
               "prefix": lambda F: F[:, 1] + 0.01 * F[:, 2], "embed": lambda F: F[:, 4],
               "verifier": lambda F: np.mean([m.predict_proba(F)[:, 1] for m in members], axis=0)}
    ok = {k: [] for k in scorers}; dtypes = []
    for i in te:
        r = recs[i]; gi = qi[r.gold_qid]; dtypes.append(getattr(r, "dtype", "hier"))
        F = np.array([features(r, re_[i], kg[j], ne[j]) for j in range(len(kg))], "float32")
        for k, sc in scorers.items():
            ok[k].append(int(int(np.argmax(sc(F))) == gi))
    ok = {k: np.array(v) for k, v in ok.items()}; dtypes = np.array(dtypes)

    rb = np.random.default_rng(0); n = len(te)
    def boot(a, b):
        d = np.array([(a[ix].mean() - b[ix].mean()) for ix in (rb.integers(0, n, n) for _ in range(10000))])
        return round(float(a.mean() - b.mean()), 4), round(float((d <= 0).mean()), 4)

    out = {"n_records": len(recs), "test": n,
           "dtype_counts": {t: int((dtypes == t).sum()) for t in set(dtypes)},
           "recall@1": {k: round(float(ok[k].mean()), 4) for k in scorers},
           "recall@1_by_dtype": {t: {k: round(float(ok[k][dtypes == t].mean()), 4) for k in scorers}
                                  for t in set(dtypes)},
           "sig_verifier_vs": {"embed": boot(ok["verifier"], ok["embed"]),
                               "prefix": boot(ok["verifier"], ok["prefix"]),
                               "fuzzy": boot(ok["verifier"], ok["fuzzy"])}}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("HETERO recall@1:", json.dumps(out["recall@1"]))
    print("HETERO sig verifier vs:", json.dumps(out["sig_verifier_vs"]))
    print("HETERO by dtype:", json.dumps(out["recall@1_by_dtype"]))
    return out


if __name__ == "__main__":
    main()
