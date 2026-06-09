"""Cross-vocabulary table+KG join: ICD-9 (table) -> ICD-10 (KG). The two code systems
share no structure (0%% prefix overlap), so exact/fuzzy/prefix code matching is
inapplicable (~0) and a learned/semantic matcher is NECESSARY. Completes the regime
map: rules suffice on hierarchical drift (run_icd_kg); rules FAIL across vocabularies.

Run: ``env/bin/python -m relational_rag_real.run_icd_xvocab``.
"""

from __future__ import annotations

import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .embed import embed

DATA = Path(__file__).resolve().parents[1] / "data" / "icd_kg" / "wikidata_icd9_icd10.json"
RESULTS = Path(__file__).resolve().parents[1] / "artifacts" / "icd_kg" / "results_xvocab.json"


def _norm(c) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(c)).upper()


def load():
    d = json.loads(DATA.read_text())["results"]["bindings"]
    seen, rows = set(), []
    for b in d:
        q = b["item"]["value"]
        if q in seen or not b.get("itemLabel"):
            continue
        seen.add(q)
        rows.append({"label": b["itemLabel"]["value"], "icd9": _norm(b["icd9"]["value"]),
                     "icd10": _norm(b["icd10"]["value"]), "icd10_raw": b["icd10"]["value"]})
    return rows


def _feat(tc, tvec, nc, nvec):
    return [1.0 if tc == nc else 0.0,
            1.0 if (tc.startswith(nc) or nc.startswith(tc)) else 0.0,
            SequenceMatcher(None, tc, nc).ratio(),
            float(tvec @ nvec)]


def main(seed: int = 0, test_frac: float = 0.4, desc_drop: float = 0.3) -> dict:
    rows = load()
    rng = random.Random(seed)
    # KG node = ICD-10 + canonical label; table record = ICD-9 + degraded free-text label.
    node_text = [f"{r['label']} (ICD-10 {r['icd10_raw']})" for r in rows]
    def drop(t):
        w = [x for x in t.split() if rng.random() > desc_drop]
        return " ".join(w) or t
    rec_text = [f"ICD-9 {r['icd9']}: {drop(r['label'])}" for r in rows]
    ne = embed(node_text); re_ = embed(rec_text, is_query=True)
    codes_n = [r["icd10"] for r in rows]; codes_t = [r["icd9"] for r in rows]

    idx = list(range(len(rows))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac)); tr, te = idx[:cut], idx[cut:]

    # learned verifier over [exact, prefix, edit, embed-cos]
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for i in tr:
        X.append(_feat(codes_t[i], re_[i], codes_n[i], ne[i])); y.append(1)
        for j in [k for k in np.argsort(-(ne @ re_[i]))[:6] if k != i][:3] + rng.sample(range(len(rows)), 3):
            X.append(_feat(codes_t[i], re_[i], codes_n[j], ne[j])); y.append(0)
    X, y = np.array(X, "float32"), np.array(y)
    members = []
    rn = np.random.default_rng(seed)
    for m in range(5):
        b = rn.integers(0, len(X), len(X))
        c = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=[(32, 16), (24,), (40, 20), (16, 16), (48, 24)][m], alpha=1e-3, max_iter=600, random_state=m))
        c.fit(X[b], y[b]); members.append(c)

    def r1(scorer):
        h = 0
        for i in te:
            F = np.array([_feat(codes_t[i], re_[i], codes_n[j], ne[j]) for j in range(len(rows))], "float32")
            h += int(int(np.argmax(scorer(F))) == i)
        return round(h / len(te), 4)

    out = {"n_diseases": len(rows), "test": len(te),
           "recall@1": {"exact": r1(lambda F: F[:, 0]), "fuzzy": r1(lambda F: F[:, 2]),
                        "prefix": r1(lambda F: F[:, 1] + 0.01 * F[:, 2]),
                        "embed": r1(lambda F: F[:, 3]),
                        "verifier": r1(lambda F: np.mean([m.predict_proba(F)[:, 1] for m in members], axis=0))}}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("XVOCAB recall@1:", json.dumps(out["recall@1"]), f"(n_test={len(te)})")
    return out


if __name__ == "__main__":
    main()
