"""Reproducible analyses behind the paper's secondary numbers: paired-bootstrap
significance, multi-seed variance, the ICD-10 clean->corrupted transfer, and the
description-noise sensitivity. Each function returns and prints the cited values.

Run all:  ``env/bin/python -m relational_rag_real.analysis``
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parents[1] / "artifacts"


def paired_bootstrap_ex(a: str = "baseline_7b_1534", b: str = "prune_values_7b_1534", n_boot: int = 10000):
    """Paired bootstrap of EX(b)-EX(a) on the saved per-question records (overall + drift)."""
    pa = json.loads((ART / "bird_sql" / f"{a}.json").read_text())["per_question"]
    pb = json.loads((ART / "bird_sql" / f"{b}.json").read_text())["per_question"]
    ao = np.array([x["ok"] for x in pa]); bo = np.array([x["ok"] for x in pb])
    dr = np.array([x["drift"] for x in pa])
    rng = np.random.default_rng(0)
    out = {}
    for name, mask in (("overall", np.ones(len(ao), bool)), ("drift", dr)):
        x, y = bo[mask], ao[mask]; m = len(x)
        d = np.array([(x[i].mean() - y[i].mean()) for i in (rng.integers(0, m, m) for _ in range(n_boot))])
        out[name] = {"n": int(m), "delta": round(float(x.mean() - y.mean()), 4),
                     "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                     "p_le0": round(float((d <= 0).mean()), 4)}
    print("paired-bootstrap EX (prune_values vs baseline):", json.dumps(out))
    return out


def multiseed(task: str = "bird", seeds=range(5)):
    """Verifier recall@1 mean+/-std across seeds. task in {bird, icd}."""
    vd = []
    if task == "bird":
        from . import run_bird
        for s in seeds:
            vd.append(run_bird.main(seed=s)["recall@1_test"]["verifier"]["drift"])
    else:
        from . import run_icd_kg
        for s in seeds:
            vd.append(run_icd_kg.main(seed=s)["r1"]["verifier"]["drift"])
    res = {"task": task, "verifier_drift_mean": round(float(np.mean(vd)), 3), "std": round(float(np.std(vd)), 3)}
    print("multiseed:", json.dumps(res))
    return res


def icd_transfer(seed: int = 0, corrupt_frac: float = 0.5):
    """Train the verifier on CLEAN drift only; evaluate zero-shot on corrupted records."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from . import icd_kg as K
    from .embed import embed
    from .run_icd_kg import features

    recs, kg = K.build(seed=seed, corrupt_frac=corrupt_frac)
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    ne = embed([f"{n['label']} {n['code']}" for n in kg])
    re_ = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)
    archs = [(32, 16), (24,), (40, 20), (16, 16), (48, 24)]

    def train(ix):
        rng = random.Random(seed); X, y = [], []
        for i in ix:
            r = recs[i]; gi = qi[r.gold_qid]
            X.append(features(r, re_[i], kg[gi], ne[gi])); y.append(1)
            hard = [j for j in np.argsort(-(ne @ re_[i]))[:8] if j != gi][:4]
            for j in dict.fromkeys(hard + rng.sample(range(len(kg)), 4)):
                X.append(features(r, re_[i], kg[j], ne[j])); y.append(0)
        X, y = np.array(X, "float32"), np.array(y); M = []; rn = np.random.default_rng(seed)
        for m in range(5):
            bi = rn.integers(0, len(X), len(X))
            c = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=archs[m], alpha=1e-3, max_iter=600, random_state=m))
            c.fit(X[bi], y[bi]); M.append(c)
        return M

    def r1(M, ix):
        h = 0
        for i in ix:
            r = recs[i]; gi = qi[r.gold_qid]
            F = np.array([features(r, re_[i], kg[j], ne[j]) for j in range(len(kg))], "float32")
            h += int(np.argmax(np.mean([m.predict_proba(F)[:, 1] for m in M], axis=0)) == gi)
        return round(h / len(ix), 3)

    idx = list(range(len(recs))); random.Random(seed).shuffle(idx); cut = int(len(idx) * 0.6)
    corr_test = [i for i in idx[cut:] if recs[i].corrupted]
    clean_tr = [i for i in idx[:cut] if not recs[i].corrupted]
    res = {"corrupted_test": len(corr_test),
           "zero_shot_clean_to_corrupt": r1(train(clean_tr), corr_test),
           "in_domain": r1(train(idx[:cut]), corr_test)}
    print("icd_transfer:", json.dumps(res))
    return res


def icd_noise(seed: int = 0, desc_noise: float = 0.5):
    from . import run_icd_kg
    a = run_icd_kg.main(seed=seed, desc_noise=0.0); b = run_icd_kg.main(seed=seed, desc_noise=desc_noise)
    res = {m: {"clean": a["r1"][m]["drift"], "noisy_desc": b["r1"][m]["drift"]} for m in ("prefix", "embed", "verifier")}
    print("icd_noise (drift recall@1):", json.dumps(res))
    return res


def hetero(seeds=range(5)):
    """Heterogeneous-drift join (Table tab:hetero): per-method recall@1 mean+/-std across
    seeds, and how often the learned verifier significantly (p<0.05) beats embedding / the
    prefix rule. Reproduces the table's numbers and the '3/5' and 'all seeds' claims."""
    from . import run_icd_hetero
    methods = ["exact", "fuzzy", "prefix", "embed", "verifier"]
    vals = {m: [] for m in methods}; sig_embed = sig_prefix = 0
    seeds = list(seeds)
    for s in seeds:
        o = run_icd_hetero.main(seed=s)
        for m in methods:
            vals[m].append(o["recall@1"][m])
        sig_embed += int(o["sig_verifier_vs"]["embed"][1] < 0.05)
        sig_prefix += int(o["sig_verifier_vs"]["prefix"][1] < 0.05)
    res = {"n_seeds": len(seeds),
           "recall@1": {m: {"mean": round(float(np.mean(vals[m])), 3),
                            "std": round(float(np.std(vals[m])), 3)} for m in methods},
           "verifier_sig_vs_embed": f"{sig_embed}/{len(seeds)}",
           "verifier_sig_vs_prefix": f"{sig_prefix}/{len(seeds)}"}
    print("hetero (recall@1 mean+/-std):", json.dumps(res))
    return res


def main():
    paired_bootstrap_ex()
    multiseed("bird"); multiseed("icd")
    icd_transfer(); icd_noise(); hetero()


if __name__ == "__main__":
    main()
