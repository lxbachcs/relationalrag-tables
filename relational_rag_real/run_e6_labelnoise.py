"""Rebuttal E6: label-protocol audit and label-noise sensitivity.

Reviewer 3(c) asks who determines the labels, how consistent that determination is, and how
label errors propagate into the gate. Our labels are deterministic protocols, stated here
and written out for a two-annotator check:

  BIRD  gold value  = the literal of a ``col = 'value'`` condition in the gold SQL, confirmed
                      to exist in that column of the database;
        drifted     = normalize(gold value) is not a substring of normalize(question).
  ICD   gold node   = the Wikidata disease node whose ICD-10 code (P494) is the category of
                      the record's billable code; drift type is assigned by construction
                      (hierarchy / OCR-corrupted / cross-vocabulary).

Sensitivity: we corrupt a fraction rho of TRAINING gold assignments (an annotator picking a
wrong candidate; on ICD half of the corruptions are same-category siblings, the realistic
confusion), retrain the verifier, and score on the CLEAN test split. We report recall@1 and,
at tau_v = 0.5, the false-positive / false-negative rates, i.e. how annotation error moves
the gate's error rates that Theorem 1 conditions on.

Run: ``.venv/bin/python -m relational_rag_real.run_e6_labelnoise``
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

from . import bird as B
from . import run_bird as RB
from .embed import embed
from .run_icd_hetero import build
from .run_icd_kg import features

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "e6"
RHOS = (0.0, 0.05, 0.10, 0.20)


# ------------------------------------------------------------------------------ BIRD
def bird(seed: int = 0, test_frac: float = 0.35, k: int = 5) -> dict:
    inst = B.value_link_instances(B.questions(), max_instances=10000)
    rng = random.Random(seed)
    rng.shuffle(inst)
    rows = RB.featurize(inst)
    cut = int(len(rows) * (1 - test_frac))
    train_rows, test_rows = rows[:cut], rows[cut:]
    Xte, yte = RB.pairs_from_rows(test_rows)
    out = {}
    for rho in RHOS:
        nrng = random.Random(seed + int(rho * 1000))
        noisy = []
        flipped = 0
        for inst_i, per in train_rows:
            inst_c = dict(inst_i)
            if rho > 0 and nrng.random() < rho:
                others = [c for c, _ in per if B.normalize(c) != B.normalize(inst_i["gold"])]
                if others:
                    inst_c["gold"] = nrng.choice(others)
                    flipped += 1
            noisy.append((inst_c, per))
        Xtr, ytr = RB.pairs_from_rows(noisy)
        members = RB.train_ensemble(Xtr, ytr, k=k, seed=seed)
        VER = lambda F: RB._proba(members, F)  # noqa: E731
        s = VER(Xte)
        out[f"rho_{rho:.2f}"] = {
            "train_instances_corrupted": flipped,
            "recall@1_all": RB.recall_at_1(test_rows, VER)[0],
            "recall@1_drift": RB.recall_at_1(test_rows, VER, drift_only=True)[0],
            "auroc": RB._auroc(s, yte),
            "gate_at_0.5": RB._confusion(s, yte),
        }
        print("E6 BIRD", rho, json.dumps({k2: v for k2, v in out[f'rho_{rho:.2f}'].items() if k2 != 'gate_at_0.5'}),
              "fp_rate", out[f"rho_{rho:.2f}"]["gate_at_0.5"]["fp_rate"], "fn_rate", out[f"rho_{rho:.2f}"]["gate_at_0.5"]["fn_rate"], flush=True)
    # annotation sample for the two-annotator check (100 instances, protocol fields blank)
    OUT.mkdir(parents=True, exist_ok=True)
    sample = random.Random(seed + 1).sample(inst, min(100, len(inst)))
    with (OUT / "annotation_sample_bird.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "db_id", "question", "evidence", "table", "column", "protocol_gold", "protocol_drift",
                    "annotator_gold", "annotator_drift", "note"])
        for i, s_ in enumerate(sample):
            w.writerow([i, s_["db_id"], s_["question"], s_["evidence"], s_["table"], s_["column"], s_["gold"], int(s_["drift"]), "", "", ""])
    return {"n_train": len(train_rows), "n_test": len(test_rows), "n_test_drift": sum(1 for i, _ in test_rows if i["drift"]),
            "by_rho": out}


# ------------------------------------------------------------------------------- ICD
def icd(seeds=(0, 1, 2), test_frac: float = 0.4) -> dict:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    per_seed = []
    for seed in seeds:
        recs, kg = build(seed=seed)
        qi = {n["qid"]: i for i, n in enumerate(kg)}
        cat_of = {i: n["norm"][:3] for i, n in enumerate(kg)}
        by_cat = {}
        for i, c in cat_of.items():
            by_cat.setdefault(c, []).append(i)
        ne = embed([f"{n['label']} {n['code']}" for n in kg])
        re_ = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)
        rng = random.Random(seed)
        idx = list(range(len(recs)))
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]
        Fte = {i: np.array([features(recs[i], re_[i], kg[j], ne[j]) for j in range(len(kg))], "float32") for i in te}
        row = {"seed": seed, "n_test": len(te)}
        for rho in RHOS:
            nrng = random.Random(seed * 100 + int(rho * 1000))
            X, y = [], []
            flipped = sib = 0
            for i in tr:
                r = recs[i]
                gi = qi[r.gold_qid]
                if rho > 0 and nrng.random() < rho:
                    sibs = [j for j in by_cat.get(cat_of[gi], []) if j != gi]
                    if sibs and nrng.random() < 0.5:
                        gi = nrng.choice(sibs); sib += 1
                    else:
                        gi = nrng.choice([j for j in range(len(kg)) if j != gi])
                    flipped += 1
                X.append(features(r, re_[i], kg[gi], ne[gi])); y.append(1)
                for j in [k for k in np.argsort(-(ne @ re_[i]))[:8] if k != gi][:4] + nrng.sample(range(len(kg)), 4):
                    X.append(features(r, re_[i], kg[j], ne[j])); y.append(0)
            X, y = np.array(X, "float32"), np.array(y)
            members = []
            rn = np.random.default_rng(seed)
            for m in range(5):
                b = rn.integers(0, len(X), len(X))
                c = make_pipeline(StandardScaler(), MLPClassifier(
                    hidden_layer_sizes=[(32, 16), (24,), (40, 20), (16, 16), (48, 24)][m], alpha=1e-3, max_iter=600, random_state=m))
                c.fit(X[b], y[b]); members.append(c)
            hits = 0
            tp = fp = fn = tn = 0
            for i in te:
                gi = qi[recs[i].gold_qid]
                p = np.mean([m.predict_proba(Fte[i])[:, 1] for m in members], axis=0)
                hits += int(int(np.argmax(p)) == gi)
                acc = p >= 0.5
                tp += int(acc[gi]); fn += int(not acc[gi])
                fp += int(acc.sum() - acc[gi]); tn += int(len(p) - 1 - (acc.sum() - acc[gi]))
            row[f"rho_{rho:.2f}"] = {"train_corrupted": flipped, "of_which_sibling": sib,
                                     "recall@1": round(hits / len(te), 4),
                                     "fp_rate": round(fp / (fp + tn), 4), "fn_rate": round(fn / (fn + tp), 4)}
            print("E6 ICD seed", seed, "rho", rho, json.dumps(row[f"rho_{rho:.2f}"]), flush=True)
        per_seed.append(row)
        if seed == seeds[0]:
            OUT.mkdir(parents=True, exist_ok=True)
            sample = random.Random(seed + 1).sample(recs, min(100, len(recs)))
            with (OUT / "annotation_sample_icd.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["id", "table_code", "table_desc", "protocol_gold_qid", "protocol_gold_code", "protocol_drift_type",
                            "annotator_gold_qid", "annotator_drift_type", "note"])
                for i, r in enumerate(sample):
                    w.writerow([i, r.table_code, r.table_desc, r.gold_qid, kg[qi[r.gold_qid]]["norm"], getattr(r, "dtype", "hier"), "", "", ""])
    agg = {}
    for rho in RHOS:
        key = f"rho_{rho:.2f}"
        agg[key] = {m: {"mean": round(float(np.mean([r[key][m] for r in per_seed])), 4),
                        "sd": round(float(np.std([r[key][m] for r in per_seed], ddof=1)), 4) if len(per_seed) > 1 else 0.0}
                    for m in ("recall@1", "fp_rate", "fn_rate")}
    return {"per_seed": per_seed, "aggregate": agg}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"protocol": {
        "bird_gold": "literal of a col = 'value' condition in the gold SQL, confirmed present in that column",
        "bird_drift": "normalize(gold) not a substring of normalize(question)",
        "icd_gold": "Wikidata node whose P494 ICD-10 code is the category of the record's billable code",
        "icd_drift_type": "assigned by construction: hierarchy / OCR-corrupted / cross-vocabulary (ICD-9 record)",
        "annotation_samples": ["artifacts/e6/annotation_sample_bird.csv", "artifacts/e6/annotation_sample_icd.csv"]},
        "bird": bird(), "icd": icd()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
