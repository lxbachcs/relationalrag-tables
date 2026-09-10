"""Rebuttal E7: end-to-end QA on the REAL ICD-10 + Wikidata table+KG benchmark.

The paper's ICD-10 + Wikidata evaluation is component-level (recall@1 of the join). Here the
same real data is run through the paper's own execution-aware planner, so the complete
framework (retrieve -> gated join -> traverse -> return / abstain) is evaluated end to end
on real identifiers, answering the "component vs. framework" concern.

Table ``records``: one clinical record per Wikidata disease node under heterogeneous drift
(hierarchy child code / OCR-corrupted code / legacy ICD-9 code, as in the paper's
heterogeneous benchmark) plus "ghost" records whose ICD-10 category has no Wikidata node.
KG: Wikidata triples fetched live (medical specialty P1995, anatomical location P927, drug
P2176, subclass-of P279, plus the parents' specialties for a 2-hop chain).

Question families (test records only; the verifier is trained on train records):
  single_table_lookup   description of record R-i
  table_kg_join         specialty treating the condition of record R-i (retrieval-ordered pool)
  bad_join_backtrack    same, with a same-category sibling node forced first in the pool
  multi_hop_traverse    specialty of the parent condition class (subclass_of -> specialty)
  abstain               ghost records: no node exists, the only correct answer is ABSTAIN

Gates compared under the identical planner: exact code match, prefix/hierarchy rule,
embedding cosine, learned MLP ensemble (paper verifier), no-verifier, oracle.

Run: ``CUDA_VISIBLE_DEVICES= .venv/bin/python -m relational_rag_real.run_e7_icd_qa --seeds 0,1,2,3,4``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from relational_rag_tables.index import build_entries
from relational_rag_tables.plan_execute import PlannerConfig, build_triple_index, exact_match, run_question
from relational_rag_tables.schemas import Corpus, Question, Table, Triple
from relational_rag_tables.verify import DeterministicJoinVerifier

from . import icd_kg as K
from .embed import embed
from .run_icd_hetero import build
from .run_icd_kg import features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "e7"
FIX = ROOT / "fixtures" / "icd_qa"
TOP_K = 8
PROPS = {"P1995": "medical_specialty", "P927": "anatomical_location", "P2176": "drug_used_for_treatment", "P279": "subclass_of"}


# ------------------------------------------------------------------------------ KG
def load_relations():
    rel = json.load((K.DATA / "wikidata_icd_relations.json").open())
    par = json.load((K.DATA / "wikidata_icd_parent_relations.json").open())
    triples: dict[tuple[str, str], str] = {}
    labels: dict[str, str] = {}
    for b in rel + par:
        p = b["p"]["value"].split("/")[-1]
        if p not in PROPS:
            continue
        subj = "disease:" + b["item"]["value"].split("/")[-1]
        if "itemLabel" in b:
            labels[subj] = b["itemLabel"]["value"]
        v = b["v"]["value"]
        obj = "disease:" + v.split("/")[-1] if p == "P279" else b.get("vLabel", {}).get("value", v)
        if obj.startswith("Q") and obj[1:].isdigit():
            continue
        key = (subj, PROPS[p])
        if key not in triples or obj < triples[key]:  # deterministic single object per (s, r)
            triples[key] = obj
    return triples, labels


# ----------------------------------------------------------------------- verifiers
class FnVerifier:
    """Adapts a (record, node) scorer to the planner's (lhs_value, rhs_id) interface."""

    def __init__(self, fn, recs_by_code, nodes_by_id):
        self.fn, self.recs, self.nodes = fn, recs_by_code, nodes_by_id

    def score(self, lhs_value: str, rhs_id: str) -> float:
        rec = self.recs.get(lhs_value)
        node = self.nodes.get(rhs_id)
        if rec is None or node is None:
            return 0.0
        return float(self.fn(rec, node))

    def uncertainty(self, lhs_value: str, rhs_id: str) -> float:
        p = min(max(self.score(lhs_value, rhs_id), 1e-12), 1 - 1e-12)
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def train_mlp(recs, tr_idx, rvec, kg, nvec, qi, rng, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for i in tr_idx:
        r = recs[i]
        gi = qi[r.gold_qid]
        X.append(features(r, rvec[i], kg[gi], nvec[gi])); y.append(1)
        for j in [k for k in np.argsort(-(nvec @ rvec[i]))[:8] if k != gi][:4] + rng.sample(range(len(kg)), 4):
            X.append(features(r, rvec[i], kg[j], nvec[j])); y.append(0)
    X, y = np.array(X, "float32"), np.array(y)
    members = []
    rn = np.random.default_rng(seed)
    for m in range(5):
        b = rn.integers(0, len(X), len(X))
        c = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=[(32, 16), (24,), (40, 20), (16, 16), (48, 24)][m], alpha=1e-3, max_iter=600, random_state=m))
        c.fit(X[b], y[b]); members.append(c)
    return members


# --------------------------------------------------------------------------- corpus
def build_corpus(seed: int, test_frac: float = 0.4, ghost_frac: float = 0.12):
    recs, kg = build(seed=seed)
    triples_map, parent_labels = load_relations()
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    node_id = {i: "disease:" + n["qid"] for i, n in enumerate(kg)}
    nvec = embed([f"{n['label']} {n['code']}" for n in kg])
    rvec = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)
    rng = random.Random(seed)
    idx = list(range(len(recs)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]

    # ghost records: billable codes whose category has no KG node
    cats = {n["norm"][:3] for n in kg}
    ghosts_pool = [(c, d) for c, d in K.load_icd10() if c[:3] not in cats and len(c) >= 4]
    rng.shuffle(ghosts_pool)
    n_ghost = int(len(te) * ghost_frac)
    ghosts = ghosts_pool[:n_ghost]
    gvec = embed([f"{d} {c}" for c, d in ghosts], is_query=True) if ghosts else np.zeros((0, nvec.shape[1]), "float32")

    # table rows (codes made unique)
    rows, code_of, recs_by_code, seen = [], {}, {}, set()
    used = []
    for i in idx:
        r = recs[i]
        code = r.table_code
        if code in seen:
            continue
        seen.add(code)
        rid = f"R-{len(rows) + 1:04d}"
        rows.append({"record_id": rid, "icd_code": code, "description": r.table_desc})
        code_of[i] = (rid, code)
        recs_by_code[code] = r
        used.append(i)
    ghost_rows = []
    for gi_, (c, d) in enumerate(ghosts):
        if c in seen:
            continue
        seen.add(c)
        rid = f"R-{len(rows) + 1:04d}"
        rows.append({"record_id": rid, "icd_code": c, "description": d})
        ghost_rows.append((rid, c, gi_))
        recs_by_code[c] = K.Record(c, d, "", True)

    triples = [Triple(s, r, o) for (s, r), o in sorted(triples_map.items())]
    for i, n in enumerate(kg):
        triples.append(Triple(node_id[i], "icd10_code", n["code"]))
    nodes_by_id = {node_id[i]: kg[i] for i in range(len(kg))}
    for subj, lab in parent_labels.items():  # parent nodes that are not ICD nodes still need features
        nodes_by_id.setdefault(subj, {"qid": subj.split(":")[1], "label": lab, "code": "", "norm": ""})
    cat_of = defaultdict(list)
    for i, n in enumerate(kg):
        cat_of[n["norm"][:3]].append(i)

    questions, join_labels = [], {}
    te_set = set(te)
    for i in used:
        if i not in te_set:
            continue
        rid, code = code_of[i]
        r = recs[i]
        gi = qi[r.gold_qid]
        gold = node_id[gi]
        pool = [node_id[j] for j in np.argsort(-(nvec @ rvec[i]))[:TOP_K]]
        join_labels[(code, gold)] = True
        for c in pool:
            join_labels.setdefault((code, c), False)
        base = dict(table="records", lookup_column="record_id", lookup_value=rid, join_column="icd_code",
                    gold_join={"lhs": code, "rhs": gold}, split="test")
        questions.append(Question(id=f"{rid}_single", text=f"What is the description for record {rid}?", answer=r.table_desc,
                                  task="single_table_lookup", table="records", lookup_column="record_id", lookup_value=rid,
                                  return_column="description", hops=0, split="test"))
        spec = triples_map.get((gold, "medical_specialty"))
        if spec:
            questions.append(Question(id=f"{rid}_join", text=f"Which medical specialty treats the condition in record {rid}?",
                                      answer=spec, task="table_kg_join", traverse_relation="medical_specialty",
                                      traverse_chain=["medical_specialty"], candidate_order=pool, hops=1, **base))
            sibs = [j for j in cat_of[kg[gi]["norm"][:3]] if j != gi]
            if sibs:
                d = node_id[rng.choice(sibs)]
                join_labels.setdefault((code, d), False)
                questions.append(Question(id=f"{rid}_backtrack",
                                          text=f"Which medical specialty treats the condition in record {rid} after rejecting the sibling code?",
                                          answer=spec, task="bad_join_backtrack", traverse_relation="medical_specialty",
                                          traverse_chain=["medical_specialty"], candidate_order=[d] + [c for c in pool if c != d],
                                          hops=1, **base))
        parent = triples_map.get((gold, "subclass_of"))
        pspec = triples_map.get((parent, "medical_specialty")) if parent else None
        if pspec:
            questions.append(Question(id=f"{rid}_hop2", text=f"Which specialty treats the parent condition class of record {rid}?",
                                      answer=pspec, task="multi_hop_traverse", traverse_chain=["subclass_of", "medical_specialty"],
                                      candidate_order=pool, hops=2, **base))
        drug = triples_map.get((gold, "drug_used_for_treatment"))
        if drug:
            questions.append(Question(id=f"{rid}_drug", text=f"Which drug is used to treat the condition in record {rid}?",
                                      answer=drug, task="table_kg_join", traverse_relation="drug_used_for_treatment",
                                      traverse_chain=["drug_used_for_treatment"], candidate_order=pool, hops=1, **base))
    for rid, c, gi_ in ghost_rows:
        pool = [node_id[j] for j in np.argsort(-(nvec @ gvec[gi_]))[:TOP_K]]
        for cnd in pool:
            join_labels.setdefault((c, cnd), False)
        questions.append(Question(id=f"{rid}_abstain", text=f"Which medical specialty treats the condition in record {rid}?",
                                  answer="ABSTAIN", task="abstain", table="records", lookup_column="record_id", lookup_value=rid,
                                  join_column="icd_code", traverse_relation="medical_specialty", traverse_chain=["medical_specialty"],
                                  candidate_order=pool, hops=1, split="test", should_abstain=True))

    corpus = Corpus(tables={"records": Table("records", rows)}, triples=triples, questions=questions, join_labels=join_labels,
                    config={"tau_v": 0.5, "tau_u": -1, "top_k": TOP_K, "index_backend": "deterministic-local", "seed": seed})
    # persist the fixture for reproducibility
    fx = FIX / f"seed_{seed}"
    (fx / "tables").mkdir(parents=True, exist_ok=True)
    (fx / "kg").mkdir(parents=True, exist_ok=True)
    with (fx / "tables" / "records.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["record_id", "icd_code", "description"]); w.writeheader(); w.writerows(rows)
    (fx / "kg" / "triples.json").write_text(json.dumps([t.__dict__ for t in triples], indent=0), encoding="utf-8")
    (fx / "questions.json").write_text(json.dumps([q.__dict__ for q in questions], indent=0), encoding="utf-8")
    (fx / "join_labels.json").write_text(json.dumps([{"lhs": a, "rhs": b, "label": l} for (a, b), l in join_labels.items()]), encoding="utf-8")
    (fx / "config.json").write_text(json.dumps(corpus.config, indent=1), encoding="utf-8")

    # verifiers
    rvec_of = {recs[i].table_code: rvec[i] for i in used}
    for rid, c, gi_ in ghost_rows:
        rvec_of[c] = gvec[gi_]
    nvec_of = {node_id[i]: nvec[i] for i in range(len(kg))}
    members = train_mlp(recs, tr, rvec, kg, nvec, qi, rng, seed)

    def _f(rec, node):
        nid = "disease:" + node["qid"]
        if nid not in nvec_of:
            return None
        return features(rec, rvec_of[rec.table_code], node, nvec_of[nid])
    def mlp(rec, node):
        F = _f(rec, node)
        return 0.0 if F is None else float(np.mean([m.predict_proba([F])[0, 1] for m in members]))
    def exact(rec, node):
        return 1.0 if K.norm(rec.table_code) == node["norm"] else 0.0
    def prefix(rec, node):
        a, b = K.norm(rec.table_code), node["norm"]
        return 1.0 if b and (a.startswith(b) or b.startswith(a)) else 0.0
    def cosine(rec, node):
        F = _f(rec, node)
        return 0.0 if F is None else F[4]
    verifiers = {
        "exact_match": (FnVerifier(exact, recs_by_code, nodes_by_id), 0.5),
        "prefix_rule": (FnVerifier(prefix, recs_by_code, nodes_by_id), 0.5),
        "embedding": (FnVerifier(cosine, recs_by_code, nodes_by_id), 0.62),
        "verifier_mlp": (FnVerifier(mlp, recs_by_code, nodes_by_id), 0.5),
        "oracle": (DeterministicJoinVerifier(join_labels), 0.5),
    }
    meta = {"n_records": len(rows), "n_ghost": len(ghost_rows), "n_test_records": len([i for i in used if i in te_set]),
            "n_questions": len(questions), "n_triples": len(triples), "by_task": dict(sorted(
                {t: sum(1 for q in questions if q.task == t) for t in {q.task for q in questions}}.items()))}
    return corpus, verifiers, meta


# ------------------------------------------------------------------------- evaluate
def rc(items):
    items = sorted(items, key=lambda t: t[0])
    n = len(items)
    errs, risk = 0, []
    for k, (_, c) in enumerate(items, 1):
        errs += 1 - c; risk.append(errs / k)
    return {"aurc": round(sum(risk) / n, 4), "accuracy_at": {str(c): round(1 - risk[max(1, int(round(n * c))) - 1], 4) for c in (1.0, 0.9, 0.8, 0.7, 0.5)}}


def run_seed(seed: int) -> dict:
    t0 = time.time()
    corpus, verifiers, meta = build_corpus(seed)
    entries = build_entries(corpus)
    tidx = build_triple_index(corpus)
    task_of = {q.id: q.task for q in corpus.questions}
    out = {"seed": seed, "meta": meta, "gates": {}}
    per_q = {}
    for name, (ver, tau) in list(verifiers.items()) + [("no_verifier", (verifiers["oracle"][0], 0.5))]:
        cfg = PlannerConfig(tau_v=tau, top_k=TOP_K, use_verifier=(name != "no_verifier"))
        ok, by = [], defaultdict(lambda: {"n": 0, "correct": 0, "abstained": 0, "wrong": 0})
        items = []
        for q in corpus.questions:
            r = run_question(corpus, entries, q, ver, cfg, tidx, None)
            c = exact_match(r.prediction, r.gold)
            b = by[task_of[q.id]]
            b["n"] += 1; b["correct"] += int(c)
            if r.status == "abstained":
                b["abstained"] += int(q.answer != "ABSTAIN")
            elif not c:
                b["wrong"] += 1
            ok.append(int(c))
            if r.status == "completed" and hasattr(ver, "uncertainty"):
                u = 0.0
                for e in r.trace:
                    if e.operation.type == "join" and e.accepted:
                        u = ver.uncertainty(str(e.operation.args["lhs_value"]), str(e.operation.args["rhs"]))
                items.append((u, int(c)))
        per_q[name] = np.array(ok)
        n = len(ok)
        out["gates"][name] = {"tau_v": tau, "em": round(sum(ok) / n, 4),
                              "wrong_rate": round(sum(b["wrong"] for b in by.values()) / n, 4),
                              "false_abstain_rate": round(sum(b["abstained"] for b in by.values()) / n, 4),
                              "by_task": {t: {"n": b["n"], "em": round(b["correct"] / b["n"], 4), "wrong": b["wrong"], "false_abstain": b["abstained"]}
                                          for t, b in sorted(by.items())},
                              "system_risk_coverage": rc(items) if items and name == "verifier_mlp" else None}
        print(f"[e7 seed {seed}] {name:13s} EM={out['gates'][name]['em']:.4f} wrong={out['gates'][name]['wrong_rate']:.4f} "
              f"false_abstain={out['gates'][name]['false_abstain_rate']:.4f} by_task={ {t: v['em'] for t, v in out['gates'][name]['by_task'].items()} }", flush=True)
    rb = np.random.default_rng(seed)
    n = len(corpus.questions)
    def boot(a, b, B=10000):
        ix = rb.integers(0, n, (B, n))
        d = a[ix].mean(axis=1) - b[ix].mean(axis=1)
        return {"delta": round(float(a.mean() - b.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)}
    out["mlp_vs"] = {k: boot(per_q["verifier_mlp"], per_q[k]) for k in per_q if k != "verifier_mlp"}
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"seed_{seed}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args(argv)
    per = [run_seed(int(s)) for s in a.seeds.split(",")]
    gates = list(per[0]["gates"])
    agg = {g: {m: {"mean": round(float(np.mean([p["gates"][g][m] for p in per])), 4),
                   "sd": round(float(np.std([p["gates"][g][m] for p in per], ddof=1)), 4) if len(per) > 1 else 0.0}
               for m in ("em", "wrong_rate", "false_abstain_rate")} for g in gates}
    tasks = sorted({t for p in per for g in gates for t in p["gates"][g]["by_task"]})
    agg_task = {g: {t: round(float(np.mean([p["gates"][g]["by_task"][t]["em"] for p in per if t in p["gates"][g]["by_task"]])), 4)
                    for t in tasks} for g in gates}
    sig = {k: {"mean_delta": round(float(np.mean([p["mlp_vs"][k]["delta"] for p in per])), 4),
               "n_seeds_sig": int(sum(p["mlp_vs"][k]["p_two_sided"] < 0.05 for p in per))} for k in per[0]["mlp_vs"]}
    res = {"n_seeds": len(per), "aggregate": agg, "aggregate_by_task": agg_task, "mlp_vs": sig,
           "system_risk_coverage_mlp": [p["gates"]["verifier_mlp"]["system_risk_coverage"] for p in per], "per_seed": per}
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("E7 aggregate:", json.dumps(agg))
    print("E7 by task:", json.dumps(agg_task))
    print("E7 mlp vs:", json.dumps(sig))
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
