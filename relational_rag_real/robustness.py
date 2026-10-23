"""Learning-curve and robustness experiments on real data.

* Q4 -- ICD learning curve: recall@1 (all/drift) and ECE as a function of the number
  of labeled training records, to quantify the verifier's data efficiency and guide
  annotation budgets.
* Q6 -- BIRD candidate-pool robustness: inject hard (high-cosine) distractor values
  into each test instance's candidate pool and report recall@1, AUROC, ECE, and the
  abstention behaviour as the pool degrades.
* Q2 -- BIRD end-to-end EX as a function of table-pruning recall: pair each schema
  linker's table-level recall with its committed Execution Accuracy, tracing EX along
  the pruning-recall axis (full schema / embedding / learned / oracle).

Each entry point writes a JSON under ``artifacts/`` consumed by the consistency gate.
Run, e.g.: ``python3 -m relational_rag_real.robustness icd_lc``
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from . import bird as B
from . import icd_kg as K
from . import run_bird as RB
from . import run_icd_kg as RI
from .embed import embed

ART = Path(__file__).resolve().parents[1] / "artifacts"


# --------------------------------------------------------------------------- Q4
def icd_learning_curve(seed: int = 0, test_frac: float = 0.4, k: int = 5,
                       sizes=(25, 50, 100, 150, 250, 354)) -> dict:
    """Data efficiency on ICD: fix the held-out test set, grow the labeled training
    record budget, and report recall@1 (all / drift) and held-out pair ECE/AUROC."""
    recs, kg = K.build(seed=seed)
    node_emb = embed([f"{n['label']} {n['code']}" for n in kg])
    rec_emb = embed([f"{r.table_desc} {r.table_code}" for r in recs], is_query=True)
    qi = {n["qid"]: i for i, n in enumerate(kg)}
    rng = random.Random(seed); idx = list(range(len(recs))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac)); tr_all, te = idx[:cut], idx[cut:]

    def evaluate(members):
        def proba(F):
            return np.mean([m.predict_proba(F)[:, 1] for m in members], axis=0)
        hit_all = hit_drift = ndrift = 0; ps, ls = [], []
        for i in te:
            gi = qi[recs[i].gold_qid]
            F = np.array([RI.features(recs[i], rec_emb[i], kg[j], node_emb[j]) for j in range(len(kg))], "float32")
            s = proba(F); top = int(np.argmax(s)); ok = int(top == gi); hit_all += ok
            if recs[i].drift:
                ndrift += 1; hit_drift += ok
            ps.append(float(s[gi])); ls.append(1)
            for j in [j for j in np.argsort(-s)[:6] if j != gi][:1]:
                ps.append(float(s[j])); ls.append(0)
        return {"recall@1_all": round(hit_all / len(te), 4),
                "recall@1_drift": round(hit_drift / ndrift, 4) if ndrift else None,
                "auroc": RI._auroc(ps, ls), "ece": RI._ece_simple(ps, ls)}

    curve = []
    for sz in sizes:
        sub = tr_all[:min(sz, len(tr_all))]
        members = RI._train_ensemble(recs, kg, rec_emb, node_emb, qi, sub, seed, k)
        row = {"n_train_records": len(sub), **evaluate(members)}
        curve.append(row); print("ICD_LC", json.dumps(row))
    out = {"seed": seed, "n_test": len(te), "n_test_drift": int(sum(recs[i].drift for i in te)),
           "n_train_records_full": len(tr_all), "curve": curve}
    p = ART / "icd_kg" / "results_learning_curve.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8"); print("wrote", p)
    return out


# --------------------------------------------------------------------------- Q6
def bird_distractor_robustness(max_q: int = 1534, k: int = 5, seed: int = 0,
                               test_frac: float = 0.35, ver_tau: float = 0.5,
                               n_distract=(0, 2, 5, 10, 20)) -> dict:
    """Inject hard distractor values *from other columns* into each test instance's
    candidate pool and track value-linking robustness. For each test instance we draw
    the highest-embedding-cosine non-gold values from a cross-instance vocabulary,
    re-embed them in this instance's column context, and recompute the full feature
    vector against this question (so the distractor's features are honest, not stale).
    We report recall@1 (all / drift) and, at acceptance threshold ``ver_tau``, the
    wrong-answer vs abstention split as the pool degrades -- testing whether the
    verifier's calibration converts pool noise into abstentions or silent wrong joins."""
    inst = B.value_link_instances(B.questions(), max_instances=max_q * 3)
    rng = random.Random(seed); rng.shuffle(inst)
    rows = RB.featurize(inst)
    cut = int(len(rows) * (1 - test_frac)); train_rows, test_rows = rows[:cut], rows[cut:]
    Xtr, ytr = RB.pairs_from_rows(train_rows)
    members = RB.train_ensemble(Xtr, ytr, k=k, seed=seed)
    VER = lambda F: RB._proba(members, F)

    # cross-instance distractor vocabulary: distinct (value, bare-embedding) drawn from
    # candidate pools, used to rank by cosine to each target question.
    vocab = {}
    for inst_i, per in test_rows:
        for c, _ in per:
            vocab.setdefault(B.normalize(c), c)
    vocab_vals = list(vocab.values())
    vocab_vecs = embed(vocab_vals, batch_size=512)
    vnorm = [B.normalize(v) for v in vocab_vals]

    def distractor_feats(inst_i, ndist, rs):
        """Top-cosine cross-column distractors for this instance, features recomputed
        in-column against this question."""
        if ndist <= 0:
            return []
        qn, evn = B.normalize(inst_i["question"]), B.normalize(inst_i["evidence"])
        qtok = set(qn.split())
        qvec = RB._vec(f"{inst_i['question']} {inst_i['evidence']}")
        gold_n = B.normalize(inst_i["gold"])
        sims = vocab_vecs @ qvec
        order = np.argsort(-sims)
        # widen then sample so distractors are hard (high-cosine) but not deterministic
        top = [int(j) for j in order[: ndist * 6]]
        rs.shuffle(top)
        out = []
        col = inst_i["column"]
        for j in top:
            if vnorm[j] == gold_n:
                continue
            val = vocab_vals[j]
            vvec = RB._vec(f"{col}: {val}")  # re-embed in THIS column context (cached)
            out.append((val, RB.features(qn, evn, qtok, qvec, val, vvec)))
            if len(out) >= ndist:
                break
        return out

    def recall_with_injection(ndist, drift_only=False, tau=None):
        rs = random.Random(1234 + ndist); hit = n = wrong = abstain = 0
        for inst_i, per in test_rows:
            if drift_only and not inst_i["drift"]:
                continue
            n += 1
            per2 = list(per) + distractor_feats(inst_i, ndist, rs)
            F = np.array([f for _, f in per2], "float32"); s = VER(F)
            j = int(np.argmax(s))
            correct = B.normalize(per2[j][0]) == B.normalize(inst_i["gold"])
            if tau is not None and s[j] < tau:
                abstain += 1
            else:
                hit += int(correct); wrong += int(not correct)
        out = {"recall@1": round(hit / n, 4) if n else 0.0}
        if tau is not None:
            out.update({"wrong": round(wrong / n, 4), "abstain": round(abstain / n, 4)})
        return out

    curve = []
    for nd in n_distract:
        row = {"n_distractors": nd,
               "all": recall_with_injection(nd)["recall@1"],
               "drift": recall_with_injection(nd, drift_only=True)["recall@1"],
               "gated": recall_with_injection(nd, tau=ver_tau)}
        curve.append(row); print("BIRD_DISTRACT", json.dumps(row), flush=True)
    n_drift = sum(1 for i, _ in test_rows if i["drift"])
    out = {"config": {"test_rows": len(test_rows), "test_drift": n_drift, "k": k, "seed": seed,
                      "ver_tau": ver_tau}, "curve": curve}
    p = ART / "bird" / "results_distractor.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8"); print("wrote", p)
    return out


# --------------------------------------------------------------------------- Q2
def bird_ex_vs_pruning_recall(seed: int = 0, n_eval: int = 1534) -> dict:
    """Trace end-to-end EX along the table-pruning-recall axis. For each schema linker
    we compute its table-level recall (fraction of gold-SQL tables retained) on the dev
    questions and pair it with that configuration's committed Execution Accuracy, so the
    EX(recall) relationship is explicit rather than three unlabeled operating points."""
    from . import schema_link as SL
    qs = B.questions()
    if n_eval < len(qs):
        qs = random.Random(seed).sample(qs, n_eval)
    members = SL.train(seed=seed)

    def table_recall(selector):
        kept = gold = 0
        for q in qs:
            g = SL.gold_tables(q["db_id"], q["SQL"])
            if not g:
                continue
            sel = selector(q)
            gold += len(g); kept += len(g & sel)
        return round(kept / gold, 4) if gold else 0.0

    selectors = {
        "full":   lambda q: set(B.schema(q["db_id"])),                                   # no pruning
        "embed":  lambda q: SL.select_tables_embed(q["db_id"], q["question"], q.get("evidence", "")),
        "learned": lambda q: SL.select_tables(members, q["db_id"], q["question"], q.get("evidence", "")),
        "oracle": lambda q: SL.gold_tables(q["db_id"], q["SQL"]) or set(B.schema(q["db_id"])),
    }
    # committed EX endpoints (full-schema=values; pruned variants), n=400 schema-linker panel
    ex_files = {"full": "values", "embed": "prune_embed_values",
                "learned": "prune_values", "oracle": "prune_oracle_values"}
    points = []
    for name, sel in selectors.items():
        rec = table_recall(sel)
        d = json.loads((ART / "bird_sql" / f"{ex_files[name]}.json").read_text())
        points.append({"linker": name, "table_recall": rec,
                       "ex": d.get("ex"), "ex_drift": d.get("ex_drift")})
        print("EX_VS_RECALL", json.dumps(points[-1]))
    points.sort(key=lambda r: r["table_recall"])
    out = {"n_eval": len(qs), "points": points}
    p = ART / "bird_sql" / "results_ex_vs_recall.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8"); print("wrote", p)
    return out


# --------------------------------------------------------------------------- Q3
def _train_regime_detector(seed: int = 0):
    """Six-feature identifier-vs-text logistic detector, trained on ICD codes
    (identifier, label 1) vs their descriptions (natural-language span, label 0).
    Trained on a SEPARATE corpus from the two test datasets, so routing generalizes
    rather than fitting BIRD/HybridQA labels."""
    from sklearn.linear_model import LogisticRegression
    recs, _ = K.build(seed=seed)
    ids = [r.table_code for r in recs]
    spans = [r.table_desc for r in recs]
    X = np.array([RI._regime_feats(s) for s in ids + spans], "float64")
    y = np.array([1] * len(ids) + [0] * len(spans))
    return LogisticRegression(max_iter=1000).fit(X, y)


def _regime_is_identifier(det, strings) -> bool:
    """Route the instance: True -> identifier regime (learned verifier), False -> text
    regime (lexical). Decided from the candidate pool's mean surface features (inputs
    only, no gold)."""
    feats = np.array([RI._regime_feats(s) for s in strings], "float64")
    return bool(det.predict_proba(feats.mean(axis=0, keepdims=True))[0, 1] >= 0.5)


def regime_router(seed: int = 0, k: int = 5, bird_q: int = 1534, hqa_dev: int = 400) -> dict:
    """Q3: a per-instance identifier-vs-text router gates between lexical matching and
    the learned verifier. We measure recall@1 for lexical-only, learned-only, and the
    router on BIRD value linking (identifier regime) and HybridQA evidence (lexical
    regime), and pooled. The claim under test: the router closes the HybridQA gap
    (where the learned verifier underperforms lexical) without hurting BIRD."""
    from . import run_hybridqa as RH
    det = _train_regime_detector(seed)

    # ---- BIRD value linking: per-instance lexical pick vs verifier pick + route
    inst = B.value_link_instances(B.questions(), max_instances=bird_q * 3)
    rng = random.Random(seed); rng.shuffle(inst)
    rows = RB.featurize(inst)
    cut = int(len(rows) * (1 - 0.35)); train_rows, test_rows = rows[:cut], rows[cut:]
    Xtr, ytr = RB.pairs_from_rows(train_rows)
    members = RB.train_ensemble(Xtr, ytr, k=k, seed=seed)
    bird = {"lexical": 0, "learned": 0, "router": 0, "route_learned": 0, "n": 0}
    for inst_i, per in test_rows:
        bird["n"] += 1
        F = np.array([f for _, f in per], "float32")
        gn = B.normalize(inst_i["gold"])
        lex_j = int(np.argmax(F[:, 3]))                       # best edit ratio (lexical)
        ver_j = int(np.argmax(RB._proba(members, F)))          # learned verifier
        use_learned = _regime_is_identifier(det, [c for c, _ in per])
        bird["route_learned"] += int(use_learned)
        pick = ver_j if use_learned else lex_j
        bird["lexical"] += int(B.normalize(per[lex_j][0]) == gn)
        bird["learned"] += int(B.normalize(per[ver_j][0]) == gn)
        bird["router"] += int(B.normalize(per[pick][0]) == gn)

    # ---- HybridQA evidence reranking: per-instance lexical pick vs verifier pick + route
    htrain = random.Random(seed).sample(RH.H.questions("train"), 800)
    hdev = random.Random(seed).sample(RH.H.questions("dev"), hqa_dev)
    HXtr, Hytr = RH.build_pairs(htrain, seed=seed)
    hmembers = RH.train_ensemble(HXtr, Hytr, k=k, seed=seed)
    hqa = {"lexical": 0, "learned": 0, "router": 0, "route_learned": 0, "n": 0}
    for q, units, gold, qvec, qtok, E in RH._records(hdev):
        hqa["n"] += 1
        dense = E @ qvec
        lex = np.array([RH._lexical(qtok, u.text)[0] for u in units])
        pool = sorted(set(np.argsort(-dense)[:20]).union(np.argsort(-lex)[:20]))
        feats = np.array([RH._features(qvec, qtok, units[i], E[i]) for i in pool], "float32")
        vs = RH._proba(hmembers, feats)
        lex_pick = pool[int(np.argmax([lex[i] for i in pool]))]
        ver_pick = pool[int(np.argmax(vs))]
        use_learned = _regime_is_identifier(det, [units[i].text for i in pool])
        hqa["route_learned"] += int(use_learned)
        pick = ver_pick if use_learned else lex_pick
        hqa["lexical"] += int(units[lex_pick].uid in gold)
        hqa["learned"] += int(units[ver_pick].uid in gold)
        hqa["router"] += int(units[pick].uid in gold)

    def rates(d):
        n = d["n"]
        return {"recall@1_lexical": round(d["lexical"] / n, 4),
                "recall@1_learned": round(d["learned"] / n, 4),
                "recall@1_router": round(d["router"] / n, 4),
                "frac_routed_to_learned": round(d["route_learned"] / n, 4), "n": n}

    nb, nh = bird["n"], hqa["n"]
    pooled = {kk: round((bird[kk2] + hqa[kk2]) / (nb + nh), 4)
              for kk, kk2 in (("recall@1_lexical", "lexical"), ("recall@1_learned", "learned"),
                              ("recall@1_router", "router"))}
    out = {"bird": rates(bird), "hybridqa": rates(hqa), "pooled": pooled,
           "config": {"seed": seed, "k": k}}
    p = ART / "hybridqa" / "results_regime_router.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("REGIME_ROUTER", json.dumps(out)); print("wrote", p)
    return out


def bird_ex_oracle_coverage(n: int = 400, seed: int = 0,
                            coverages=(0.4, 0.6, 0.8, 1.0),
                            model_id: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit") -> dict:
    """Q2 (controlled): EX as a function of table-pruning *recall*. Starting from the
    oracle table set (gold-SQL tables, which is the recall=1 ceiling), we drop each gold
    table independently with probability ``1-c`` to synthesize an oracle subset at target
    coverage ``c``, then run the same value-augmented text-to-SQL pipeline and measure the
    realized table-recall and Execution Accuracy. This traces EX down the pruning-recall
    axis under otherwise identical prompts."""
    import time

    from mlx_lm import generate, load
    from . import bird_sql as S
    from . import schema_link as SL

    model, tok = load(model_id)
    qs = random.Random(seed).sample(B.questions(), min(n, len(B.questions())))
    rows = []
    for c in coverages:
        rs = random.Random(7000 + int(c * 100)); t0 = time.time()
        ok = 0; kept = goldn = 0; nn = 0
        for q in qs:
            gold = SL.gold_tables(q["db_id"], q["SQL"])
            if not gold:
                continue
            nn += 1
            tables = {t for t in gold if rs.random() < c} or set(list(gold)[:1])  # keep >=1 to form a prompt
            kept += len(tables & gold); goldn += len(gold)
            ev = q.get("evidence", "")
            prompt = S.build_prompt(q["db_id"], q["question"], ev, include_values=True, tables=tables)
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, tokenize=False)
            pred = S.extract_sql(generate(model, tok, prompt=text, max_tokens=256, verbose=False))
            ok += int(S.execution_match(q["db_id"], pred, q["SQL"]))
        row = {"target_coverage": c, "table_recall": round(kept / goldn, 4) if goldn else 0.0,
               "ex": round(ok / nn, 4) if nn else 0.0, "n": nn, "runtime_s": round(time.time() - t0, 1)}
        rows.append(row); print("EX_ORACLE_COV", json.dumps(row), flush=True)
    out = {"config": {"n": n, "seed": seed, "model": model_id}, "by_coverage": rows}
    p = ART / "bird_sql" / "results_ex_oracle_coverage.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8"); print("wrote", p)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "ex_oracle_cov":
        bird_ex_oracle_coverage(n=next((int(a) for a in sys.argv if a.isdigit()), 400))
        sys.exit(0)
    if which in ("icd_lc", "all"):
        icd_learning_curve()
    if which in ("bird_distract", "all"):
        bird_distractor_robustness()
    if which in ("ex_recall", "all"):
        bird_ex_vs_pruning_recall()
    if which in ("router", "all"):
        regime_router()
