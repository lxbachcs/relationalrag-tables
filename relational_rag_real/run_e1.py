"""Rebuttal E1: verifier-isolated end-to-end BIRD ablation (CUDA server, vLLM backend).

Holds the pipeline fixed at "learned schema pruning + column values" and varies ONLY the
value linker that produces the value hints injected into the prompt:

  none      no hints (the paper's ``prune_values`` row)
  exact     containment: the DB value appears verbatim in the question
  fuzzy     best token edit ratio (the paper's fuzzy tool, threshold 0.6)
  embed     bge-small cosine between question and ``column: value`` (threshold 0.62)
  verifier  the learned MLP ensemble of ``run_bird`` over identifier+embedding features,
            cross-fitted over two question folds so no question scores its own training pair
  oracle    gold literals parsed from the gold SQL (upper bound)

All linkers share the same candidate pool (low-cardinality text columns of pruned tables
whose name is relevant to the question), the same cap of three hints, and the same
prompt slot; only the matching function differs. Per-question records keep the predicted
SQL, hints, scores, and the verifier's uncertainty so that end-to-end selective
prediction can be computed offline.

Run (server):
  .venv/bin/python -m relational_rag_real.run_e1 --configs all --n 1534
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from . import bird as B
from . import bird_sql as S
from . import run_bird as RB
from . import schema_link as SL
from .embed import embed
from .run_bird_sql import _low_card_text_cols

RESULTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "e1"
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
LINKERS = ("none", "exact", "fuzzy", "embed", "verifier", "oracle")
THRESH = {"exact": 0.5, "fuzzy": 0.6, "embed": 0.62, "verifier": 0.5, "oracle": 0.5}
COL_THR = 0.34
MAX_HINTS = 3

_colval_cache: dict[str, list] = {}


# --------------------------------------------------------------------------- SQLite
def execute(db_id: str, sql: str, timeout: float = 30.0):
    """Read-only execution with a wall-clock interrupt (a runaway query must not hang)."""
    try:
        con = sqlite3.connect(f"file:{B.db_path(db_id)}?mode=ro", uri=True, timeout=timeout)
        con.text_factory = lambda b: b.decode("utf-8", "ignore")
        timer = threading.Timer(timeout, con.interrupt)
        timer.start()
        try:
            rows = con.cursor().execute(sql).fetchall()
        finally:
            timer.cancel()
            con.close()
        return rows, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def execution_match(db_id: str, pred_sql: str, gold_sql: str) -> bool:
    pred, perr = execute(db_id, pred_sql)
    gold, gerr = execute(db_id, gold_sql)
    if perr is not None or gerr is not None or gold is None:
        return False
    return {tuple(r) for r in pred} == {tuple(r) for r in gold}


# ------------------------------------------------------------------------- linkers
def _db_entries(db_id: str):
    if db_id not in _colval_cache:
        cols = _low_card_text_cols(db_id)
        flat, spans = [], []
        for table, col, tv in cols:
            spans.append((table, col, tv))
            flat.extend(f"{col}: {v}" for v in tv)
        E = embed(flat, batch_size=512) if flat else np.zeros((0, 384), "float32")
        colvecs = embed([f"{t} {c}" for t, c, _ in spans]) if spans else np.zeros((0, 384), "float32")
        entry, idx = [], 0
        for k, (table, col, tv) in enumerate(spans):
            entry.append((table, col, tv, E[idx: idx + len(tv)], colvecs[k]))
            idx += len(tv)
        _colval_cache[db_id] = entry
    return _colval_cache[db_id]


def gold_literals(q: dict) -> list[tuple[str, str, str]]:
    """(table, column, value) for every ``col = 'value'`` literal of the gold SQL."""
    out = []
    sch = B.schema(q["db_id"])
    for col, val in B.extract_conditions(q["SQL"]):
        for table, cols in sch.items():
            lower = {c.lower(): c for c in cols}
            if col.lower() not in lower:
                continue
            real = lower[col.lower()]
            vals = B.column_values(q["db_id"], table, real)
            if any(B.normalize(v) == B.normalize(val) for v in vals):
                out.append((table, real, val))
                break
    return out


def link_hints(q: dict, linker: str, tables, members=None, *, skip_verbatim: bool = False):
    """Return (hints, scores) for one question under one linker (``name@thr`` overrides the threshold)."""
    linker, _, thr_s = linker.partition("@")
    thr = float(thr_s) if thr_s else THRESH.get(linker, 0.5)
    if linker == "none":
        return [], []
    qn = B.normalize(q["question"])
    evn = B.normalize(q.get("evidence", ""))
    qtok = set(qn.split())
    if linker == "oracle":
        hints = [(1.0, f"{t}.{c} = '{v}'") for t, c, v in gold_literals(q)
                 if (tables is None or t.lower() in {x.lower() for x in tables})
                 and not (skip_verbatim and B.normalize(v) in qn)]
        hints = hints[:MAX_HINTS]
        return [h for _, h in hints], [s for s, _ in hints]
    qv_query = embed([f"{q['question']} {q.get('evidence', '')}"], is_query=True)[0]
    qv_plain = embed([f"{q['question']} {q.get('evidence', '')}"])[0] if linker == "verifier" else None
    keep = {t.lower() for t in tables} if tables else None
    cands = []
    for table, col, tv, E, cvec in _db_entries(q["db_id"]):
        if keep is not None and table.lower() not in keep:
            continue
        if len(E) == 0:
            continue
        col_tok = set(B.normalize(col).split())
        col_rel = max(len(col_tok & qtok) / len(col_tok) if col_tok else 0.0, float(cvec @ qv_query))
        if col_rel < COL_THR:
            continue
        if linker == "exact":
            scores = np.array([1.0 if B.normalize(v) and B.normalize(v) in qn else 0.0 for v in tv])
            scores = scores + 1e-3 * np.array([len(v) for v in tv]) * (scores > 0)  # prefer longest
        elif linker == "fuzzy":
            scores = np.array([max((SequenceMatcher(None, B.normalize(v), t).ratio() for t in qtok), default=0.0)
                               for v in tv])
        elif linker == "embed":
            scores = E @ qv_query
        elif linker == "verifier":
            F = np.array([RB.features(qn, evn, qtok, qv_plain, v, E[j]) for j, v in enumerate(tv)], "float32")
            scores = RB._proba(members, F)
        else:
            raise ValueError(linker)
        j = int(np.argmax(scores))
        v = tv[j]
        s = float(scores[j])
        if s < thr:
            continue
        if skip_verbatim and B.normalize(v) in qn:
            continue
        cands.append((s + 0.1 * col_rel, s, f"{table}.{col} = '{v}'"))
    cands.sort(reverse=True)
    cands = cands[:MAX_HINTS]
    return [h for _, _, h in cands], [round(s, 4) for _, s, _ in cands]


def train_crossfit_verifiers(qs: list[dict], seed: int, k: int = 5):
    """Two verifier ensembles; fold f is trained on the OTHER fold's value-linking pairs."""
    folds = [[q for i, q in enumerate(qs) if i % 2 == f] for f in (0, 1)]
    members = []
    for f in (0, 1):
        inst = B.value_link_instances(folds[1 - f], max_instances=10000)
        rows = RB.featurize(inst)
        X, y = RB.pairs_from_rows(rows)
        members.append(RB.train_ensemble(X, y, k=k, seed=seed + f))
        print(f"[crossfit] fold {f}: trained on {len(inst)} instances / {len(X)} pairs from fold {1 - f}", flush=True)
    return members


# --------------------------------------------------------------------------- prompts
def build_prompt(q: dict, *, include_values: bool, hints, tables, tok, max_prompt_tokens: int):
    prompt = S.build_prompt(q["db_id"], q["question"], q.get("evidence", ""),
                            include_values=include_values, hints=hints, tables=tables)
    truncated = False
    if len(tok(prompt)["input_ids"]) > max_prompt_tokens and include_values:
        prompt = S.build_prompt(q["db_id"], q["question"], q.get("evidence", ""),
                                include_values=False, hints=hints, tables=tables)
        truncated = True
    ids = tok(prompt)["input_ids"]
    if len(ids) > max_prompt_tokens:  # hard cut inside the schema block, keep the question tail
        head, tail = prompt.split("\n\nQuestion:", 1)
        head_ids = tok(head)["input_ids"][: max_prompt_tokens - len(tok("\n\nQuestion:" + tail)["input_ids"]) - 8]
        prompt = tok.decode(head_ids) + "\n\nQuestion:" + tail
        truncated = True
    return prompt, truncated


def chat(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)


# -------------------------------------------------------------------------- one run
def run_config(llm, tok, sp, qs, *, tag: str, prune: bool, include_values: bool, linker: str,
               skip_verbatim: bool, sl_members, cf_members, max_prompt_tokens: int, seed: int) -> dict:
    t0 = time.time()
    records, prompts = [], []
    for i, q in enumerate(qs):
        ev = q.get("evidence", "")
        tables = SL.select_tables(sl_members, q["db_id"], q["question"], ev, top_t=6) if prune else None
        members = cf_members[i % 2] if (linker.startswith("verifier") and cf_members) else None
        hints, scores = link_hints(q, linker, tables, members, skip_verbatim=skip_verbatim)
        prompt, truncated = build_prompt(q, include_values=include_values, hints=hints, tables=tables,
                                         tok=tok, max_prompt_tokens=max_prompt_tokens)
        gold = {B.normalize(v) for _t, _c, v in gold_literals(q)}
        hint_vals = {B.normalize(h.split(" = '", 1)[1][:-1]) for h in hints}
        records.append({"i": i, "db": q["db_id"], "difficulty": q.get("difficulty", ""),
                        "drift": any(B.normalize(v) not in B.normalize(q["question"])
                                     for _c, v in B.extract_conditions(q["SQL"])),
                        "hints": hints, "hint_scores": scores,
                        "hint_min_score": min(scores) if scores else None,
                        "n_gold_literals": len(gold), "hint_hits": len(hint_vals & gold),
                        "n_tables": len(tables) if tables else None, "truncated": truncated})
        prompts.append(chat(tok, prompt))
    t_prep = time.time() - t0
    outs = llm.generate(prompts, sp)
    t_gen = time.time() - t0 - t_prep
    for rec, out in zip(records, outs):
        q = qs[rec["i"]]
        pred = S.extract_sql(out.outputs[0].text)
        _, perr = execute(q["db_id"], pred)
        rec["sql"] = pred
        rec["err"] = perr is not None
        rec["ok"] = bool(execution_match(q["db_id"], pred, q["SQL"])) if perr is None else False
    drift = [r for r in records if r["drift"]]
    with_hints = [r for r in records if r["hints"]]
    with_gold = [r for r in records if r["n_gold_literals"]]
    res = {
        "config": {"tag": tag, "model": MODEL, "n": len(records), "seed": seed, "prune": prune,
                   "include_values": include_values, "linker": linker, "skip_verbatim": skip_verbatim,
                   "max_hints": MAX_HINTS, "thresholds": THRESH, "col_thr": COL_THR,
                   "prep_s": round(t_prep, 1), "gen_s": round(t_gen, 1), "runtime_s": round(time.time() - t0, 1)},
        "ex": round(sum(r["ok"] for r in records) / len(records), 4),
        "ex_drift": round(sum(r["ok"] for r in drift) / len(drift), 4) if drift else None,
        "drift_n": len(drift),
        "exec_error_rate": round(sum(r["err"] for r in records) / len(records), 4),
        "hints": {"questions_with_hints": len(with_hints),
                  "mean_hints": round(sum(len(r["hints"]) for r in records) / len(records), 3),
                  "hint_precision": round(sum(r["hint_hits"] for r in with_hints) / max(1, sum(len(r["hints"]) for r in with_hints)), 4),
                  "gold_literal_recall": round(sum(r["hint_hits"] for r in with_gold) / max(1, sum(r["n_gold_literals"] for r in with_gold)), 4),
                  "ex_with_hints": round(sum(r["ok"] for r in with_hints) / len(with_hints), 4) if with_hints else None,
                  "truncated": sum(r["truncated"] for r in records)},
        "per_question": records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[{tag}] EX={res['ex']} EX_drift={res['ex_drift']} (n_drift={len(drift)}) err={res['exec_error_rate']} "
          f"hints={res['hints']['questions_with_hints']}q prec={res['hints']['hint_precision']} "
          f"rec={res['hints']['gold_literal_recall']} trunc={res['hints']['truncated']} "
          f"[{res['config']['runtime_s']}s]", flush=True)
    return res


def config_list(which: str):
    base = [("baseline", False, False, "none", False), ("pv_none", True, True, "none", False)]
    hinted = [(f"pv_{l}", True, True, l, False) for l in LINKERS if l != "none"]
    drift_only = [(f"pv_{l}_driftonly", True, True, l, True) for l in LINKERS if l != "none"]
    strict = [("pv_verifier_t80", True, True, "verifier@0.8", False), ("pv_verifier_t90", True, True, "verifier@0.9", False),
              ("pv_embed_t75", True, True, "embed@0.75", False), ("pv_verifier_t80_driftonly", True, True, "verifier@0.8", True)]
    if which == "all":
        return base + hinted + drift_only + strict
    if which == "core":
        return base + hinted
    if which == "driftonly":
        return drift_only
    if which == "smoke":
        return [("pv_none", True, True, "none", False), ("pv_verifier", True, True, "verifier", False)]
    return [c for c in base + hinted + drift_only if c[0] in which.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="core")
    ap.add_argument("--n", type=int, default=1534)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu-mem", type=float, default=0.115)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    a = ap.parse_args(argv)

    embed(["warm up the embedding model on the GPU before vLLM profiles free memory"])
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, enforce_eager=True, max_num_seqs=a.max_num_seqs, seed=a.seed)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    max_prompt_tokens = a.max_model_len - a.max_tokens - 64

    qs = random.Random(a.seed).sample(B.questions(), min(a.n, len(B.questions())))
    configs = config_list(a.configs)
    sl_members = SL.train(seed=a.seed) if any(c[1] for c in configs) else None
    cf_members = train_crossfit_verifiers(qs, a.seed) if any(c[3].startswith("verifier") for c in configs) else None
    for tag, prune, values, linker, skip in configs:
        run_config(llm, tok, sp, qs, tag=tag, prune=prune, include_values=values, linker=linker,
                   skip_verbatim=skip, sl_members=sl_members, cf_members=cf_members,
                   max_prompt_tokens=max_prompt_tokens, seed=a.seed)


if __name__ == "__main__":
    main()
