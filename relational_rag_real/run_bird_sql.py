"""Phase 5 end-to-end BIRD text-to-SQL with value-linking ablation.

Modes:
  baseline  -- schema columns only (model guesses literals)
  values    -- schema + sample column values (perfect value availability; a ceiling)
  verifier  -- columns only + value hints from the value-linking verifier (drift-focused)

``repair=True`` re-generates once on execution error. Writes per-question records so
EX can be sliced by drift / difficulty later.
Run: ``env/bin/python -m relational_rag_real.run_bird_sql <mode>``.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from . import bird as B
from . import bird_sql as S
from . import schema_link as SL

RESULTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "bird_sql"
DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

_colval_cache: dict[str, list] = {}


def _low_card_text_cols(db_id: str, max_card: int = 50, max_val_len: int = 25):
    """Low-cardinality *categorical* text columns (short values) -- value-linking
    targets. Excludes free-text columns (comments, descriptions) by value length."""
    out = []
    for table, cols in B.schema(db_id).items():
        for col in cols:
            vals = B.column_values(db_id, table, col, cap=max_card + 1)
            tv = [v for v in vals if v and not v.replace(".", "").replace("-", "").isdigit()]
            if 2 <= len(tv) <= max_card and max(len(v) for v in tv) <= max_val_len:
                out.append((table, col, tv))
    return out


def verifier_value_hints(db_id: str, question: str, evidence: str, *,
                         thr: float = 0.66, col_thr: float = 0.34, max_hints: int = 3):
    """Column-GROUNDED, drift-focused value linking. A hint is proposed only when the
    column is relevant to the question (its name overlaps/embeds near the question) AND
    its best value matches the question strongly AND is not verbatim (a drift case).
    Conservative (<=3 hints) to avoid the noise that ungrounded hints caused (Q1)."""
    from .embed import embed

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
            entry.append((table, col, tv, E[idx : idx + len(tv)], colvecs[k]))
            idx += len(tv)
        _colval_cache[db_id] = entry

    qn = B.normalize(question)
    qtok = set(qn.split())
    qv = embed([f"{question} {evidence}"], is_query=True)[0]
    cands = []
    for table, col, tv, E, cvec in _colval_cache[db_id]:
        if len(E) == 0:
            continue
        col_tok = set(B.normalize(col).split())
        col_rel = max(len(col_tok & qtok) / len(col_tok) if col_tok else 0.0, float(cvec @ qv))
        if col_rel < col_thr:           # column not relevant to the question -> skip
            continue
        scores = E @ qv
        j = int(np.argmax(scores))
        v = tv[j]
        if B.normalize(v) in qn:        # verbatim -> model can already find it
            continue
        if float(scores[j]) < thr:
            continue
        cands.append((float(scores[j]) + 0.1 * col_rel, f"{table}.{col} = '{v}'"))
    cands.sort(reverse=True)
    return [h for _s, h in cands][:max_hints]


def _gen(model, tok, generate, prompt: str, max_tokens: int) -> str:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    return S.extract_sql(generate(model, tok, prompt=text, max_tokens=max_tokens, verbose=False))


def _eval(model, tok, generate, qs, mode, repair, max_tokens, model_id, seed, linker=None) -> dict:
    t0 = time.time()
    per_q = []
    for q in qs:
        ev = q.get("evidence", "")
        if mode in ("verifier", "prune_sel"):
            thr, mh = (0.62, 3) if mode == "verifier" else (0.72, 2)
            hints = verifier_value_hints(q["db_id"], q["question"], ev, thr=thr, max_hints=mh)
        elif mode == "verifier_sel":
            hints = verifier_value_hints(q["db_id"], q["question"], ev, thr=0.72, max_hints=2)
        else:
            hints = None
        tables = None
        if mode.startswith("prune"):
            if "oracle" in mode:
                tables = SL.gold_tables(q["db_id"], q["SQL"])          # upper bound
            elif "embed" in mode:
                tables = SL.select_tables_embed(q["db_id"], q["question"], ev, top_t=6)  # no-training
            elif linker is not None:
                tables = SL.select_tables(linker, q["db_id"], q["question"], ev, top_t=6)  # learned
        prompt = S.build_prompt(q["db_id"], q["question"], ev,
                                include_values=("values" in mode), hints=hints, tables=tables)
        pred = _gen(model, tok, generate, prompt, max_tokens)
        _, perr = S.execute(q["db_id"], pred)
        if repair and perr is not None:
            rp = prompt + f"\n\n-- The query above failed with error: {perr[:120]}\n-- Corrected SQL:"
            pred = _gen(model, tok, generate, rp, max_tokens)
            _, perr = S.execute(q["db_id"], pred)
        ok = S.execution_match(q["db_id"], pred, q["SQL"])
        is_drift = any(B.normalize(v) not in B.normalize(q["question"])
                       for _c, v in B.extract_conditions(q["SQL"]))
        per_q.append({"db": q["db_id"], "difficulty": q.get("difficulty", ""),
                      "ok": bool(ok), "err": bool(perr is not None), "drift": bool(is_drift),
                      "n_hints": len(hints or [])})
    drift = [r for r in per_q if r["drift"]]
    results = {
        "config": {"mode": mode, "repair": repair, "model": model_id, "n": len(per_q),
                   "seed": seed, "runtime_s": round(time.time() - t0, 1)},
        "ex": round(sum(r["ok"] for r in per_q) / len(per_q), 4),
        "exec_error_rate": round(sum(r["err"] for r in per_q) / len(per_q), 4),
        "ex_drift": round(sum(r["ok"] for r in drift) / len(drift), 4) if drift else None,
        "drift_n": len(drift), "per_question": per_q,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = mode + ("_repair" if repair else "")
    (RESULTS_DIR / f"{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[{tag}] EX={results['ex']:.4f} exec_err={results['exec_error_rate']:.4f} "
          f"EX_drift={results['ex_drift']} (drift_n={len(drift)}) "
          f"n={len(per_q)} {results['config']['runtime_s']}s", flush=True)
    return results


def main(mode: str = "baseline", n: int = 1534, seed: int = 0,
         model_id: str = DEFAULT_MODEL, max_tokens: int = 256, repair: bool = False) -> dict:
    from mlx_lm import generate, load

    model, tok = load(model_id)
    qs = random.Random(seed).sample(B.questions(), min(n, len(B.questions())))
    linker = SL.train(seed=seed) if mode.startswith("prune") else None
    return _eval(model, tok, generate, qs, mode, repair, max_tokens, model_id, seed, linker)


# The exact ablation reported in the paper's end-to-end EX table (tab:ex).
PAPER_CONFIGS = (("baseline", False), ("values", False), ("verifier", False),
                 ("verifier_sel", False), ("prune", False), ("prune_values", False), ("prune_sel", False))
# The schema-linker comparison in tab:slink (full / no-training-embed / learned / oracle).
SLINK_CONFIGS = (("values", False), ("prune_embed_values", False), ("prune_values", False), ("prune_oracle_values", False))


def run_ablation(n: int = 1534, seed: int = 0, model_id: str = DEFAULT_MODEL,
                 configs=PAPER_CONFIGS) -> None:
    """Load the model ONCE and run every (mode, repair) config in sequence."""
    from mlx_lm import generate, load

    model, tok = load(model_id)
    qs = random.Random(seed).sample(B.questions(), min(n, len(B.questions())))
    linker = SL.train(seed=seed) if any(m.startswith("prune") for m, _ in configs) else None
    for mode, repair in configs:
        _eval(model, tok, generate, qs, mode, repair, 256, model_id, seed, linker)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        run_ablation(n=next((int(a) for a in sys.argv if a.isdigit()), 1534))
    elif len(sys.argv) > 1 and sys.argv[1] == "slink":
        run_ablation(n=next((int(a) for a in sys.argv if a.isdigit()), 800), configs=SLINK_CONFIGS)
    else:
        mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
        nn = next((int(a) for a in sys.argv if a.isdigit()), 1534)
        main(mode=mode, n=nn, repair="--repair" in sys.argv)
