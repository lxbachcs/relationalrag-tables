"""Cheap probe: does an off-the-shelf cross-encoder beat lexical at HybridQA top-1?

Reranks the lexical top-k pool with a pretrained MiniLM cross-encoder (no
fine-tuning). If cross-encoder recall@1 > lexical recall@1, the stronger-expert
path (Section: universal winner) is worth pursuing; if it only ties, the task is
lexically ceilinged. Run: ``env/bin/python -m relational_rag_real.probe_ce``.
"""

from __future__ import annotations

import random
import time

import numpy as np

from . import hybridqa as H


def _cov(qtok: set, text: str) -> float:
    t = set(H.normalize(text).split())
    return len(qtok & t) / len(qtok) if (qtok and t) else 0.0


def main(n_dev: int = 400, topk: int = 20, seed: int = 0,
         model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
    from sentence_transformers import CrossEncoder

    t0 = time.time()
    ce = CrossEncoder(model)
    dev = random.Random(seed).sample(H.questions("dev"), n_dev)

    lex_r1 = ce_r1 = pool_ceiling = n = 0
    for q in dev:
        try:
            units = H.table_units(q["table_id"])
        except FileNotFoundError:
            continue
        gold = {u.uid for u in H.answer_bearing_units(q, units)}
        if not gold or not units:
            continue
        n += 1
        qtok = set(H.normalize(q["question"]).split())
        order = sorted(range(len(units)), key=lambda i: -_cov(qtok, units[i].text))
        pool = order[:topk]
        if {units[i].uid for i in pool} & gold:
            pool_ceiling += 1
        if units[pool[0]].uid in gold:        # lexical top-1
            lex_r1 += 1
        scores = ce.predict([(q["question"], units[i].text) for i in pool])
        best = pool[int(np.argmax(scores))]   # cross-encoder top-1
        if units[best].uid in gold:
            ce_r1 += 1

    print(f"n={n}  pool_recall@{topk}={pool_ceiling/n:.4f}  "
          f"lexical_r1={lex_r1/n:.4f}  cross_encoder_r1={ce_r1/n:.4f}  "
          f"({time.time()-t0:.0f}s, {model})")


if __name__ == "__main__":
    main()
