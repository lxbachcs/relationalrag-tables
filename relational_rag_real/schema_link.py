"""Learned schema linker for BIRD: select the relevant tables for a question so the
prompt schema can be pruned (a standard EX booster on large databases) and value
hints grounded. Trained by distant supervision on which tables the gold SQL uses.
"""

from __future__ import annotations

import random
import re

import numpy as np

from . import bird as B
from .embed import embed

_ctx_cache: dict[str, tuple] = {}  # db_id -> (table_names, context_embeddings)


def gold_tables(db_id: str, sql: str) -> set[str]:
    """Tables referenced in the gold SQL (word-boundary name match)."""
    flat = " " + re.sub(r"[^A-Za-z0-9_]", " ", sql).lower() + " "
    return {t for t in B.schema(db_id) if f" {t.lower()} " in flat}


def _ctx(db_id: str):
    if db_id not in _ctx_cache:
        sch = B.schema(db_id)
        tables = list(sch)
        texts = [f"{t}: {', '.join(sch[t])}" for t in tables]
        E = embed(texts) if texts else np.zeros((0, 384), "float32")
        _ctx_cache[db_id] = (tables, E)
    return _ctx_cache[db_id]


def _feats(qvec, qtok, table, cols, tvec) -> list[float]:
    cos = float(qvec @ tvec)
    name_tok = set(B.normalize(table).split()) | set(B.normalize(" ".join(cols)).split())
    overlap = len(name_tok & qtok) / len(qtok) if qtok else 0.0
    table_in_q = 1.0 if B.normalize(table) in " ".join(qtok) else 0.0
    return [cos, overlap, table_in_q, min(len(cols), 20) / 20.0]


def train(n_train: int = 1500, seed: int = 0, k: int = 3):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    qs = random.Random(seed).sample(B.questions(), min(n_train, len(B.questions())))
    qvecs = embed([f"{q['question']} {q.get('evidence', '')}" for q in qs], is_query=True)
    X, y = [], []
    for q, qvec in zip(qs, qvecs):
        gold = gold_tables(q["db_id"], q["SQL"])
        tables, E = _ctx(q["db_id"])
        sch = B.schema(q["db_id"])
        qtok = set(B.normalize(q["question"]).split())
        for ti, t in enumerate(tables):
            X.append(_feats(qvec, qtok, t, sch[t], E[ti]))
            y.append(1 if t in gold else 0)
    X, y = np.array(X, "float32"), np.array(y)
    archs = [(32, 16), (24,), (40, 20)]
    rng = np.random.default_rng(seed)
    members = []
    for m in range(k):
        idx = rng.integers(0, len(X), len(X))
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=archs[m % len(archs)], alpha=1e-3,
                                          max_iter=400, random_state=seed + m))
        clf.fit(X[idx], y[idx])
        members.append(clf)
    return members


def select_tables_embed(db_id: str, question: str, evidence: str, *, top_t: int = 6) -> set[str]:
    """No-training baseline: select tables by raw embedding cosine to the question."""
    tables, E = _ctx(db_id)
    if len(tables) <= top_t:
        return set(tables)
    qv = embed([f"{question} {evidence}"], is_query=True)[0]
    s = E @ qv
    return {tables[i] for i in np.argsort(-s)[:top_t]}


def select_tables(members, db_id: str, question: str, evidence: str, *, top_t: int = 5) -> set[str]:
    tables, E = _ctx(db_id)
    if len(tables) <= top_t:
        return set(tables)
    sch = B.schema(db_id)
    qvec = embed([f"{question} {evidence}"], is_query=True)[0]
    qtok = set(B.normalize(question).split())
    feats = np.array([_feats(qvec, qtok, t, sch[t], E[ti]) for ti, t in enumerate(tables)], "float32")
    s = np.mean([m.predict_proba(feats)[:, 1] for m in members], axis=0)
    return {tables[i] for i in np.argsort(-s)[:top_t]}
