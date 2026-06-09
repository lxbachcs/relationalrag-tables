"""Synthetic heterogeneous (tables + KG) corpus and question generator.

This is the self-contained analogue of the paper's "synthetic re-instantiation
script" (Section: Reproducibility). It deterministically builds a relational
``accounts`` table joined against a financial-instruments knowledge graph and a
suite of questions spanning the paper's task families:

* ``single_table_lookup`` — answerable from the table alone;
* ``table_kg_join`` — 1-hop table->KG join + traverse;
* ``multi_hop_traverse`` — 2-3 hop traversal chains after the join;
* ``bad_join_backtrack`` — a near-duplicate KG node is offered first and must be
  rejected by the verifier before the correct node is used;
* ``abstain`` — "ghost" accounts absent from the KG, whose only correct answer
  is to refuse to answer.

The join challenge mirrors the paper's ``ICD-10 J45.901`` vs ``J45.9`` example:
table account codes use a dash (``ACC-1001``) while KG nodes drop it
(``account:ACC1001``) and near-duplicates differ by a single digit.

Everything is seeded, so the same seed reproduces byte-identical fixtures.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

RISK_RATINGS = ["low", "medium", "high"]
REGIONS = ["emea", "amer", "apac"]
SECTORS = ["technology", "energy", "healthcare", "financials", "industrials"]
INSTRUMENTS = [
    "equity_swap",
    "municipal_bond",
    "treasury_note",
    "credit_default_swap",
    "convertible_bond",
    "index_future",
]
# instrument -> asset_class
ASSET_CLASS = {
    "equity_swap": "derivative",
    "municipal_bond": "fixed_income",
    "treasury_note": "fixed_income",
    "credit_default_swap": "derivative",
    "convertible_bond": "hybrid",
    "index_future": "derivative",
}
# asset_class -> regulator
REGULATOR = {
    "derivative": "cftc",
    "fixed_income": "sec",
    "hybrid": "finra",
}
COMPLIANCE_FLAGS = ["cleared", "review_required", "escalated"]
FIRST_NAMES = ["Alice", "Ben", "Carla", "Deepa", "Elias", "Fatima", "Geon", "Hana", "Ivan", "Julia"]
LAST_NAMES = ["Nguyen", "Carter", "Singh", "Rao", "Khan", "Owusu", "Park", "Silva", "Petrov", "Cohen"]


@dataclass
class GenConfig:
    num_accounts: int = 600
    seed: int = 7
    train_frac: float = 0.5
    dev_frac: float = 0.1
    ghost_frac: float = 0.12  # accounts present in table but absent from KG
    # Fraction of (non-ghost) KG nodes whose identifier is format-drifted relative
    # to the table key: the numeric key is preserved but the alphabetic prefix is
    # mutated (e.g. table ``ACC-1001`` vs node ``account:ACX1001``). This breaks
    # substring/exact tools while leaving the join recoverable by a verifier that
    # has learned to weight digit agreement -- the paper's ``J45.901`` vs ``J45.9``
    # regime, surfaced in the end-to-end QA joins rather than only in isolation.
    qa_join_drift: float = 0.0
    questions_per_account: float = 1.0
    tau_v: float = 0.5
    # Abstention threshold on the uncertainty head's signal, which is the entropy of
    # the ensemble-mean decision (range [0, ln 2]); a question is deferred when its
    # entropy exceeds this. The study uses selective prediction (ranking by this
    # signal) rather than a single hard cut, so this default only affects the optional
    # --uncertainty-gate path.
    tau_u: float = 0.5
    top_k: int = 8
    # Verifier learning set: graded identifier corruption produces a realistic
    # hard region with an aleatoric "same-entity" boundary, so the trained
    # verifier has a non-degenerate operating point and the deep ensemble has
    # meaningful predictive variance.
    n_verifier_train: int = 4000
    n_verifier_eval: int = 1500


@dataclass
class GeneratedCorpus:
    accounts: list[dict[str, str]]
    triples: list[dict[str, str]]
    questions: list[dict[str, object]]
    join_labels: list[dict[str, object]]
    verifier_pairs: list[dict[str, object]]
    verifier_eval: list[dict[str, object]]
    config: dict[str, object]
    meta: dict[str, object] = field(default_factory=dict)


# Probability that a corrupted identifier still refers to the SAME entity,
# indexed by the number of character edits. The boundary at 2 edits is
# deliberately ~50/50 so observable features cannot fully resolve the label.
_SAME_ENTITY_PROB = {0: 0.99, 1: 0.85, 2: 0.5, 3: 0.18, 4: 0.04}
_DIGITS = "0123456789"
_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _corrupt(code: str, n_edits: int, rng: random.Random) -> str:
    chars = list(code)
    for _ in range(n_edits):
        if not chars:
            break
        pos = rng.randrange(len(chars))
        op = rng.choice(("sub", "sub", "del", "ins"))  # bias toward substitution
        if op == "del" and len(chars) > 3:
            chars.pop(pos)
        elif op == "ins":
            alphabet = _DIGITS if chars[pos].isdigit() else _ALPHA
            chars.insert(pos, rng.choice(alphabet))
        else:
            alphabet = _DIGITS if chars[pos].isdigit() else _ALPHA
            chars[pos] = rng.choice(alphabet)
    return "".join(chars)


def _drift_alpha(code: str, rng: random.Random) -> str:
    """Mutate only the alphabetic prefix of ``code`` by one edit (digits preserved).

    Returns a variant that differs from ``code`` in its prefix so that substring /
    exact matching against the table key fails, while the numeric key -- the signal
    a learned verifier can exploit -- is left intact.
    """
    split = 0
    while split < len(code) and not code[split].isdigit():
        split += 1
    alpha, digits = code[:split], code[split:]
    variants: list[str] = []
    if alpha:
        pos = rng.randrange(len(alpha))
        variants.append(alpha[:pos] + rng.choice(_ALPHA) + alpha[pos + 1 :])  # substitute
        if len(alpha) > 1:
            variants.append(alpha[:pos] + alpha[pos + 1 :])  # delete
    insert_pos = rng.randrange(len(alpha) + 1)
    variants.append(alpha[:insert_pos] + rng.choice(_ALPHA) + alpha[insert_pos:])  # insert
    candidates = [v for v in variants if v and v != alpha] or [alpha + rng.choice(_ALPHA)]
    return rng.choice(candidates) + digits


def _verifier_pair(idx: int, rng: random.Random) -> tuple[str, str, bool]:
    """Build one (lhs_code, rhs_node, is_same_entity) corruption pair."""
    base = f"ACC{1000 + idx}"
    lhs = f"ACC-{1000 + idx}"
    n_edits = rng.choices([0, 1, 2, 3, 4], weights=[3, 4, 4, 3, 2])[0]
    variant = _corrupt(base, n_edits, rng) if n_edits else base
    same = rng.random() < _SAME_ENTITY_PROB[n_edits]
    rhs = f"account:{variant}"
    return lhs, rhs, same


def make_verifier_dataset(
    cfg: GenConfig, rng: random.Random
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train: list[dict[str, object]] = []
    eval_set: list[dict[str, object]] = []
    for i in range(cfg.n_verifier_train):
        lhs, rhs, same = _verifier_pair(i % cfg.num_accounts, rng)
        train.append({"lhs": lhs, "rhs": rhs, "label": same})
    for i in range(cfg.n_verifier_eval):
        lhs, rhs, same = _verifier_pair((i + 7919) % cfg.num_accounts, rng)
        eval_set.append({"lhs": lhs, "rhs": rhs, "label": same})
    return train, eval_set


def _node_id(idx: int) -> str:
    return f"account:ACC{1000 + idx}"


def _account_code(idx: int) -> str:
    return f"ACC-{1000 + idx}"


def _account_id(idx: int) -> str:
    return f"A-{1000 + idx}"


def generate(cfg: GenConfig) -> GeneratedCorpus:
    rng = random.Random(cfg.seed)
    n = cfg.num_accounts

    # Assign each account to a split.
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = int(n * cfg.train_frac)
    n_dev = int(n * cfg.dev_frac)
    train_idx = set(indices[:n_train])
    dev_idx = set(indices[n_train : n_train + n_dev])
    # Ghost accounts (no KG node) drawn from the non-train pool only.
    eval_pool = indices[n_train:]
    n_ghost = int(len(eval_pool) * cfg.ghost_frac)
    ghost_idx = set(eval_pool[:n_ghost])

    # Format-drift a fraction of KG node identifiers (digits preserved). When the
    # drift fraction is zero no randomness is consumed, so default corpora remain
    # byte-identical to the undrifted generator.
    node_ids: dict[int, str] = {}
    if cfg.qa_join_drift > 0:
        for idx in range(n):
            if idx in ghost_idx:
                continue
            if rng.random() < cfg.qa_join_drift:
                node_ids[idx] = f"account:{_drift_alpha(f'ACC{1000 + idx}', rng)}"

    def node_of(idx: int) -> str:
        return node_ids.get(idx, _node_id(idx))

    accounts: list[dict[str, str]] = []
    triples: list[dict[str, str]] = []
    account_facts: dict[int, dict[str, str]] = {}

    # Shared catalog edges (instrument -> asset_class -> regulator).
    catalog_edges: list[dict[str, str]] = []
    for instrument, asset_class in ASSET_CLASS.items():
        catalog_edges.append(
            {"subject": f"instrument:{instrument}", "relation": "asset_class", "object": asset_class}
        )
    for asset_class, regulator in REGULATOR.items():
        catalog_edges.append(
            {"subject": f"class:{asset_class}", "relation": "regulated_by", "object": regulator}
        )

    for idx in range(n):
        risk = rng.choice(RISK_RATINGS)
        region = rng.choice(REGIONS)
        sector = rng.choice(SECTORS)
        instrument = rng.choice(INSTRUMENTS)
        flag = rng.choice(COMPLIANCE_FLAGS)
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        balance = rng.randrange(20_000, 500_000, 1_000)

        accounts.append(
            {
                "account_id": _account_id(idx),
                "customer_id": f"CUS-{2000 + idx}",
                "customer_name": name,
                "risk_rating": risk,
                "region": region,
                "sector": sector,
                "account_code": _account_code(idx),
                "balance": str(balance),
            }
        )
        account_facts[idx] = {
            "instrument": instrument,
            "asset_class": ASSET_CLASS[instrument],
            "regulator": REGULATOR[ASSET_CLASS[instrument]],
            "flag": flag,
            "is_ghost": idx in ghost_idx,
        }

        if idx in ghost_idx:
            continue  # no KG node for ghost accounts
        node = node_of(idx)
        triples.append({"subject": node, "relation": "holds_instrument", "object": instrument})
        triples.append({"subject": node, "relation": "compliance_flag", "object": flag})
        triples.append({"subject": node, "relation": "region", "object": region})

    triples.extend(catalog_edges)

    questions: list[dict[str, object]] = []
    join_label_set: dict[tuple[str, str], bool] = {}

    def split_of(idx: int) -> str:
        if idx in train_idx:
            return "train"
        if idx in dev_idx:
            return "dev"
        return "test"

    def near_duplicates(idx: int) -> list[str]:
        # Distractors are real neighbouring KG nodes (different entity, hence a
        # false join) so that *accepting* a bad join resolves to a concrete wrong
        # answer -- otherwise a no-verifier planner is never punished for it.
        cands = []
        for delta in (1, -1, 10):
            j = idx + delta
            if 0 <= j < n and j != idx and j not in ghost_idx:
                cands.append(node_of(j))
        return cands

    # --- Verifier learning set: graded-corruption hard region (decoupled from QA). ---
    verifier_pairs, verifier_eval = make_verifier_dataset(cfg, rng)

    # --- Clean QA candidate labels (ground truth for QA-time confusion). ---
    def register_join_pair(lhs: str, rhs: str, label: bool) -> None:
        join_label_set[(lhs, rhs)] = label

    for idx in range(n):
        code = _account_code(idx)
        if idx not in ghost_idx:
            register_join_pair(code, node_of(idx), True)
        for dup in near_duplicates(idx):
            register_join_pair(code, dup, False)

    # --- Questions (dev + test only; train accounts reserved for the verifier). ---
    qid = 0
    for idx in range(n):
        split = split_of(idx)
        if split == "train":
            continue
        facts = account_facts[idx]
        acc_id = _account_id(idx)
        code = _account_code(idx)
        node = node_of(idx)
        dups = near_duplicates(idx)

        # 1) single-table lookup
        col = rng.choice(["risk_rating", "sector", "region", "customer_name"])
        questions.append(
            {
                "id": f"q{qid:05d}_single",
                "text": f"What is the {col.replace('_', ' ')} for account {acc_id}?",
                "answer": accounts[idx][col],
                "task": "single_table_lookup",
                "table": "accounts",
                "lookup_column": "account_id",
                "lookup_value": acc_id,
                "return_column": col,
                "hops": 0,
                "split": split,
            }
        )
        qid += 1

        if facts["is_ghost"]:
            # 2) abstain — no KG node exists for this account.
            candidate_order = dups[:2] if dups else []
            questions.append(
                {
                    "id": f"q{qid:05d}_abstain",
                    "text": f"Which instrument does account {acc_id} hold in the knowledge graph?",
                    "answer": "ABSTAIN",
                    "task": "abstain",
                    "table": "accounts",
                    "lookup_column": "account_id",
                    "lookup_value": acc_id,
                    "join_column": "account_code",
                    "traverse_relation": "holds_instrument",
                    "traverse_chain": ["holds_instrument"],
                    "candidate_order": candidate_order,
                    "hops": 1,
                    "split": split,
                    "should_abstain": True,
                }
            )
            qid += 1
            continue

        # 3) 1-hop table->KG join
        questions.append(
            {
                "id": f"q{qid:05d}_join",
                "text": f"Which instrument does account {acc_id} hold in the knowledge graph?",
                "answer": facts["instrument"],
                "task": "table_kg_join",
                "table": "accounts",
                "lookup_column": "account_id",
                "lookup_value": acc_id,
                "join_column": "account_code",
                "traverse_relation": "holds_instrument",
                "traverse_chain": ["holds_instrument"],
                "candidate_order": [node],
                "gold_join": {"lhs": code, "rhs": node},
                "hops": 1,
                "split": split,
            }
        )
        qid += 1

        # 4) bad-join backtrack (near-duplicate offered first)
        if dups:
            order = [dups[0], node]
            questions.append(
                {
                    "id": f"q{qid:05d}_backtrack",
                    "text": (
                        f"What compliance flag is attached to account {acc_id} "
                        "after rejecting the near-duplicate account?"
                    ),
                    "answer": facts["flag"],
                    "task": "bad_join_backtrack",
                    "table": "accounts",
                    "lookup_column": "account_id",
                    "lookup_value": acc_id,
                    "join_column": "account_code",
                    "traverse_relation": "compliance_flag",
                    "traverse_chain": ["compliance_flag"],
                    "candidate_order": order,
                    "gold_join": {"lhs": code, "rhs": node},
                    "hops": 1,
                    "split": split,
                }
            )
            qid += 1

        # 5) multi-hop traverse (2-hop and 3-hop)
        questions.append(
            {
                "id": f"q{qid:05d}_hop2",
                "text": f"What asset class is the instrument held by account {acc_id}?",
                "answer": facts["asset_class"],
                "task": "multi_hop_traverse",
                "table": "accounts",
                "lookup_column": "account_id",
                "lookup_value": acc_id,
                "join_column": "account_code",
                "traverse_chain": ["holds_instrument", "asset_class"],
                "candidate_order": [node],
                "gold_join": {"lhs": code, "rhs": node},
                "hops": 2,
                "split": split,
            }
        )
        qid += 1
        questions.append(
            {
                "id": f"q{qid:05d}_hop3",
                "text": f"Which regulator oversees the instrument held by account {acc_id}?",
                "answer": facts["regulator"],
                "task": "multi_hop_traverse",
                "table": "accounts",
                "lookup_column": "account_id",
                "lookup_value": acc_id,
                "join_column": "account_code",
                "traverse_chain": ["holds_instrument", "asset_class", "regulated_by"],
                "candidate_order": [node],
                "gold_join": {"lhs": code, "rhs": node},
                "hops": 3,
                "split": split,
            }
        )
        qid += 1

    join_labels = [
        {"lhs": lhs, "rhs": rhs, "label": label} for (lhs, rhs), label in join_label_set.items()
    ]

    config = {
        "tau_v": cfg.tau_v,
        "tau_u": cfg.tau_u,
        "top_k": cfg.top_k,
        "index_backend": "deterministic-local",
        "seed": cfg.seed,
        "num_accounts": cfg.num_accounts,
        "qa_join_drift": cfg.qa_join_drift,
    }
    meta = {
        "num_accounts": n,
        "train_accounts": len(train_idx),
        "dev_accounts": len(dev_idx),
        "ghost_accounts": len(ghost_idx),
        "drifted_nodes": len(node_ids),
        "num_questions": len(questions),
        "num_verifier_train": len(verifier_pairs),
        "num_verifier_eval": len(verifier_eval),
        "num_join_labels": len(join_labels),
    }
    return GeneratedCorpus(
        accounts=accounts,
        triples=triples,
        questions=questions,
        join_labels=join_labels,
        verifier_pairs=verifier_pairs,
        verifier_eval=verifier_eval,
        config=config,
        meta=meta,
    )


def write_corpus(corpus: GeneratedCorpus, out_dir: str | Path) -> dict[str, object]:
    root = Path(out_dir)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "kg").mkdir(parents=True, exist_ok=True)

    fieldnames = list(corpus.accounts[0].keys())
    with (root / "tables" / "accounts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corpus.accounts)

    _dump(root / "kg" / "triples.json", corpus.triples)
    _dump(root / "questions.json", corpus.questions)
    _dump(root / "join_labels.json", corpus.join_labels)
    _dump(root / "verifier_pairs.json", corpus.verifier_pairs)
    _dump(root / "verifier_eval.json", corpus.verifier_eval)
    _dump(root / "config.json", corpus.config)
    _dump(root / "meta.json", corpus.meta)
    return corpus.meta


def _load_pairs(path: Path) -> list[tuple[str, str, bool]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        items = json.load(handle)
    return [(item["lhs"], item["rhs"], bool(item["label"])) for item in items]


def load_verifier_pairs(fixtures_dir: str | Path) -> list[tuple[str, str, bool]]:
    return _load_pairs(Path(fixtures_dir) / "verifier_pairs.json")


def load_verifier_eval(fixtures_dir: str | Path) -> list[tuple[str, str, bool]]:
    return _load_pairs(Path(fixtures_dir) / "verifier_eval.json")


def _dump(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
