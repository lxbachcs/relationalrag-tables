"""Execution-aware planner and executor.

The planner emits a typed operation trace (``select``, ``join``, ``traverse``,
``return``, ``abstain``), consults the join verifier before committing any join,
backtracks past rejected candidates, follows multi-hop traversal chains, and can
abstain when the deep-ensemble uncertainty head is unconfident. It is configured
through :class:`PlannerConfig`, which also expresses the ablations and the
retrieval-only baseline used in the evaluation.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol

from .index import normalize_answer
from .retrieve import retrieve, retrieved_values
from .schemas import Corpus, IndexEntry, Operation, Question, RunResult, TraceEvent
from .verify import JoinVerifier


class UncertaintyHead(Protocol):
    def uncertainty(self, lhs_value: str, rhs_id: str) -> float: ...


@dataclass
class PlannerConfig:
    tau_v: float = 0.5
    tau_u: float | None = None
    top_k: int = 8
    use_verifier: bool = True
    use_uncertainty: bool = False
    mode: str = "planner"  # "planner" | "retrieval_only"
    label: str = "full"


def _chain_for(question: Question) -> list[str]:
    if question.traverse_chain:
        return list(question.traverse_chain)
    if question.traverse_relation:
        return [question.traverse_relation]
    return []


def build_triple_index(corpus: Corpus) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for triple in corpus.triples:
        index.setdefault((triple.subject, triple.relation), triple.object)
    return index


def _resolve(index: dict[tuple[str, str], str], value: str, relation: str) -> str | None:
    for subject in (value, f"instrument:{value}", f"class:{value}"):
        if (subject, relation) in index:
            return index[(subject, relation)]
    return None


def _traverse_chain(index: dict[tuple[str, str], str], start: str, chain: list[str]) -> str | None:
    current = start
    for relation in chain:
        nxt = _resolve(index, current, relation)
        if nxt is None:
            return None
        current = nxt
    return current


def run_question(
    corpus: Corpus,
    entries: list[IndexEntry],
    question: Question,
    verifier: JoinVerifier,
    cfg: PlannerConfig,
    triple_index: dict[tuple[str, str], str] | None = None,
    uncertainty: Optional[UncertaintyHead] = None,
    exclude: set[str] | None = None,
) -> RunResult:
    started = time.perf_counter()
    retrieval_started = time.perf_counter()
    retrieved = retrieve(entries, question.text, cfg.top_k)
    retrieval_ms = _elapsed_ms(retrieval_started)
    index = triple_index if triple_index is not None else build_triple_index(corpus)

    execution_started = time.perf_counter()
    trace: list[TraceEvent] = []

    if cfg.mode == "retrieval_only":
        prediction, status = _execute_retrieval_only(question, retrieved, trace)
    elif question.task == "single_table_lookup":
        prediction = _execute_single_table_lookup(corpus, question, trace)
        status = "completed"
    elif question.task in {"table_kg_join", "bad_join_backtrack", "multi_hop_traverse", "abstain"}:
        prediction, status = _execute_join(
            corpus, question, retrieved.edges, verifier, cfg, index, uncertainty, trace, exclude
        )
    else:
        trace.append(TraceEvent(Operation("abstain", {"reason": f"unknown task {question.task}"}), output="ABSTAIN"))
        prediction, status = "ABSTAIN", "abstained"

    execution_ms = _elapsed_ms(execution_started)
    return RunResult(
        question_id=question.id,
        question=question.text,
        prediction=prediction,
        gold=question.answer,
        status=status,
        trace=trace,
        retrieved_values=retrieved_values(retrieved),
        latency_ms={
            "retrieval": retrieval_ms,
            "execution": execution_ms,
            "total": _elapsed_ms(started),
        },
    )


def run_all(
    corpus: Corpus,
    entries: list[IndexEntry],
    verifier: JoinVerifier | None = None,
    cfg: PlannerConfig | None = None,
    uncertainty: Optional[UncertaintyHead] = None,
) -> list[RunResult]:
    """Run the planner over every question.

    With no explicit verifier/config this reproduces the dependency-free smoke
    path (deterministic verifier built from the fixture's join labels).
    """
    if cfg is None:
        tau_u = corpus.config.get("tau_u")
        cfg = PlannerConfig(
            tau_v=float(corpus.config["tau_v"]),
            tau_u=tau_u if isinstance(tau_u, (int, float)) and tau_u >= 0 else None,
            top_k=int(corpus.config["top_k"]),
            use_uncertainty=uncertainty is not None and isinstance(tau_u, (int, float)) and tau_u >= 0,
        )
    if verifier is None:
        from .verify import DeterministicJoinVerifier

        verifier = DeterministicJoinVerifier(corpus.join_labels)

    index = build_triple_index(corpus)
    return [
        run_question(corpus, entries, question, verifier, cfg, index, uncertainty)
        for question in corpus.questions
    ]


def result_to_json(result: RunResult) -> dict[str, Any]:
    return {
        "question_id": result.question_id,
        "question": result.question,
        "prediction": result.prediction,
        "gold": result.gold,
        "status": result.status,
        "trace": [
            {
                "operation": asdict(event.operation),
                "output": event.output,
                "verifier_score": event.verifier_score,
                "accepted": event.accepted,
            }
            for event in result.trace
        ],
        "latency_ms": result.latency_ms,
    }


def _execute_single_table_lookup(corpus: Corpus, question: Question, trace: list[TraceEvent]) -> str:
    table = corpus.tables[question.table]
    trace.append(
        TraceEvent(
            Operation(
                "select",
                {"table": question.table, "where": {question.lookup_column: question.lookup_value}},
            )
        )
    )
    for row in table.rows:
        if row.get(question.lookup_column or "") == question.lookup_value:
            value = row.get(question.return_column or "", "ABSTAIN")
            trace[-1].output = row
            trace.append(TraceEvent(Operation("return", {"column": question.return_column}), output=value))
            return value
    trace.append(TraceEvent(Operation("abstain", {"reason": "lookup_not_found"}), output="ABSTAIN"))
    return "ABSTAIN"


def _execute_join(
    corpus: Corpus,
    question: Question,
    retrieved_edges: list[IndexEntry],
    verifier: JoinVerifier,
    cfg: PlannerConfig,
    index: dict[tuple[str, str], str],
    uncertainty: Optional[UncertaintyHead],
    trace: list[TraceEvent],
    exclude: set[str] | None = None,
) -> tuple[str, str]:
    table = corpus.tables[question.table]
    row = next(
        (r for r in table.rows if r.get(question.lookup_column or "") == question.lookup_value),
        None,
    )
    trace.append(
        TraceEvent(
            Operation(
                "select",
                {"table": question.table, "where": {question.lookup_column: question.lookup_value}},
            ),
            output=row,
        )
    )
    if row is None:
        trace.append(TraceEvent(Operation("abstain", {"reason": "lookup_not_found"}), output="ABSTAIN"))
        return "ABSTAIN", "abstained"

    lhs_value = row.get(question.join_column or "", "")
    chain = _chain_for(question)
    candidate_subjects = _candidate_subjects(corpus, question, retrieved_edges, exclude)

    for rhs_id in candidate_subjects:
        if cfg.use_verifier:
            score = verifier.score(lhs_value, rhs_id)
            accepted = score >= cfg.tau_v
        else:
            score, accepted = 1.0, True  # no-verifier ablation accepts first candidate

        event = TraceEvent(
            Operation(
                "join",
                {
                    "lhs": f"{question.table}.{question.join_column}",
                    "lhs_value": lhs_value,
                    "rhs": rhs_id,
                },
            ),
            verifier_score=score,
            accepted=accepted,
        )
        trace.append(event)
        if not accepted:
            continue

        # Uncertainty-gated abstention on the committed join.
        if cfg.use_uncertainty and uncertainty is not None and cfg.tau_u is not None:
            u = uncertainty.uncertainty(lhs_value, rhs_id)
            event.operation.args["uncertainty"] = round(u, 6)
            if u > cfg.tau_u:
                trace.append(
                    TraceEvent(Operation("abstain", {"reason": "uncertainty", "u": round(u, 6)}), output="ABSTAIN")
                )
                return "ABSTAIN", "abstained"

        value = _traverse_chain(index, rhs_id, chain)
        trace.append(
            TraceEvent(Operation("traverse", {"subject": rhs_id, "chain": chain}), output=value)
        )
        if value is None:
            continue
        trace.append(TraceEvent(Operation("return", {"source": "kg"}), output=value))
        return value, "completed"

    trace.append(TraceEvent(Operation("abstain", {"reason": "no_verified_join"}), output="ABSTAIN"))
    return "ABSTAIN", "abstained"


def _execute_retrieval_only(
    question: Question, retrieved: object, trace: list[TraceEvent]
) -> tuple[str, str]:
    """Retrieval-only RAG baseline: no relational execution, never abstains.

    Returns the most salient retrieved value: for table lookups, the top
    retrieved cell of the requested column; otherwise the object of the top
    retrieved edge. Cannot follow multi-hop chains and cannot refuse to answer.
    """
    if question.task == "single_table_lookup":
        for cell in retrieved.cells:  # type: ignore[attr-defined]
            if cell.payload.get("column") == question.return_column:
                value = str(cell.payload.get("value", ""))
                trace.append(TraceEvent(Operation("return", {"source": "retrieval"}), output=value))
                return value, "completed"
    for edge in retrieved.edges:  # type: ignore[attr-defined]
        value = str(edge.payload.get("object", ""))
        if value:
            trace.append(TraceEvent(Operation("return", {"source": "retrieval"}), output=value))
            return value, "completed"
    trace.append(TraceEvent(Operation("abstain", {"reason": "empty_retrieval"}), output="ABSTAIN"))
    return "ABSTAIN", "abstained"


def _candidate_subjects(
    corpus: Corpus,
    question: Question,
    retrieved_edges: list[IndexEntry],
    exclude: set[str] | None = None,
) -> list[str]:
    exclude = exclude or set()
    subjects: list[str] = []
    for candidate in question.candidate_order:
        if candidate not in subjects and candidate not in exclude:
            subjects.append(candidate)
    for edge in retrieved_edges:
        subject = str(edge.payload.get("subject", ""))
        if subject and subject not in subjects and subject not in exclude:
            subjects.append(subject)
    return subjects


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)
