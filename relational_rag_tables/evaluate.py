"""Reference-scale evaluation metrics."""

from __future__ import annotations

from .index import normalize_answer
from .plan_execute import exact_match
from .schemas import Corpus, RunResult
from .verify import label_for


def evaluate_results(corpus: Corpus, results: list[RunResult]) -> dict[str, object]:
    total = len(results)
    correct = sum(exact_match(result.prediction, result.gold) for result in results)
    abstentions = sum(1 for result in results if result.status == "abstained")
    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "unknown": 0}

    for result in results:
        for event in result.trace:
            if event.operation.type != "join":
                continue
            lhs_value = str(event.operation.args["lhs_value"])
            rhs_id = str(event.operation.args["rhs"])
            label = label_for(corpus.join_labels, lhs_value, rhs_id)
            if label is None or event.accepted is None:
                confusion["unknown"] += 1
            elif label and event.accepted:
                confusion["tp"] += 1
            elif not label and event.accepted:
                confusion["fp"] += 1
            elif not label and not event.accepted:
                confusion["tn"] += 1
            elif label and not event.accepted:
                confusion["fn"] += 1

    per_question = [
        {
            "id": result.question_id,
            "prediction": result.prediction,
            "gold": result.gold,
            "exact_match": exact_match(result.prediction, result.gold),
            "execution_success": exact_match(result.prediction, result.gold),
            "faithful": _is_faithful(result),
            "status": result.status,
        }
        for result in results
    ]

    return {
        "num_questions": total,
        "correct": correct,
        "em": correct / total if total else 0.0,
        "execution_success": correct / total if total else 0.0,
        "abstentions": abstentions,
        "verifier_confusion": confusion,
        "latency_ms": {
            "total": round(sum(result.latency_ms["total"] for result in results), 3),
            "avg": round(sum(result.latency_ms["total"] for result in results) / total, 3)
            if total
            else 0.0,
        },
        "per_question": per_question,
    }


def _is_faithful(result: RunResult) -> bool:
    answer = normalize_answer(result.prediction)
    if answer == "abstain":
        return True
    evidence = {normalize_answer(value) for value in result.retrieved_values}
    if answer in evidence:
        return True
    # ...or the deterministic result of a traversal/lookup recorded in the trace.
    for event in result.trace:
        if event.operation.type in {"traverse", "select"} and event.output is not None:
            if normalize_answer(str(event.output)) == answer:
                return True
    return False
