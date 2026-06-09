"""Reference-scale benchmark over the synthetic tables+KG corpus.

This wires the synthetic generator (:mod:`datagen`) into the full plan-execute
pipeline. It is the large-scale analogue of the ``smoke`` command: synthesise a
heterogeneous ``accounts`` table joined against a financial-instruments KG,
optionally train the learned join verifier / deep ensemble on the generator's
held-out join-pair split, run the planner over the question set, and report QA
metrics, a per-task breakdown, held-out verifier discrimination, and (for the
ensemble) a deep-ensemble selective-prediction curve.

The default ``deterministic`` verifier reproduces the dependency-free pipeline on
the larger corpus; ``learned`` and ``ensemble`` exercise :mod:`features`,
:mod:`verify`, and :mod:`uncertainty` and require scikit-learn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import datagen
from .datagen import GenConfig
from .evaluate import evaluate_results
from .index import write_index
from .ingest import load_corpus
from .plan_execute import PlannerConfig, exact_match, result_to_json, run_all
from .schemas import Corpus, RunResult
from .verify import DeterministicJoinVerifier, ExactMatchVerifier, JoinVerifier, LearnedJoinVerifier

VERIFIER_KINDS = ("deterministic", "exact_match", "learned", "ensemble")


def synthesize_corpus(
    out_dir: str | Path,
    *,
    num_accounts: int = 600,
    seed: int = 7,
    qa_join_drift: float = 0.0,
) -> dict[str, object]:
    """Generate and persist the synthetic benchmark corpus, returning its meta."""
    corpus = datagen.generate(
        GenConfig(num_accounts=num_accounts, seed=seed, qa_join_drift=qa_join_drift)
    )
    return datagen.write_corpus(corpus, out_dir)


def _build_verifier(
    kind: str, corpus: Corpus, fixtures_dir: str | Path, *, ensemble_k: int, seed: int
) -> tuple[JoinVerifier, Any]:
    """Return ``(verifier, uncertainty_head)``; the head is ``None`` unless ensemble."""
    if kind == "deterministic":
        return DeterministicJoinVerifier(corpus.join_labels), None
    if kind == "exact_match":
        return ExactMatchVerifier(), None

    pairs = datagen.load_verifier_pairs(fixtures_dir)
    if not pairs:
        raise ValueError(
            f"verifier_pairs.json not found in {fixtures_dir}; "
            "(re)synthesize the corpus before training a learned verifier"
        )
    if kind == "learned":
        return LearnedJoinVerifier.train(pairs, seed=seed), None
    if kind == "ensemble":
        from .uncertainty import EnsembleVerifier

        ensemble = EnsembleVerifier.train(pairs, k=ensemble_k, seed=seed)
        return ensemble, ensemble
    raise ValueError(f"unknown verifier kind: {kind!r} (expected one of {VERIFIER_KINDS})")


def run_benchmark(
    fixtures_dir: str | Path,
    out_dir: str | Path,
    *,
    verifier: str = "deterministic",
    ensemble_k: int = 5,
    seed: int = 0,
    uncertainty_gate: bool = False,
    tau_u: float | None = None,
    write_artifacts: bool = True,
) -> dict[str, object]:
    """Run the full pipeline over a synthesised corpus and return metrics."""
    corpus = load_corpus(fixtures_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = write_index(corpus, out / "indexes")

    verifier_obj, uncertainty_head = _build_verifier(
        verifier, corpus, fixtures_dir, ensemble_k=ensemble_k, seed=seed
    )

    config_tau_u = tau_u if tau_u is not None else corpus.config.get("tau_u")
    use_uncertainty = uncertainty_gate and uncertainty_head is not None
    cfg = PlannerConfig(
        tau_v=float(corpus.config["tau_v"]),
        tau_u=float(config_tau_u) if isinstance(config_tau_u, (int, float)) and config_tau_u >= 0 else None,
        top_k=int(corpus.config["top_k"]),
        use_uncertainty=use_uncertainty,
        label=f"verifier={verifier}",
    )
    results = run_all(
        corpus,
        entries,
        verifier=verifier_obj,
        cfg=cfg,
        uncertainty=uncertainty_head if use_uncertainty else None,
    )

    qa = evaluate_results(corpus, results)
    qa_summary = {key: value for key, value in qa.items() if key != "per_question"}

    metrics: dict[str, object] = {
        "verifier": verifier,
        "config": {
            "tau_v": cfg.tau_v,
            "tau_u": cfg.tau_u,
            "top_k": cfg.top_k,
            "uncertainty_gate": use_uncertainty,
            "ensemble_k": ensemble_k if verifier == "ensemble" else None,
            "num_accounts": corpus.config.get("num_accounts"),
            "seed": seed,
        },
        "qa": qa_summary,
        "by_task": _metrics_by_task(corpus, results),
    }

    eval_pairs = datagen.load_verifier_eval(fixtures_dir)
    if eval_pairs:
        metrics["verifier_eval"] = evaluate_verifier(verifier_obj, eval_pairs, tau_v=cfg.tau_v)
        if uncertainty_head is not None:
            metrics["selective_prediction"] = selective_prediction(
                uncertainty_head, eval_pairs, tau_v=cfg.tau_v
            )

    if write_artifacts:
        _dump(out / "traces.json", [result_to_json(result) for result in results])
        _dump(out / "metrics.json", metrics)
        _dump(out / "per_question.json", qa["per_question"])
    return metrics


def _metrics_by_task(corpus: Corpus, results: list[RunResult]) -> dict[str, object]:
    task_of = {question.id: question.task for question in corpus.questions}
    buckets: dict[str, dict[str, int]] = {}
    for result in results:
        task = task_of.get(result.question_id, "unknown")
        bucket = buckets.setdefault(task, {"total": 0, "correct": 0, "abstained": 0})
        bucket["total"] += 1
        bucket["correct"] += int(exact_match(result.prediction, result.gold))
        bucket["abstained"] += int(result.status == "abstained")
    return {
        task: {
            "total": stats["total"],
            "em": round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0,
            "abstained": stats["abstained"],
        }
        for task, stats in sorted(buckets.items())
    }


def evaluate_verifier(
    verifier: JoinVerifier, pairs: list[tuple[str, str, bool]], *, tau_v: float
) -> dict[str, object]:
    """Confusion-based discrimination metrics for a verifier on labelled pairs."""
    scores = [verifier.score(lhs, rhs) for lhs, rhs, _ in pairs]
    labels = [1 if is_match else 0 for _, _, is_match in pairs]

    tp = fp = tn = fn = 0
    brier = 0.0
    for score, label in zip(scores, labels, strict=True):
        predicted = 1 if score >= tau_v else 0
        if label and predicted:
            tp += 1
        elif not label and predicted:
            fp += 1
        elif not label and not predicted:
            tn += 1
        else:
            fn += 1
        brier += (score - label) ** 2

    n = len(pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n,
        "positives": sum(labels),
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "brier": round(brier / n, 4) if n else 0.0,
        "auroc": _auroc(scores, labels),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def selective_prediction(
    head: Any,
    pairs: list[tuple[str, str, bool]],
    *,
    tau_v: float,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.5),
) -> dict[str, object]:
    """Deep-ensemble risk-coverage curve over held-out pairs.

    Pairs are ranked by the ensemble's uncertainty (most confident first); at each
    coverage we abstain on the most-uncertain tail and measure accuracy on what
    remains. A rising accuracy as coverage drops is the selective-prediction
    signal the deep ensemble is meant to provide.
    """
    items = []
    for lhs, rhs, is_match in pairs:
        score = head.score(lhs, rhs)
        predicted = 1 if score >= tau_v else 0
        correct = predicted == (1 if is_match else 0)
        items.append((head.uncertainty(lhs, rhs), correct))
    items.sort(key=lambda item: item[0])  # ascending uncertainty = most confident first

    n = len(items)
    curve = []
    for coverage in coverages:
        keep = max(1, int(round(n * coverage)))
        kept = items[:keep]
        accuracy = sum(1 for _, correct in kept if correct) / keep
        curve.append({"coverage": round(keep / n, 4), "accuracy": round(accuracy, 4)})

    u_correct = [u for u, correct in items if correct]
    u_wrong = [u for u, correct in items if not correct]
    return {
        "risk_coverage": curve,
        "mean_uncertainty_correct": round(_mean(u_correct), 6),
        "mean_uncertainty_wrong": round(_mean(u_wrong), 6),
    }


def _auroc(scores: list[float], labels: list[int]) -> float | None:
    """Rank-based area under the ROC curve (ties broken by average rank)."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        average_rank = (i + j - 1) / 2.0 + 1.0  # 1-based average rank for the tie group
        for k in range(i, j):
            ranks[order[k]] = average_rank
        i = j

    sum_positive_ranks = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    auc = (sum_positive_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)
    return round(auc, 4)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dump(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
