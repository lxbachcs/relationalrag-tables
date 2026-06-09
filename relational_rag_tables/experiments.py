"""Reproducible experiment matrix for the synthetic controlled study.

Every number reported in the paper's evaluation section is regenerated here from
the seeded synthetic corpus. There is no external data, GPU, or network access:
running ``python3 -m relational_rag_tables.cli experiments`` reproduces the tables
end to end on a laptop CPU in well under a minute.

The matrix covers:

* **QA by verifier** — exact-match baseline, single learned verifier, deep
  ensemble, and the oracle (deterministic) verifier, plus a no-verifier ablation,
  each scored as exact-match accuracy per task family.
* **Held-out verifier discrimination** — accuracy / F1 / AUROC / Brier on the
  graded-corruption evaluation split that no verifier has seen.
* **Deep-ensemble selective prediction** — risk-coverage curve and the
  correct-vs-wrong uncertainty gap.
* **Ensemble-size sweep** — how discrimination, selective prediction, and QA move
  with ``K`` (members are nested, so smaller ``K`` is a prefix of the largest).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import datagen
from .benchmark import _metrics_by_task, evaluate_verifier, selective_prediction
from .evaluate import evaluate_results
from .index import build_entries
from .ingest import load_corpus
from .plan_execute import PlannerConfig, run_all
from .uncertainty import EnsembleVerifier
from .verify import DeterministicJoinVerifier, ExactMatchVerifier


def _qa(corpus, entries, *, verifier, tau_v, top_k, uncertainty=None, use_verifier=True, tau_u=None):
    cfg = PlannerConfig(
        tau_v=tau_v,
        tau_u=tau_u,
        top_k=top_k,
        use_verifier=use_verifier,
        use_uncertainty=tau_u is not None and uncertainty is not None,
    )
    results = run_all(
        corpus,
        entries,
        verifier=verifier,
        cfg=cfg,
        uncertainty=uncertainty if cfg.use_uncertainty else None,
    )
    metrics = evaluate_results(corpus, results)
    return {
        "em": round(metrics["em"], 4),
        "abstentions": metrics["abstentions"],
        "verifier_confusion": metrics["verifier_confusion"],
        "latency_ms_avg": metrics["latency_ms"]["avg"],
        "by_task": _metrics_by_task(corpus, results),
    }


def run_matrix(fixtures_dir, *, ensemble_k: int = 5, k_sweep=(1, 3, 5), seed: int = 0) -> dict:
    corpus = load_corpus(fixtures_dir)
    entries = build_entries(corpus)
    tau_v = float(corpus.config["tau_v"])
    top_k = int(corpus.config["top_k"])
    train_pairs = datagen.load_verifier_pairs(fixtures_dir)
    eval_pairs = datagen.load_verifier_eval(fixtures_dir)

    # Train the largest ensemble once; nested prefixes give every smaller K with
    # identical members, so the sweep is apples-to-apples.
    max_k = max(max(k_sweep), ensemble_k)
    full = EnsembleVerifier.train(train_pairs, k=max_k, seed=seed)
    ensemble = EnsembleVerifier(members=full.members[:ensemble_k])
    learned = full.members[0]
    exact = ExactMatchVerifier()
    oracle = DeterministicJoinVerifier(corpus.join_labels)

    qa_by_verifier = {
        "exact_match": _qa(corpus, entries, verifier=exact, tau_v=tau_v, top_k=top_k),
        "learned": _qa(corpus, entries, verifier=learned, tau_v=tau_v, top_k=top_k),
        "ensemble": _qa(corpus, entries, verifier=ensemble, tau_v=tau_v, top_k=top_k),
        "oracle": _qa(corpus, entries, verifier=oracle, tau_v=tau_v, top_k=top_k),
        "no_verifier": _qa(
            corpus, entries, verifier=oracle, tau_v=tau_v, top_k=top_k, use_verifier=False
        ),
    }

    discrimination = {
        "exact_match": evaluate_verifier(exact, eval_pairs, tau_v=tau_v),
        "learned": evaluate_verifier(learned, eval_pairs, tau_v=tau_v),
        "ensemble": evaluate_verifier(ensemble, eval_pairs, tau_v=tau_v),
    }

    selective = selective_prediction(ensemble, eval_pairs, tau_v=tau_v)

    sweep = []
    for k in k_sweep:
        member_subset = EnsembleVerifier(members=full.members[:k])
        sweep.append(
            {
                "k": k,
                "auroc": evaluate_verifier(member_subset, eval_pairs, tau_v=tau_v)["auroc"],
                "qa_em": _qa(corpus, entries, verifier=member_subset, tau_v=tau_v, top_k=top_k)["em"],
                "selective": selective_prediction(member_subset, eval_pairs, tau_v=tau_v),
            }
        )

    return {
        "corpus": dict(corpus.config),
        "meta": {
            "num_questions": len(corpus.questions),
            "num_verifier_train": len(train_pairs),
            "num_verifier_eval": len(eval_pairs),
            "tau_v": tau_v,
            "ensemble_k": ensemble_k,
            "seed": seed,
        },
        "qa_by_verifier": qa_by_verifier,
        "verifier_discrimination": discrimination,
        "selective_prediction": selective,
        "ensemble_size_sweep": sweep,
    }


def run(fixtures_dir, out_dir, *, ensemble_k: int = 5, seed: int = 0) -> dict:
    results = run_matrix(fixtures_dir, ensemble_k=ensemble_k, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results
