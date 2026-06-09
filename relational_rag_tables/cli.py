"""Command line entrypoint for the reference artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate_results
from .index import load_index, write_index
from .ingest import load_corpus
from .plan_execute import result_to_json, run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="relational-rag-tables")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-fixtures", help="Validate fixture data")
    prepare.add_argument("--fixtures", default="fixtures/smoke")

    build = subparsers.add_parser("build-index", help="Build deterministic local indexes")
    build.add_argument("--fixtures", default="fixtures/smoke")
    build.add_argument("--out", default="artifacts/indexes")

    run = subparsers.add_parser("run-qa", help="Run QA over a built index")
    run.add_argument("--fixtures", default="fixtures/smoke")
    run.add_argument("--indexes", default="artifacts/indexes")
    run.add_argument("--out", default="artifacts/run")

    evaluate = subparsers.add_parser("evaluate", help="Run QA and write metrics")
    evaluate.add_argument("--fixtures", default="fixtures/smoke")
    evaluate.add_argument("--indexes", default="artifacts/indexes")
    evaluate.add_argument("--out", default="artifacts/eval")

    smoke = subparsers.add_parser("smoke", help="Run the full reference artifact pipeline")
    smoke.add_argument("--fixtures", default="fixtures/smoke")
    smoke.add_argument("--out", default="artifacts/smoke")

    synthesize = subparsers.add_parser(
        "synthesize", help="Generate the synthetic tables+KG benchmark corpus"
    )
    synthesize.add_argument("--out", default="fixtures/synthetic")
    synthesize.add_argument("--num-accounts", type=int, default=600)
    synthesize.add_argument("--seed", type=int, default=7)
    synthesize.add_argument(
        "--qa-join-drift",
        type=float,
        default=0.0,
        help="Fraction of KG node ids format-drifted from the table key (digits preserved)",
    )

    benchmark = subparsers.add_parser(
        "benchmark", help="Run the full pipeline over a synthetic corpus and report metrics"
    )
    benchmark.add_argument("--fixtures", default="fixtures/synthetic")
    benchmark.add_argument("--out", default="artifacts/benchmark")
    benchmark.add_argument(
        "--verifier",
        choices=["deterministic", "exact_match", "learned", "ensemble"],
        default="deterministic",
        help="Join verifier: deterministic/exact_match need no training; learned/ensemble require scikit-learn",
    )
    benchmark.add_argument("--ensemble-k", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument(
        "--uncertainty-gate",
        action="store_true",
        help="Abstain on high-uncertainty joins (ensemble verifier only)",
    )
    benchmark.add_argument(
        "--tau-u", type=float, default=None, help="Override the uncertainty abstention threshold"
    )

    experiments = subparsers.add_parser(
        "experiments", help="Regenerate the full paper experiment matrix from a synthetic corpus"
    )
    experiments.add_argument("--fixtures", default="fixtures/synthetic")
    experiments.add_argument("--out", default="artifacts/experiments")
    experiments.add_argument("--ensemble-k", type=int, default=5)
    experiments.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.command == "prepare-fixtures":
        corpus = load_corpus(args.fixtures)
        print(f"validated {len(corpus.questions)} questions from {args.fixtures}")
        return 0

    if args.command == "build-index":
        corpus = load_corpus(args.fixtures)
        entries = write_index(corpus, args.out)
        print(f"wrote {len(entries)} index entries to {args.out}")
        return 0

    if args.command == "run-qa":
        corpus = load_corpus(args.fixtures)
        entries = load_index(args.indexes)
        results = run_all(corpus, entries)
        _write_results(Path(args.out), corpus, results, metrics=False)
        print(f"wrote QA traces to {args.out}")
        return 0

    if args.command == "evaluate":
        corpus = load_corpus(args.fixtures)
        entries = load_index(args.indexes)
        results = run_all(corpus, entries)
        _write_results(Path(args.out), corpus, results, metrics=True)
        print(f"wrote metrics to {Path(args.out) / 'metrics.json'}")
        return 0

    if args.command == "smoke":
        corpus = load_corpus(args.fixtures)
        out = Path(args.out)
        index_dir = out / "indexes"
        write_index(corpus, index_dir)
        entries = load_index(index_dir)
        results = run_all(corpus, entries)
        _write_results(out, corpus, results, metrics=True)
        metrics = evaluate_results(corpus, results)
        if metrics["em"] != 1.0:
            print(json.dumps(metrics, indent=2))
            return 1
        print(f"smoke passed: EM={metrics['em']:.1f}, traces={out / 'traces.json'}")
        return 0

    if args.command == "synthesize":
        from .benchmark import synthesize_corpus

        meta = synthesize_corpus(
            args.out,
            num_accounts=args.num_accounts,
            seed=args.seed,
            qa_join_drift=args.qa_join_drift,
        )
        print(f"wrote synthetic corpus to {args.out}: {json.dumps(meta)}")
        return 0

    if args.command == "benchmark":
        from .benchmark import run_benchmark

        metrics = run_benchmark(
            args.fixtures,
            args.out,
            verifier=args.verifier,
            ensemble_k=args.ensemble_k,
            seed=args.seed,
            uncertainty_gate=args.uncertainty_gate,
            tau_u=args.tau_u,
        )
        qa = metrics["qa"]
        print(
            f"benchmark[{args.verifier}]: EM={qa['em']:.3f} "
            f"abstain={qa['abstentions']} over {qa['num_questions']} questions "
            f"-> {Path(args.out) / 'metrics.json'}"
        )
        verifier_eval = metrics.get("verifier_eval")
        if verifier_eval:
            print(
                f"  held-out verifier: acc={verifier_eval['accuracy']:.3f} "
                f"f1={verifier_eval['f1']:.3f} auroc={verifier_eval['auroc']}"
            )
        return 0

    if args.command == "experiments":
        from .experiments import run

        results = run(args.fixtures, args.out, ensemble_k=args.ensemble_k, seed=args.seed)
        qa = results["qa_by_verifier"]
        disc = results["verifier_discrimination"]
        print(f"experiments -> {Path(args.out) / 'results.json'}")
        print(
            "  QA EM: "
            f"exact_match={qa['exact_match']['em']:.3f} "
            f"learned={qa['learned']['em']:.3f} "
            f"ensemble={qa['ensemble']['em']:.3f} "
            f"oracle={qa['oracle']['em']:.3f} "
            f"no_verifier={qa['no_verifier']['em']:.3f}"
        )
        print(
            "  held-out AUROC: "
            f"exact_match={disc['exact_match']['auroc']} "
            f"learned={disc['learned']['auroc']} "
            f"ensemble={disc['ensemble']['auroc']}"
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _write_results(out: Path, corpus: object, results: object, metrics: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "traces.json").open("w", encoding="utf-8") as handle:
        json.dump([result_to_json(result) for result in results], handle, indent=2)
    if metrics:
        with (out / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(evaluate_results(corpus, results), handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
