from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from relational_rag_tables import datagen
from relational_rag_tables.benchmark import run_benchmark, synthesize_corpus
from relational_rag_tables.datagen import GenConfig
from relational_rag_tables.ingest import load_corpus


ROOT = Path(__file__).resolve().parents[1]
SMALL = GenConfig(num_accounts=50, seed=9, n_verifier_train=200, n_verifier_eval=120)


class BenchmarkTest(unittest.TestCase):
    def test_synthesize_writes_a_loadable_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = synthesize_corpus(tmp, num_accounts=50, seed=9)
            corpus = load_corpus(tmp)
            self.assertEqual(len(corpus.questions), meta["num_questions"])
            self.assertTrue((Path(tmp) / "verifier_pairs.json").exists())

    def test_deterministic_benchmark_is_exact_with_per_task_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            datagen.write_corpus(datagen.generate(SMALL), tmp)
            out = Path(tmp) / "out"
            metrics = run_benchmark(tmp, out, verifier="deterministic")
            self.assertEqual(metrics["qa"]["em"], 1.0)
            self.assertIn("single_table_lookup", metrics["by_task"])
            self.assertIn("verifier_eval", metrics)
            self.assertEqual(metrics["verifier_eval"]["n"], SMALL.n_verifier_eval)
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "traces.json").exists())

    def test_ensemble_benchmark_reports_selective_prediction(self) -> None:
        # A larger join-pair split gives the ensemble a stable operating point so
        # the selective-prediction signal is not an artefact of a tiny sample.
        config = GenConfig(num_accounts=80, seed=0, n_verifier_train=600, n_verifier_eval=300)
        with tempfile.TemporaryDirectory() as tmp:
            datagen.write_corpus(datagen.generate(config), tmp)
            metrics = run_benchmark(
                tmp, Path(tmp) / "out", verifier="ensemble", ensemble_k=3, seed=0
            )
        self.assertGreaterEqual(metrics["qa"]["em"], 0.0)
        self.assertLessEqual(metrics["qa"]["em"], 1.0)
        curve = metrics["selective_prediction"]["risk_coverage"]
        self.assertEqual(curve[0]["coverage"], 1.0)
        # Abstaining on the most-uncertain tail should not lower accuracy.
        self.assertGreaterEqual(curve[-1]["accuracy"], curve[0]["accuracy"])
        # Wrong predictions should, on average, carry higher ensemble uncertainty.
        self.assertGreater(
            metrics["selective_prediction"]["mean_uncertainty_wrong"],
            metrics["selective_prediction"]["mean_uncertainty_correct"],
        )

    def test_cli_synthesize_and_benchmark_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixtures = Path(tmp) / "synth"
            synth = subprocess.run(
                [sys.executable, "-m", "relational_rag_tables.cli", "synthesize",
                 "--out", str(fixtures), "--num-accounts", "50", "--seed", "9"],
                cwd=ROOT, check=False, text=True, capture_output=True,
            )
            self.assertEqual(synth.returncode, 0, synth.stderr + synth.stdout)
            bench = subprocess.run(
                [sys.executable, "-m", "relational_rag_tables.cli", "benchmark",
                 "--fixtures", str(fixtures), "--out", str(Path(tmp) / "bench"),
                 "--verifier", "deterministic"],
                cwd=ROOT, check=False, text=True, capture_output=True,
            )
            self.assertEqual(bench.returncode, 0, bench.stderr + bench.stdout)
            metrics = json.loads((Path(tmp) / "bench" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["qa"]["em"], 1.0)


if __name__ == "__main__":
    unittest.main()
