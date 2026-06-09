from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from relational_rag_tables.evaluate import evaluate_results
from relational_rag_tables.index import load_index, write_index
from relational_rag_tables.ingest import load_corpus
from relational_rag_tables.plan_execute import run_all
from relational_rag_tables.retrieve import retrieve
from relational_rag_tables.verify import DeterministicJoinVerifier


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "smoke"


class SmokePipelineTest(unittest.TestCase):
    def test_fixture_schema_validation(self) -> None:
        corpus = load_corpus(FIXTURES)
        self.assertEqual(len(corpus.questions), 3)
        self.assertIn("accounts", corpus.tables)

    def test_index_construction_writes_loadable_artifacts(self) -> None:
        corpus = load_corpus(FIXTURES)
        with tempfile.TemporaryDirectory() as tmp:
            entries = write_index(corpus, tmp)
            loaded = load_index(tmp)
            self.assertGreater(len(entries), 0)
            self.assertEqual(len(entries), len(loaded))
            self.assertTrue((Path(tmp) / "manifest.json").exists())

    def test_retriever_returns_expected_join_candidate(self) -> None:
        corpus = load_corpus(FIXTURES)
        with tempfile.TemporaryDirectory() as tmp:
            entries = write_index(corpus, tmp)
            result = retrieve(entries, "account A-1001 holds instrument", top_k=5)
            edge_text = " ".join(edge.text for edge in result.edges)
            self.assertIn("account:ACC1001", edge_text)

    def test_verifier_accepts_good_join_and_rejects_bad_join(self) -> None:
        corpus = load_corpus(FIXTURES)
        verifier = DeterministicJoinVerifier(corpus.join_labels)
        self.assertGreaterEqual(verifier.score("ACC-1001", "account:ACC1001"), 0.6)
        self.assertLess(verifier.score("ACC-1001", "account:ACC1002"), 0.6)

    def test_planner_executor_completes_end_to_end(self) -> None:
        corpus = load_corpus(FIXTURES)
        with tempfile.TemporaryDirectory() as tmp:
            entries = write_index(corpus, tmp)
            results = run_all(corpus, entries)
        metrics = evaluate_results(corpus, results)
        self.assertEqual(metrics["em"], 1.0)
        self.assertGreaterEqual(metrics["verifier_confusion"]["tp"], 1)
        self.assertGreaterEqual(metrics["verifier_confusion"]["tn"], 1)
        backtrack = next(result for result in results if result.question_id == "q_bad_join_backtrack")
        joins = [event for event in backtrack.trace if event.operation.type == "join"]
        self.assertEqual([event.accepted for event in joins[:2]], [False, True])

    def test_cli_smoke_command_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "relational_rag_tables.cli",
                    "smoke",
                    "--fixtures",
                    str(FIXTURES),
                    "--out",
                    tmp,
                ],
                cwd=ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            metrics_path = Path(tmp) / "metrics.json"
            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["em"], 1.0)


if __name__ == "__main__":
    unittest.main()
