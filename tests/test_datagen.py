from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from relational_rag_tables import datagen
from relational_rag_tables.datagen import GenConfig
from relational_rag_tables.evaluate import evaluate_results
from relational_rag_tables.index import write_index
from relational_rag_tables.ingest import load_corpus
from relational_rag_tables.plan_execute import run_all


SMALL = GenConfig(num_accounts=60, seed=11, n_verifier_train=200, n_verifier_eval=120)


class DatagenTest(unittest.TestCase):
    def test_generation_is_deterministic_for_a_seed(self) -> None:
        first = datagen.generate(SMALL)
        second = datagen.generate(SMALL)
        self.assertEqual(first.accounts, second.accounts)
        self.assertEqual(first.triples, second.triples)
        self.assertEqual(first.questions, second.questions)
        self.assertEqual(first.join_labels, second.join_labels)
        self.assertEqual(first.verifier_pairs, second.verifier_pairs)
        self.assertEqual(first.verifier_eval, second.verifier_eval)

    def test_verifier_split_sizes_match_config(self) -> None:
        corpus = datagen.generate(SMALL)
        self.assertEqual(len(corpus.verifier_pairs), SMALL.n_verifier_train)
        self.assertEqual(len(corpus.verifier_eval), SMALL.n_verifier_eval)
        self.assertEqual(len(corpus.accounts), SMALL.num_accounts)

    def test_train_accounts_are_reserved_from_questions(self) -> None:
        corpus = datagen.generate(SMALL)
        splits = {question["split"] for question in corpus.questions}
        # Train accounts back the verifier and must not leak into the QA set.
        self.assertNotIn("train", splits)
        self.assertTrue({"dev", "test"} & splits)

    def test_ghost_accounts_have_no_kg_node(self) -> None:
        corpus = datagen.generate(SMALL)
        instrument_subjects = {
            triple["subject"] for triple in corpus.triples if triple["relation"] == "holds_instrument"
        }
        self.assertEqual(
            len(instrument_subjects), corpus.meta["num_accounts"] - corpus.meta["ghost_accounts"]
        )
        # Every ghost account yields exactly one abstain question.
        abstain_questions = [q for q in corpus.questions if q["task"] == "abstain"]
        self.assertEqual(len(abstain_questions), corpus.meta["ghost_accounts"])
        self.assertTrue(all(q["answer"] == "ABSTAIN" for q in abstain_questions))

    def test_written_corpus_round_trips_through_loader(self) -> None:
        corpus = datagen.generate(SMALL)
        with tempfile.TemporaryDirectory() as tmp:
            meta = datagen.write_corpus(corpus, tmp)
            loaded = load_corpus(tmp)
            self.assertEqual(len(loaded.questions), meta["num_questions"])
            self.assertIn("accounts", loaded.tables)
            self.assertEqual(
                len(datagen.load_verifier_pairs(tmp)), SMALL.n_verifier_train
            )
            self.assertEqual(len(datagen.load_verifier_eval(tmp)), SMALL.n_verifier_eval)

    def test_qa_join_drift_preserves_digits_but_breaks_containment(self) -> None:
        from relational_rag_tables.index import normalize_identifier

        corpus = datagen.generate(GenConfig(num_accounts=200, seed=3, qa_join_drift=0.7))
        self.assertGreater(corpus.meta["drifted_nodes"], 0)
        positives = [(p["lhs"], p["rhs"]) for p in corpus.join_labels if p["label"]]
        broke_containment = 0
        for lhs, rhs in positives:
            left, right = normalize_identifier(lhs), normalize_identifier(rhs)
            left_digits = "".join(c for c in left if c.isdigit())
            right_digits = "".join(c for c in right if c.isdigit())
            self.assertEqual(left_digits, right_digits)  # numeric key always preserved
            if not (left in right or right in left):
                broke_containment += 1
        # Drift must actually defeat substring matching on a real fraction of joins.
        self.assertGreater(broke_containment, 0)

    def test_drift_is_off_by_default(self) -> None:
        corpus = datagen.generate(GenConfig(num_accounts=80, seed=1))
        self.assertEqual(corpus.meta["drifted_nodes"], 0)

    def test_deterministic_pipeline_solves_the_synthetic_corpus(self) -> None:
        corpus = datagen.generate(SMALL)
        with tempfile.TemporaryDirectory() as tmp:
            datagen.write_corpus(corpus, tmp)
            loaded = load_corpus(tmp)
            entries = write_index(loaded, Path(tmp) / "indexes")
            results = run_all(loaded, entries)
        metrics = evaluate_results(loaded, results)
        # The oracle (deterministic) verifier holds the gold join labels, so the
        # full control flow — select/join/traverse/backtrack/abstain — is exact.
        self.assertEqual(metrics["em"], 1.0)
        self.assertEqual(metrics["abstentions"], corpus.meta["ghost_accounts"])


if __name__ == "__main__":
    unittest.main()
