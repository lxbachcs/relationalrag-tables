from __future__ import annotations

import math
import tempfile
import unittest

from relational_rag_tables import datagen
from relational_rag_tables.datagen import GenConfig
from relational_rag_tables.uncertainty import EnsembleVerifier, _binary_entropy


def _pairs() -> list[tuple[str, str, bool]]:
    corpus = datagen.generate(GenConfig(num_accounts=60, seed=5, n_verifier_train=300))
    return [(item["lhs"], item["rhs"], bool(item["label"])) for item in corpus.verifier_pairs]


class EnsembleVerifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ensemble = EnsembleVerifier.train(_pairs(), k=3, seed=0)

    def test_ensemble_has_requested_members(self) -> None:
        self.assertEqual(self.ensemble.k, 3)
        self.assertEqual(len(self.ensemble.member_scores("ACC-1020", "account:ACC1020")), 3)

    def test_scores_are_probabilities(self) -> None:
        score = self.ensemble.score("ACC-1020", "account:ACC1020")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_uncertainty_is_entropy_of_the_mean_decision(self) -> None:
        lhs, rhs = "ACC-1020", "account:ACC1021"
        expected = _binary_entropy(self.ensemble.score(lhs, rhs))
        self.assertAlmostEqual(self.ensemble.uncertainty(lhs, rhs), expected, places=9)
        self.assertGreaterEqual(self.ensemble.uncertainty(lhs, rhs), 0.0)
        self.assertLessEqual(self.ensemble.uncertainty(lhs, rhs), math.log(2) + 1e-9)

    def test_variance_is_non_negative(self) -> None:
        self.assertGreaterEqual(self.ensemble.variance("ACC-1020", "account:ACC1020"), 0.0)

    def test_clean_match_outscores_an_unrelated_node(self) -> None:
        clean = self.ensemble.score("ACC-1020", "account:ACC1020")
        unrelated = self.ensemble.score("ACC-1020", "account:ZZ0000")
        self.assertGreater(clean, unrelated)

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.ensemble.save(tmp)
            reloaded = EnsembleVerifier.load(tmp)
        self.assertEqual(reloaded.k, self.ensemble.k)
        self.assertAlmostEqual(
            reloaded.score("ACC-1020", "account:ACC1020"),
            self.ensemble.score("ACC-1020", "account:ACC1020"),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
