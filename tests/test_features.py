from __future__ import annotations

import unittest

from relational_rag_tables.features import FEATURE_NAMES, join_features


def feature(lhs: str, rhs: str, name: str) -> float:
    return join_features(lhs, rhs)[FEATURE_NAMES.index(name)]


class JoinFeatureTest(unittest.TestCase):
    def test_vector_length_matches_named_features(self) -> None:
        vector = join_features("ACC-1001", "account:ACC1001")
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        for value in vector:
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_exact_match_on_identical_normalised_identifiers(self) -> None:
        self.assertEqual(feature("ACC1001", "ACC1001", "exact_match"), 1.0)
        self.assertEqual(feature("ACC-1001", "account:ACC1002", "exact_match"), 0.0)

    def test_dash_drift_is_a_containment_and_digit_match(self) -> None:
        # The paper's J45.901 vs J45.9 regime: same entity, different formatting.
        self.assertEqual(feature("ACC-1001", "account:ACC1001", "containment"), 1.0)
        self.assertEqual(feature("ACC-1001", "account:ACC1001", "digit_full_match"), 1.0)
        self.assertEqual(feature("ACC-1001", "account:ACC1001", "digit_suffix_match"), 1.0)

    def test_near_duplicate_breaks_digit_agreement(self) -> None:
        self.assertEqual(feature("ACC-1001", "account:ACC1002", "digit_full_match"), 0.0)
        self.assertEqual(feature("ACC-1001", "account:ACC1002", "containment"), 0.0)
        # A different last digit still leaves strong n-gram/edit overlap, so the
        # signal is graded rather than binary — that is the point of learning.
        self.assertGreater(feature("ACC-1001", "account:ACC1002", "edit_ratio"), 0.5)

    def test_unrelated_identifiers_score_low_overlap(self) -> None:
        self.assertLess(feature("ACC-1001", "account:ZZ9999", "char3_jaccard"), 0.3)
        self.assertEqual(feature("ACC-1001", "account:ZZ9999", "digit_suffix_match"), 0.0)


if __name__ == "__main__":
    unittest.main()
