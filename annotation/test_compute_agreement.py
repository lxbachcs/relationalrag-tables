#!/usr/bin/env python3
"""Standard-library tests for package validation and agreement arithmetic."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_agreement import kappa, validate_package


class KappaTests(unittest.TestCase):
    def test_perfect_nondegenerate_agreement(self) -> None:
        self.assertEqual(kappa([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)

    def test_known_disagreement(self) -> None:
        self.assertEqual(kappa([0, 0, 1, 1], [0, 1, 0, 1]), 0.0)

    def test_degenerate_agreement_is_undefined(self) -> None:
        self.assertIsNone(kappa([1, 1, 1], [1, 1, 1]))

    def test_empty_agreement_is_undefined(self) -> None:
        self.assertIsNone(kappa([], []))

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            kappa([1], [])


class PackageTests(unittest.TestCase):
    def test_frozen_samples_and_offline_reference(self) -> None:
        _, result = validate_package()
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["datasets"]["bird"]["sample_rows"], 100)
        self.assertEqual(result["datasets"]["icd"]["sample_rows"], 100)
        self.assertEqual(result["offline_icd_reference"]["sample_qids_covered"], 99)
        self.assertEqual(result["offline_context"]["bird_schema_databases"], 11)
        self.assertEqual(result["offline_context"]["bird_column_value_rows"], 100)
        self.assertGreater(result["offline_context"]["icd10cm_reference_rows"], 100)
        self.assertEqual(result["external_python_dependencies"], [])

    def test_every_bird_row_carries_sql_and_resolved_table(self) -> None:
        loaded, _ = validate_package()
        for row in loaded["bird"]["raw"]:
            self.assertTrue(row["gold_sql"].strip(), row["id"])
            self.assertNotIn("UNRESOLVED", row["sql_column_table"], row["id"])
            self.assertIn(f"{row['table']}.{row['column']}=present", row["same_name_columns"], row["id"])

    def test_every_icd_row_carries_code_validity(self) -> None:
        loaded, _ = validate_package()
        for row in loaded["icd"]["raw"]:
            self.assertIn(row["record_code_in_icd10cm"], {"yes", "no"}, row["id"])
            if row["record_code_in_icd10cm"] == "yes":
                self.assertTrue(row["record_code_icd10cm_desc"].strip(), row["id"])


if __name__ == "__main__":
    unittest.main()
