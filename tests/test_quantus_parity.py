"""Required source/fixture guards; a missing evaluator dependency is a failure."""

from functools import lru_cache
import unittest

from scripts.audit_quantus import run_audit


@lru_cache(maxsize=1)
def audit():
    return run_audit()


class QuantusParityTests(unittest.TestCase):
    def assert_group_passes(self, prefix):
        report = audit()
        matching = [check for check in report["checks"] if check["name"].startswith(prefix)]
        self.assertTrue(matching, report)
        failures = [check for check in matching if check["status"] != "passed"]
        self.assertFalse(failures, failures)

    def test_pinned_source_identity(self):
        self.assert_group_passes("pinned_source_identity")

    def test_perturbation_and_undefined_correlation(self):
        self.assert_group_passes("replacement_and_undefined_pearson")

    def test_upstream_shape_limit_is_explicit(self):
        self.assert_group_passes("unpatched_shape_validation")

    def test_actual_evaluate_batch_matches_numpy_reference(self):
        self.assert_group_passes("batch_")

    def test_public_api_matches_and_retains_expected_2d_failure(self):
        self.assert_group_passes("public_")


if __name__ == "__main__":
    unittest.main()
