"""Optional integration tests; an absent Quantus installation is an explicit skip."""

from functools import lru_cache
from importlib.util import find_spec
import unittest

from scripts.audit_quantus import run_audit


@lru_cache(maxsize=1)
def audit():
    return run_audit()


@unittest.skipUnless(find_spec("quantus"), "Optional quantus==0.6.0 is not installed")
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

    def test_actual_evaluate_batch_matches_engine_traces(self):
        self.assert_group_passes("batch_")

    @unittest.skipUnless(find_spec("torch"), "Optional torch is not installed")
    def test_public_api_matches_and_retains_expected_2d_failure(self):
        self.assert_group_passes("public_")


if __name__ == "__main__":
    unittest.main()
