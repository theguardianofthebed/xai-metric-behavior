"""Public-Quantus adapter controls, separate from experiment/confirmation runs.

Native scoring tests delegate to genuine public APIs. Fault injection is limited
to the restoration test; it cannot alter a saved experimental result.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import quantus
import torch

from xai_metric_behavior import quantus_backend as backend


class PublicQuantusBackendTests(unittest.TestCase):
    def setUp(self):
        previous_rng = np.random.get_state()
        previous_dtype = torch.get_default_dtype()
        self.addCleanup(np.random.set_state, previous_rng)
        self.addCleanup(torch.set_default_dtype, previous_dtype)
        self.x = np.linspace(-1.37, 2.83, 20, dtype=np.float64) + np.pi / 100
        self.weights = np.array([2.5, -2.0, 1.5, 1.0, -0.5] + [0.0] * 15)
        self.exact = self.x * self.weights
        self.a = self.exact.copy()
        self.a[5], self.a[0] = self.a[0], 0.0
        self.b = self.a.copy()
        self.b[6], self.b[1] = self.b[1], 0.0
        self.seed = backend.evaluator_seed_words(2026110402, 0, 99, 0)

    def fc(self, attribution=None, *, k=5, budget=25, seed=None, weights=None):
        return backend.public_score(
            "FC", self.x, self.weights if weights is None else weights,
            self.exact if attribution is None else attribution,
            self.seed if seed is None else seed,
            subset_size=k, budget=budget,
        )

    def assert_rng_equal(self, expected):
        actual = np.random.get_state()
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2:], expected[2:])

    def assert_recorded(self, result):
        self.assertIsNone(result.exception, result.exception)
        self.assertIn("subsets", result.trace)
        self.assertIn("U", result.trace)
        self.assertIn("V", result.trace)

    def test_real_public_fc_and_fe_preserve_non_dyadic_float64_inputs(self):
        # A post-truncation cast to double would fail this input comparison.
        self.assertFalse(np.array_equal(self.x, self.x.astype(np.float32).astype(np.float64)))
        for metric_name, metric_class, expected_draws in (
            ("FC", quantus.FaithfulnessCorrelation, 25),
            ("FE", quantus.FaithfulnessEstimate, 20),
        ):
            with self.subTest(metric=metric_name):
                original_call = metric_class.__call__
                with patch.object(metric_class, "__call__", autospec=True,
                                  side_effect=original_call) as public_call:
                    result = (self.fc() if metric_name == "FC" else
                              backend.public_score("FE", self.x, self.weights,
                                                   self.exact, self.seed))
                public_call.assert_called_once()
                self.assert_recorded(result)
                self.assertTrue(result.valid, result.reason)
                self.assertTrue(result.analysis_eligible, result.reason)
                self.assertAlmostEqual(result.score, 1.0, places=12)
                np.testing.assert_array_equal(result.trace["model_inputs"][0], self.x)
                self.assertEqual(result.trace["model_inputs"].dtype, np.float64)
                self.assertEqual(result.trace["input_dtypes"], ["torch.float64"] * (expected_draws + 1))
                self.assertEqual(len(result.trace["subsets"]), expected_draws)
                self.assertEqual(len(result.trace["U"]), expected_draws)
                self.assertEqual(len(result.trace["V"]), expected_draws)

    def test_native_draws_are_paired_across_candidates_k_and_budget(self):
        reference = self.fc(k=19, budget=25)
        self.assert_recorded(reference)
        for label, attribution, k, budget in (
            ("A", self.a, 19, 25),
            ("B", self.b, 19, 25),
            ("A smaller k", self.a, 5, 25),
            ("B smaller budget", self.b, 5, 5),
            ("A singleton", self.a, 1, 5),
        ):
            with self.subTest(condition=label):
                result = self.fc(attribution, k=k, budget=budget)
                self.assert_recorded(result)
                np.testing.assert_array_equal(
                    result.trace["subsets"], reference.trace["subsets"][:budget, :k])
                backend.verify_pairing(reference.trace, result.trace,
                                       budget=budget, subset_size=k)

    def test_repeat_seed_reproduces_draws_and_fresh_repeat_changes_them(self):
        first = self.fc(self.a)
        replay = self.fc(self.a)
        fresh_seed = backend.evaluator_seed_words(2026110402, 0, 99, 1)
        fresh = self.fc(self.a, seed=fresh_seed)
        for result in (first, replay, fresh):
            self.assert_recorded(result)
        np.testing.assert_array_equal(first.trace["subsets"], replay.trace["subsets"])
        self.assertEqual(first.score, replay.score)
        self.assertFalse(np.array_equal(self.seed, fresh_seed))
        self.assertFalse(np.array_equal(first.trace["subsets"], fresh.trace["subsets"]))

    def test_global_rng_and_dtype_restore_after_success_and_public_exception(self):
        np.random.seed(90321)
        np.random.standard_normal()  # Include the legacy normal-cache fields.
        expected_rng = np.random.get_state()
        torch.set_default_dtype(torch.float32)
        success = self.fc(self.a, budget=5)
        self.assert_recorded(success)
        self.assert_rng_equal(expected_rng)
        self.assertEqual(torch.get_default_dtype(), torch.float32)
        with patch.object(quantus.FaithfulnessCorrelation, "__call__",
                          side_effect=RuntimeError("injected public-call failure")):
            failed = self.fc(self.a, budget=5)
        self.assertFalse(failed.valid)
        self.assertIn("injected public-call failure", failed.exception)
        self.assert_rng_equal(expected_rng)
        self.assertEqual(torch.get_default_dtype(), torch.float32)

    def test_real_public_constant_response_retains_nan_reason_and_warning(self):
        result = self.fc(np.arange(1.0, 21.0), weights=np.zeros(20))
        self.assert_recorded(result)
        self.assertFalse(result.valid)
        self.assertTrue(np.isnan(result.score))
        self.assertEqual(result.reason, "constant_response")
        self.assertFalse(result.analysis_eligible)
        np.testing.assert_array_equal(result.trace["V"], np.zeros(25))
        self.assertGreater(np.ptp(result.trace["U"]), 0)
        self.assertTrue(any(w["category"] == "ConstantInputWarning" for w in result.warnings),
                        result.warnings)

    def test_real_public_validation_exception_has_no_numpy_score_fallback(self):
        original_call = quantus.FaithfulnessCorrelation.__call__
        with patch.object(quantus.FaithfulnessCorrelation, "__call__", autospec=True,
                          side_effect=original_call) as public_call:
            with patch.object(backend, "checked_pearson", wraps=backend.checked_pearson) as replay:
                result = self.fc(self.a, k=20, budget=5)  # Excluded k=d boundary.
        public_call.assert_called_once()
        replay.assert_not_called()
        self.assertFalse(result.valid)
        self.assertTrue(np.isnan(result.score))
        self.assertIn("subset_size", result.exception)
        self.assertTrue(result.reason.startswith("exception:"))

    def test_recorded_near_constant_smoke_value_is_preserved_but_ineligible(self):
        # Captured from first official attempt, source 02cd3d2, call 344:
        # input 2, repetition 1, candidate A, k=19, B=5. Literal inputs make
        # this independent of later result files or generator refactoring.
        x = np.array([0.9289979736951923, -1.4621129036211873, -0.32088905891569236, 0.33162073351301047, 0.5362366220559647, -0.16397913503256398, 0.9537486395213012, -0.09966578548825879, 1.5796151188185499, 1.9021361539137442, -0.40496696757653317, 0.3346197102928255, 1.3991829271685752, 1.4119256038002874, -0.38587986033628946, -1.870462549453295, -1.1831467759926897, 0.46111968314396323, -0.012310808861654783, 0.17776035714289476], dtype=np.float64)
        attribution = np.array([-0.0, 2.9242258072423746, -0.48133358837353857, 0.33162073351301047, -0.26811831102798234, 2.322494934237981, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0], dtype=np.float64)
        seed = backend.evaluator_seed_words(2026110402, 0, 2, 1)
        result = backend.public_score("FC", x, self.weights, attribution, seed,
                                      subset_size=19, budget=5)
        self.assert_recorded(result)
        self.assertTrue(result.valid)  # The official return really is finite.
        self.assertEqual(result.score, 0.31622776601683805)
        self.assertFalse(result.analysis_eligible)
        self.assertEqual(result.reason, "near_constant_input")
        self.assertTrue(any(w["category"] == "NearConstantInputWarning" for w in result.warnings))
        self.assertTrue(result.checks["near_constant_warning"])
        self.assertEqual(result.checks["score_parity_status"], "not_asserted_near_constant")
        self.assertAlmostEqual(result.checks["trace_replay_absolute_error"],
                               abs(result.score - 0.2581988897471609), places=14)
        self.assertGreater(result.checks["trace_replay_absolute_error"], 1e-10)
        self.assertTrue(result.checks["input_exactly_preserved"])
        self.assertTrue(result.checks["original_replacement_verified"])
        np.testing.assert_array_equal(result.trace["model_inputs"][0], x)
        self.assertEqual(result.trace["subsets"].shape, (5, 19))
        self.assertEqual(np.ptp(result.trace["U"]), 8.881784197001252e-16)
        self.assertGreater(np.ptp(result.trace["V"]), 0)


if __name__ == "__main__":
    unittest.main()
