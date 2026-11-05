"""Regression for wrapper-cache retention observed during full development.

Only synthetic fixtures and allowed subset sizes are used. Real public Quantus
calls exercise its cached wrappers. Weak references distinguish live model/log
retention from Python allocator arenas that may remain resident after release.
"""
from __future__ import annotations

import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch
from quantus.helpers.model.pytorch_model import PyTorchModel

from xai_metric_behavior import quantus_backend as backend


class QuantusModelCacheLifecycleTests(unittest.TestCase):
    def setUp(self):
        rng, dtype = np.random.get_state(), torch.get_default_dtype()
        self.addCleanup(np.random.set_state, rng)
        self.addCleanup(torch.set_default_dtype, dtype)
        self.caches = (PyTorchModel._get_last_softmax_layer_index,
                       PyTorchModel._get_model_with_linear_top)
        for cache in self.caches:
            cache.cache_clear()
            self.addCleanup(cache.cache_clear)
        self.x = np.linspace(-1.75, 2.25, 20, dtype=np.float64)
        self.weights = np.array([2.5, -2.0, 1.5, 1.0, -.5] + [0.0] * 15)
        self.a = self.x * self.weights
        self.a[[0, 5]] = self.a[[5, 0]]
        self.seed = np.array([1729, 41, 0, 9], dtype=np.uint32)

    def call(self, *, kind="FC", k=5, budget=400, seed=None):
        return backend.public_score(
            kind, self.x, self.weights, self.a,
            self.seed if seed is None else seed,
            subset_size=k if kind == "FC" else None,
            budget=budget if kind == "FC" else None,
        )

    def assert_cache_empty(self):
        for cache in self.caches:
            self.assertEqual(cache.cache_info().currsize, 0)

    def assert_state_restored(self, expected_rng):
        actual = np.random.get_state()
        self.assertEqual(actual[0], expected_rng[0])
        np.testing.assert_array_equal(actual[1], expected_rng[1])
        self.assertEqual(actual[2:], expected_rng[2:])
        self.assertEqual(torch.get_default_dtype(), torch.float32)

    def test_actual_fc_fe_calls_release_models_and_preserve_saved_traces(self):
        references, cache_sizes_during_validation = [], []
        validate = backend._validate_trace

        def observe(model, *args, **kwargs):
            # A plain wrapper, not Mock call history, must not itself retain model.
            references.append(weakref.ref(model))
            cache_sizes_during_validation.append(self.caches[0].cache_info().currsize)
            return validate(model, *args, **kwargs)

        first, saved = None, None
        with patch.object(backend, "_validate_trace", new=observe):
            for index in range(8):
                result = self.call(k=(1, 5, 10, 19)[index % 4],
                                   seed=np.array([1729, 41, index, 9], dtype=np.uint32))
                self.assertIsNone(result.exception, result.exception)
                self.assertEqual(result.checks["score_parity_status"], "passed")
                self.assert_cache_empty()
                if first is None:
                    first = result
                    saved = {key: np.array(result.trace[key], copy=True)
                             for key in ("model_inputs", "subsets", "U", "V", "perturbed_outputs")}
            fe = self.call(kind="FE")
            self.assertIsNone(fe.exception, fe.exception)
            self.assertEqual(len(fe.trace["input_dtypes"]), 21)
            self.assert_cache_empty()

        self.assertEqual(cache_sizes_during_validation, [1] * 9)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references),
                        "Completed calls must not leave cached wrappers rooting model prediction logs")
        self.assertEqual(first.trace["input_dtypes"], ["torch.float64"] * 401)
        for key, value in saved.items():
            np.testing.assert_array_equal(first.trace[key], value)

    def test_public_prediction_exception_cleans_cache_and_keeps_partial_trace(self):
        np.random.seed(90321)
        np.random.standard_normal()
        expected_rng = np.random.get_state()
        torch.set_default_dtype(torch.float32)
        forward, references, observed_cache_sizes = backend.ScalarLogitModel.forward, [], []

        def fail_second_prediction(model, value):
            result = forward(model, value)
            if len(model.inputs) == 2:
                references.append(weakref.ref(model))
                observed_cache_sizes.append(self.caches[0].cache_info().currsize)
                raise RuntimeError("injected failure after a real perturbed prediction")
            return result

        with patch.object(backend.ScalarLogitModel, "forward", new=fail_second_prediction):
            failed = self.call(budget=25)

        self.assertIn("injected failure", failed.exception)
        self.assertEqual(failed.exception_stage, "official_call")
        self.assertTrue(np.isnan(failed.score))
        self.assertEqual(observed_cache_sizes, [1])
        self.assertEqual(failed.trace["model_inputs"].shape, (2, 20))
        self.assertEqual(failed.trace["subsets"].shape, (1, 5))
        self.assertEqual(failed.trace["input_dtypes"], ["torch.float64"] * 2)
        self.assert_cache_empty()
        self.assert_state_restored(expected_rng)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_post_score_exception_keeps_return_and_full_exception_trace(self):
        expected = self.call(budget=25)
        self.assertIsNone(expected.exception, expected.exception)
        np.random.seed(90321)
        np.random.standard_normal()
        expected_rng = np.random.get_state()
        torch.set_default_dtype(torch.float32)
        validate, references, observed_cache_sizes = backend._validate_trace, [], []

        def fail_after_trace(model, *args, **kwargs):
            validate(model, *args, **kwargs)
            references.append(weakref.ref(model))
            observed_cache_sizes.append(self.caches[0].cache_info().currsize)
            raise RuntimeError("injected failure after official score and trace validation")

        with patch.object(backend, "_validate_trace", new=fail_after_trace):
            failed = self.call(budget=25)

        self.assertIn("injected failure", failed.exception)
        self.assertEqual(failed.exception_stage, "trace_validation")
        self.assertEqual(failed.score, expected.score)
        self.assertFalse(failed.analysis_eligible)
        self.assertEqual(observed_cache_sizes, [1])
        for key in ("model_inputs", "subsets", "U", "V"):
            np.testing.assert_array_equal(failed.trace[key], expected.trace[key])
        self.assertEqual(failed.trace["input_dtypes"], ["torch.float64"] * 26)
        self.assert_cache_empty()
        self.assert_state_restored(expected_rng)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))


if __name__ == "__main__":
    unittest.main()
