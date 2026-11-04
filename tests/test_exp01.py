"""Scientific and provenance gates; no reserved d=20 confirmation runs."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from xai_metric_behavior.analysis import analyze_scores, point_judgment
from xai_metric_behavior.core import (LinearOracle, checked_pearson, contribution_vectors,
                                      draw_plan, enumerate_fc, perturbation_trace, singleton_fe)
from xai_metric_behavior.experiment import (DEFAULT_WEIGHTS, freeze_confirmation, load_config,
                                            run_experiment)
from xai_metric_behavior.quantus_backend import PublicScore, public_score as official_public_score


ROOT = Path(__file__).resolve().parents[1]


class MeasurementContractTests(unittest.TestCase):
    def test_worked_toy_has_hand_computed_trace(self):
        model = LinearOracle(np.array([2.0, 1.0, 0.0]))
        x = np.array([3.0, 4.0, 5.0])
        plan = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        result = perturbation_trace(model, x, 0.0, {"corrupt": np.array([0.0, 4.0, 6.0])}, plan, 1)
        np.testing.assert_array_equal(result["U"][:, 0], [0, 4, 6])
        np.testing.assert_array_equal(result["V"], [6, 4, 0])
        np.testing.assert_array_equal(result["perturbed_outputs"], [4, 6, 10])
        self.assertAlmostEqual(checked_pearson(result["U"][:, 0], result["V"]).score, -13 / 14)
        np.testing.assert_array_equal(x, [3, 4, 5])  # replacements never accumulate

    def test_additive_nonlinear_identity_by_exact_enumeration(self):
        x = np.array([0.7, -1.3, 0.5, 2.1])
        base = np.array([0.1, 0.2, -0.3, 0.4])
        def parts(z):
            return np.stack([np.sin(z[..., 0]), z[..., 1] ** 2,
                             np.exp(0.2 * z[..., 2]), 2 * z[..., 3]], axis=-1)
        def model(z):
            return parts(np.asarray(z)).sum(axis=-1) + 1.4
        v = parts(x) - parts(base)
        a = np.array([2.0, -0.4, 0.7, 3.0])
        expected = checked_pearson(a, v).score
        for k in [1, 2, 3]:
            actual = enumerate_fc(model, x, base, {"candidate": a}, k)["candidate"]
            self.assertTrue(actual.valid)
            self.assertAlmostEqual(actual.score, expected, places=12)
        self.assertAlmostEqual(singleton_fe(model, x, base, a).score, expected, places=12)

    def test_undefined_cases_are_not_zero_scores(self):
        for u, v, reason in [([1, 1, 1], [1, 2, 3], "constant_attribution_sums"),
                             ([1, 2, 3], [2, 2, 2], "constant_response"),
                             ([0, 0], [0, 0], "constant_both"),
                             ([1], [2], "fewer_than_two_observations"),
                             ([1, np.inf], [1, 2], "nonfinite_input")]:
            result = checked_pearson(np.array(u), np.array(v))
            self.assertFalse(result.valid)
            self.assertTrue(np.isnan(result.score))
            self.assertEqual(result.reason, reason)
        model = LinearOracle(np.array([2.0, 1.0, 0.0]))
        for k in [0, 3]:
            result = enumerate_fc(model, np.array([3.0, 4.0, 5.0]), 0.0,
                                  {"a": np.array([6.0, 4.0, 0.0])}, k)["a"]
            self.assertFalse(result.valid)

    def test_interaction_is_a_boundary_not_a_theorem_refutation(self):
        def model(z):
            z = np.asarray(z)
            return z[..., 0] * z[..., 1] + 2.0 * z[..., 2]
        x = np.array([2.0, 3.0, 1.0, 4.0])
        baseline_shapley = np.array([3.0, 3.0, 2.0, 0.0])
        result = enumerate_fc(model, x, 0.0, {"shapley": baseline_shapley}, 1)["shapley"]
        self.assertTrue(result.valid)
        self.assertLess(result.score, 0.99)  # exact allocation need not equal deletion effects
        self.assertAlmostEqual(checked_pearson(baseline_shapley, baseline_shapley).score, 1.0)

    def test_nonuniform_sampling_changes_the_reference_population(self):
        a, v = np.array([1.0, -0.5, 2.0]), np.array([3.0, 1.0, -1.0])
        p = np.array([0.8, 0.15, 0.05])
        ac, vc = a - np.dot(p, a), v - np.dot(p, v)
        nonuniform = np.dot(p, ac * vc) / np.sqrt(np.dot(p, ac ** 2) * np.dot(p, vc ** 2))
        uniform = checked_pearson(a, v).score
        self.assertGreater(abs(nonuniform - uniform), 0.1)

    def test_relocation_pair_preserves_values_and_has_derived_margin(self):
        weights = np.array(DEFAULT_WEIGHTS)
        x = np.random.default_rng(7).normal(size=(8, 20))
        candidates = contribution_vectors(x, weights, 0.0)
        v = candidates["exact"]
        for name in ["A", "B"]:
            np.testing.assert_array_equal(np.sort(candidates[name], axis=1), np.sort(v, axis=1))
            np.testing.assert_allclose(np.abs(candidates[name]).sum(axis=1), np.abs(v).sum(axis=1))
        for i in range(len(x)):
            a = checked_pearson(candidates["A"][i], v[i]).score
            b = checked_pearson(candidates["B"][i], v[i]).score
            margin = v[i, 1] ** 2 / np.sum((v[i] - v[i].mean()) ** 2)
            self.assertAlmostEqual(a - b, margin, places=12)
            self.assertGreaterEqual(a - b, -1e-12)

    def test_pairing_preserves_subsets_and_budget_prefixes(self):
        plan = draw_plan(20, 100, 12)
        np.testing.assert_array_equal(np.sort(plan, axis=1), np.tile(np.arange(20), (100, 1)))
        np.testing.assert_array_equal(plan, draw_plan(20, 100, 12))
        self.assertFalse(np.array_equal(plan, draw_plan(20, 100, 13)))
        # Every k-prefix is exactly uniform when the primitive permutations are uniform.
        all_orders = np.array(list(permutations(range(4))))
        unique, counts = np.unique(np.sort(all_orders[:, :2], axis=1), axis=0, return_counts=True)
        self.assertEqual(len(unique), 6)
        np.testing.assert_array_equal(counts, [4] * 6)
        model, x = LinearOracle(np.array(DEFAULT_WEIGHTS)), np.arange(20) / 3
        candidates = contribution_vectors(x, model.weights)
        trace = perturbation_trace(model, x, 0.0, candidates, plan, 5)
        shorter = perturbation_trace(model, x, 0.0, candidates, plan[:25], 5)
        np.testing.assert_array_equal(trace["U"][:25], shorter["U"])
        np.testing.assert_array_equal(trace["V"][:25], shorter["V"])
        self.assertAlmostEqual(checked_pearson(trace["U"][:, 0], trace["V"]).score, 1.0)

    def test_common_eligibility_precedes_condition_averaging(self):
        scores = np.full((2, 3, 2, 1, 3), 0.4)
        scores[0, :, 0, 0, 1] = [0.8, np.nan, -0.9]
        scores[0, :, 1, 0, 1] = [0.9, 0.3, np.nan]
        scores[1, :, 0, 0, 1] = [0.1, 0.1, 0.1]
        scores[1, :, 1, 0, 1] = [0.2, 0.2, 0.2]
        reasons = np.where(np.isfinite(scores), "valid", "constant_response")
        results = analyze_scores(scores, reasons, np.array([[1, 0.5, 0.3], [1, 0.6, 0.2]]), [1, 5], [25],
                                 tie_tolerance=0.01, bootstrap_resamples=20, bootstrap_confidence=0.95, analysis_seed=42)
        label = "Delta_A_k5_B25_minus_k1_B25"
        record = next(r for r in results["contrast_records"] if r["contrast_id"] == label and r["input_id"] == 0)
        self.assertEqual(record["joint_valid_repetitions"], "[0]")
        self.assertAlmostEqual(record["mean_difference_given_joint_valid"], 0.1)
        summary = next(r for r in results["contrast_summary"] if r["contrast_id"] == label)
        self.assertAlmostEqual(summary["mean_difference"], 0.1)
        self.assertEqual(summary["n_joint_valid"], 4)
        self.assertEqual(summary["selection_warning"], "conditional_only_broader_judgment_unresolved")

    def test_tie_and_undefined_states_are_distinct(self):
        self.assertEqual(point_judgment(0.0, 0.01), "exact_numerical_tie")
        self.assertEqual(point_judgment(0.005, 0.01), "practical_tie")
        self.assertEqual(point_judgment(-0.1, 0.01), "B_preferred")
        self.assertEqual(point_judgment(float("nan"), 0.01), "undefined")


class ProvenanceTests(unittest.TestCase):
    def small_config(self, path: Path, phase="smoke"):
        config = yaml.safe_load((ROOT / "configs" / "exp01_development.yaml").read_text())
        config.update(phase=phase, n_inputs=2, repetitions=2, subset_sizes=[1, 5], budgets=[5, 10])
        config["analysis"]["bootstrap_resamples"] = 20
        path.write_text(yaml.safe_dump(config))
        return config

    def test_config_rejects_silent_contract_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            config = self.small_config(path)
            config["abs"] = True
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "unexpected/missing"):
                load_config(path)
            config.pop("abs")
            config["subset_sizes"] = [2]
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "reserved"):
                load_config(path)

    @patch("xai_metric_behavior.plotting.plot_results")
    def test_replay_outputs_traces_and_no_clobber(self, _plot):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = folder / "config.yaml"
            self.small_config(config)
            one, two = folder / "one", folder / "two"
            first, second = run_experiment(config, one), run_experiment(config, two)
            self.assertEqual(first["status"], "completed")
            self.assertEqual((one / "scores.csv").read_text(), (two / "scores.csv").read_text())
            self.assertEqual(first["counts"], second["counts"])
            self.assertEqual(first["metric_variant"], "exp01-quantus-public-0.6.0-v1")
            self.assertEqual(first["counts"]["n_public_FC_calls"], 48)
            self.assertEqual(first["counts"]["n_public_FE_calls"], 6)
            self.assertEqual(first["counts"]["n_trace_checks_passed"], 54)
            self.assertTrue(first["validation"]["all_FC_calls_paired"])
            calls = [json.loads(line) for line in (one / "metric_calls.jsonl").read_text().splitlines()]
            self.assertEqual(len(calls), 54)
            self.assertTrue(all(call["pairing_verified"] for call in calls if call["metric"] == "FC"))
            self.assertTrue(all(call["exception"] is None for call in calls))
            with np.load(one / "traces" / "input_0000.npz") as trace:
                self.assertNotIn("permutations", trace.files)
                self.assertEqual(trace["subsets_k1"].shape, (2, 10, 1))
                self.assertEqual(trace["subsets_k5"].shape, (2, 10, 5))
                np.testing.assert_array_equal(trace["subsets_k1"], trace["subsets_k5"][:, :, :1])
                for rep in [0, 1]:
                    recovered = checked_pearson(trace["U"][rep, 0, :5, 1], trace["V"][rep, 0, :5])
                    with (one / "scores.csv").open() as stream:
                        row = next(r for r in csv.DictReader(stream) if r["input_id"] == "0" and r["repetition"] == str(rep)
                                   and r["k"] == "1" and r["budget"] == "5" and r["explanation"] == "A")
                    if recovered.valid:
                        self.assertAlmostEqual(recovered.score, float(row["score"]))
                    else:
                        self.assertEqual(recovered.reason, row["reason"])
            with np.load(one / "traces" / "FE_input_0000_A.npz") as trace:
                np.testing.assert_array_equal(np.sort(trace["subsets"][:, 0]), np.arange(20))
                self.assertEqual(trace["model_inputs"].shape, (21, 20))
            with (one / "population_scores.csv").open() as stream:
                self.assertTrue(all(row["singleton_FE_source"] == "quantus.FaithfulnessEstimate.__call__"
                                    for row in csv.DictReader(stream)))
            with self.assertRaises(FileExistsError):
                run_experiment(config, one)

    @patch("xai_metric_behavior.plotting.plot_results")
    def test_failure_preserves_counts_and_diagnostic(self, _plot):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = folder / "config.yaml"
            self.small_config(config)
            def failed_after_official_return(kind, *args, **kwargs):
                if kind == "FE":
                    return official_public_score(kind, *args, **kwargs)
                return PublicScore(0.4321, False, "exception:RuntimeError",
                                   exception="RuntimeError: injected diagnostic failure after a finite official return")
            with patch("xai_metric_behavior.experiment.public_score", side_effect=failed_after_official_return):
                with self.assertRaisesRegex(RuntimeError, "injected diagnostic"):
                    run_experiment(config, folder / "failed")
            manifest = json.loads((folder / "failed" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["counts"]["n_attempted_scores"], 1)
            self.assertEqual(manifest["counts"]["n_exception_scores"], 1)
            self.assertEqual(manifest["counts"]["n_exceptions"], 1)
            self.assertGreater(manifest["counts"]["n_planned_scores"], 0)
            self.assertIn("injected diagnostic failure", (folder / "failed" / "failures.csv").read_text())
            with (folder / "failed" / "scores.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["score"]), 0.4321)
            self.assertEqual(row["valid"], "False")
            self.assertIn("injected diagnostic failure", row["exception"])
            last_call = json.loads((folder / "failed" / "metric_calls.jsonl").read_text().splitlines()[-1])
            self.assertEqual(last_call["score"], 0.4321)
            self.assertFalse(last_call["valid"])
            self.assertTrue((folder / "failed" / last_call["trace_file"]).exists())
            self.assertFalse((folder / "failed" / "contrast_summary.csv").exists())

    @patch("xai_metric_behavior.plotting.plot_results")
    def test_freeze_never_runs_reserved_conditions_and_rejects_changes(self, _plot):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dev_path, confirm_path = folder / "dev.yaml", folder / "confirm.yaml"
            self.small_config(dev_path, phase="development")
            development = run_experiment(dev_path, folder / "development")
            confirm = yaml.safe_load((ROOT / "configs" / "exp01_confirmation.yaml").read_text())
            confirm["analysis"]["bootstrap_resamples"] = 20
            confirm_path.write_text(yaml.safe_dump(confirm))
            with self.assertRaisesRegex(ValueError, "requires --freeze"):
                run_experiment(confirm_path, folder / "should_not_exist")
            self.assertFalse((folder / "should_not_exist").exists())
            freeze = freeze_confirmation(confirm_path, folder / "development", folder / "freeze.json")
            self.assertEqual(freeze["configuration"]["subset_sizes"], [2, 7, 13])
            self.assertEqual(freeze["status"], "frozen_before_confirmation")
            self.assertEqual(freeze["backend_execution_sha256"], development["backend_execution_sha256"])
            changed_backend = {**development["quantus_backend"], "execution_sha256": "different-runtime"}
            with patch("xai_metric_behavior.experiment.backend_provenance", return_value=changed_backend), \
                    patch("xai_metric_behavior.experiment.public_score") as no_reserved_call:
                with self.assertRaisesRegex(ValueError, "backend changed after development"):
                    freeze_confirmation(confirm_path, folder / "development", folder / "invalid-freeze.json")
                with self.assertRaisesRegex(ValueError, "backend differs from the freeze"):
                    run_experiment(confirm_path, folder / "should_not_exist", freeze_path=folder / "freeze.json")
                no_reserved_call.assert_not_called()
            self.assertFalse((folder / "invalid-freeze.json").exists())
            self.assertFalse((folder / "should_not_exist").exists())
            with self.assertRaises(FileExistsError):
                freeze_confirmation(confirm_path, folder / "development", folder / "freeze.json")
            confirm["n_inputs"] += 1
            confirm_path.write_text(yaml.safe_dump(confirm))
            with self.assertRaisesRegex(ValueError, "differs from its frozen"):
                run_experiment(confirm_path, folder / "should_not_exist", freeze_path=folder / "freeze.json")
            self.assertFalse((folder / "should_not_exist").exists())


if __name__ == "__main__":
    unittest.main()
