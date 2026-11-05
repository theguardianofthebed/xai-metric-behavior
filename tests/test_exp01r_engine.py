"""EXP-01-R safeguards using synthetic fixtures, never reserved Adult records."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from xai_metric_behavior.exp01r import load_exp01r_config, run_exp01r
from xai_metric_behavior.exp01r_analysis import analyze_real_scores
from xai_metric_behavior.exp01r_data import PreparedAdult
from xai_metric_behavior.quantus_backend import evaluator_seed_words, public_score


ROOT = Path(__file__).resolve().parents[1]


class RealEngineTests(unittest.TestCase):
    def config(self, folder: Path) -> Path:
        config = yaml.safe_load((ROOT / "configs/exp01r_smoke.yaml").read_text())
        config.update(repetitions=2, subset_sizes=[1, 5], budgets=[5, 10])
        config["analysis"]["bootstrap_resamples"] = 20
        path = folder / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def fixture(self, reject_first: bool = False) -> PreparedAdult:
        weights = np.linspace(-1.1, 1.3, 13)
        x = np.random.default_rng(921).normal(size=(4, 13))
        if reject_first:
            x[0] = -np.sign(weights) * np.arange(1., 14.) / 2
        reference = x * weights
        return PreparedAdult(
            x, np.array([0, 1, 0, 1]), np.arange(100, 104), weights, -1.372941,
            reference, {"IG": reference.copy(), "A": np.roll(reference, 1, axis=1),
                        "B": np.roll(reference, 2, axis=1)},
            {"model_sha256": "synthetic_fixture_only", "partition_sha256": "no_Adult_records_read"},
        )

    def test_nonzero_intercept_is_exposed_in_actual_public_fc_and_fe_traces(self):
        data = self.fixture()
        seed = evaluator_seed_words(2026110503, 10, 0, 0)
        for kind in ("FC", "FE"):
            with self.subTest(metric=kind):
                result = public_score(kind, data.x[0], data.weights, data.explanations["A"][0], seed,
                                      subset_size=5, budget=10, bias=data.bias)
                self.assertIsNone(result.exception, result.exception)
                self.assertTrue(result.analysis_eligible)
                self.assertAlmostEqual(result.trace["original_output"], float(data.x[0] @ data.weights + data.bias), places=12)
                expected = np.sum(result.trace["model_inputs"][1:] * data.weights, axis=1) + data.bias
                np.testing.assert_allclose(result.trace["perturbed_outputs"], expected, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(result.trace["V"], result.trace["original_output"] - result.trace["perturbed_outputs"])

    def test_reserved_and_unknown_contract_rejected_before_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = self.config(folder)
            base = yaml.safe_load(path.read_text())
            with patch("xai_metric_behavior.exp01r.prepare_adult") as no_data:
                for changed, error in (({"phase": "confirmation"}, "blocked"),
                                       ({"subset_sizes": [2]}, "reserved"),
                                       ({"abs": True}, "unexpected/missing")):
                    path.write_text(yaml.safe_dump({**base, **changed}))
                    with self.assertRaisesRegex(ValueError, error):
                        run_exp01r(path, folder / "not-created", folder / "data")
                    self.assertFalse((folder / "not-created").exists())
                no_data.assert_not_called()

    @patch("xai_metric_behavior.exp01r.plot_real_results")
    def test_public_runner_records_actual_calls_intercept_and_direct_answer(self, _plot):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = self.config(folder)
            with patch("xai_metric_behavior.exp01r.prepare_adult", return_value=self.fixture()):
                result = run_exp01r(path, folder / "run", folder / "data")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["counts"]["n_public_FC_calls"], 96)
            self.assertEqual(result["counts"]["n_public_FE_calls"], 12)
            self.assertEqual(result["counts"]["n_exception_scores"], 0)
            self.assertTrue(result["validation"]["all_nonexception_FC_calls_paired"])
            self.assertEqual(result["contract"]["model_intercept"], self.fixture().bias)
            self.assertEqual(result["validation"]["reserved_records_evaluated"], 0)
            calls = [json.loads(line) for line in (folder / "run/metric_calls.jsonl").read_text().splitlines()]
            self.assertEqual(len(calls), 108)
            self.assertTrue(all(call["public_api"].startswith("quantus.") for call in calls))
            with np.load(folder / "run/traces/input_0000.npz") as trace:
                self.assertEqual(trace["explanation_names"].tolist(), ["IG", "A", "B"])
                self.assertTrue((trace["mask_coverage"] == 10).all())
                np.testing.assert_array_equal(trace["subsets_k1"], trace["subsets_k5"][:, :, :1])
            with (folder / "run/score_summary.csv").open() as stream:
                names = {row["explanation"] for row in csv.DictReader(stream)}
            self.assertEqual(names, {"IG", "A", "B"})
            answer = json.loads((folder / "run/rq_answer.json").read_text())
            self.assertEqual(answer["status"], "observed_smoke_not_confirmation")
            self.assertEqual(answer["enumeration_n_valid_comparisons"], 6)
            self.assertEqual(len(answer["matched_budget_error"]), 2)
            self.assertIn("RMSE", (folder / "run/rq_answer.txt").read_text())
            with self.assertRaises(FileExistsError):
                run_exp01r(path, folder / "run", folder / "data")

    @patch("xai_metric_behavior.exp01r.plot_real_results")
    def test_real_upstream_all_negative_rejections_are_retained_without_resampling(self, _plot):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = self.config(folder)
            with patch("xai_metric_behavior.exp01r.prepare_adult", return_value=self.fixture(reject_first=True)):
                result = run_exp01r(path, folder / "run", folder / "data")
            self.assertEqual(result["status"], "completed")
            counts = result["counts"]
            self.assertEqual(counts["n_attempted_scores"], 96)
            self.assertEqual(counts["n_exception_scores"], 24)
            self.assertEqual(counts["n_FE_exceptions"], 3)
            self.assertEqual(counts["n_public_validation_exceptions"], 27)
            self.assertEqual(result["validation"]["n_public_calls_without_exception"], 81)
            self.assertTrue(result["validation"]["all_nonexception_public_calls_trace_checked"])
            with (folder / "run/scores.csv").open() as stream:
                rejected = [row for row in csv.DictReader(stream) if row["input_id"] == "0"]
            self.assertEqual(len(rejected), 24)
            self.assertTrue(all("less than zero" in row["exception"] for row in rejected))
            self.assertTrue(all(row["analysis_eligible"] == "False" for row in rejected))
            with np.load(folder / "run/traces/input_0000.npz") as trace:
                self.assertFalse(trace["mask_coverage"].any())
                self.assertTrue((trace["subsets_k1"] == -1).all())
            answer = json.loads((folder / "run/rq_answer.json").read_text())
            self.assertEqual(answer["matched_budget_error"][0]["n_input_repetitions_joint_all_settings"], 6)

    @patch("xai_metric_behavior.exp01r.plot_real_results")
    def test_pairing_guard_saves_official_return_and_partial_trace_before_halting(self, _plot):
        seen = 0

        def tamper_second_call(kind, *args, **kwargs):
            nonlocal seen
            result = public_score(kind, *args, **kwargs)
            if kind == "FC":
                seen += 1
                if seen == 2:
                    result.trace["subsets"] = result.trace["subsets"][:, ::-1]
            return result

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = self.config(folder)
            with patch("xai_metric_behavior.exp01r.prepare_adult", return_value=self.fixture()), \
                    patch("xai_metric_behavior.exp01r.public_score", side_effect=tamper_second_call):
                with self.assertRaisesRegex(RuntimeError, "paired k/B/candidate prefixes"):
                    run_exp01r(path, folder / "failed", folder / "data")
            manifest = json.loads((folder / "failed/manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["counts"]["n_attempted_scores"], 2)
            self.assertEqual(manifest["counts"]["n_exception_scores"], 1)
            with (folder / "failed/scores.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-1]["valid"], "True")
            self.assertEqual(rows[-1]["analysis_eligible"], "False")
            self.assertEqual(rows[-1]["exception_stage"], "trace_pairing")
            self.assertFalse((folder / "failed/rq_answer.json").exists())
            self.assertTrue((folder / "failed/traces/input_0000.npz").exists())

    def test_budget_error_uses_common_eligible_repeats_and_equal_input_weights(self):
        # A warned but finite huge error cannot enter one B while its counterpart
        # enters another. Unequal remaining repeat counts must not weight inputs.
        scores = np.zeros((2, 3, 1, 2, 3))
        scores[..., 0] = 1
        scores[0, :, 0, 0, 1:3] = [[.1, .1], [.9, .9], [.9, .9]]
        scores[0, :, 0, 1, 1:3] = [[.2, .2], [.8, .8], [.8, .8]]
        scores[1, :, 0, 0, 1:3] = .5
        scores[1, :, 0, 1, 1:3] = .4
        reasons = np.full(scores.shape, "valid", dtype="U64")
        reasons[0, 1:, 0, 0, 1:3] = "near_constant_input"
        eligible = reasons == "valid"
        result = analyze_real_scores(
            scores, reasons, np.array([[1., 0., 0.], [1., 0., 0.]]), [1], [5, 25],
            row_ids=np.array([10, 20]), tie_tolerance=.01, bootstrap_resamples=20,
            bootstrap_confidence=.95, analysis_seed=201, analysis_eligible=eligible,
        )
        first, second = result["rq_budget_summary"]
        self.assertEqual(first["n_input_repetitions_joint_all_settings"], 4)
        self.assertAlmostEqual(first["rmse_nonexact_controls_given_joint_all_settings"], np.sqrt((.1 ** 2 + .5 ** 2) / 2))
        self.assertAlmostEqual(second["rmse_nonexact_controls_given_joint_all_settings"], np.sqrt((.2 ** 2 + .4 ** 2) / 2))
        self.assertEqual(result["score_summary"][0]["explanation"], "IG")


if __name__ == "__main__":
    unittest.main()
