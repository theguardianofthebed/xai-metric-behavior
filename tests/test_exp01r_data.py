"""Offline contract regressions; no Adult outcome or reserved record is used."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from xai_metric_behavior.exp01r_data import (
    DATASET_ID, FEATURE_NAMES, IG_PARAMETERS, MODEL_PARAMETERS, PARTITION_SIZES,
    FrozenLogOdds, PreparationError, SourceFile, _load_or_fit_model, _load_or_reserve_partition,
    fetch_source, fit_train_pipeline, ig_diagnostics, make_partitions, prepare_adult,
    read_adult_csv, transform_with_saved_scaler,
)


def config() -> dict:
    return {
        "phase": "smoke", "n_inputs": 4,
        "seeds": {"split": 2026110501, "model": 2026110502},
        "data": {"dataset": DATASET_ID, "preprocessing": "standard_scaler_train_only",
                 "confirmation_size": 512},
        "model": dict(MODEL_PARAMETERS), "ig": dict(IG_PARAMETERS),
    }


def training_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(81)
    x = rng.normal(size=(80, len(FEATURE_NAMES))) + np.arange(len(FEATURE_NAMES))
    y = (0.8 * x[:, 0] - 0.2 * x[:, 1] > 0).astype(np.int64)
    return x, y


class AdultContractTests(unittest.TestCase):
    def test_reservations_are_disjoint_and_do_not_change_global_rng(self):
        before = np.random.get_state()
        first = make_partitions(9045, 2026110501)
        second = make_partitions(9045, 2026110501)
        self.assertEqual(first, second)
        self.assertEqual(first["sizes"], PARTITION_SIZES)
        groups = [set(first["ids"][phase]) for phase in PARTITION_SIZES]
        self.assertEqual(sum(map(len, groups)), len(set.union(*groups)))
        self.assertTrue(all(0 <= i < 9045 for group in groups for i in group))
        self.assertNotEqual(first["partition_sha256"], make_partitions(9045, 2026110502)["partition_sha256"])
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_confirmation_is_blocked_before_source_access(self):
        settings = config()
        settings["phase"], settings["n_inputs"] = "confirmation", 512
        with tempfile.TemporaryDirectory() as folder, patch(
            "xai_metric_behavior.exp01r_data.fetch_source"
        ) as download:
            output = Path(folder) / "output"
            with self.assertRaisesRegex(PreparationError, "Confirmation is reserved"):
                prepare_adult(settings, Path(folder) / "data", output)
            download.assert_not_called()
            failure = json.loads((output / "data_preparation_failure.json").read_text())
            self.assertEqual(failure["stage"], "configuration")

    def test_reservation_cache_rejects_changed_seed_and_corrupted_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            first, reused = _load_or_reserve_partition(9045, config(), path)
            self.assertFalse(reused)
            before = (path / "partition.json").read_bytes()
            development = config()
            development["phase"], development["n_inputs"] = "development", 128
            second, reused = _load_or_reserve_partition(9045, development, path)
            self.assertTrue(reused)
            self.assertEqual(first, second)
            development["seeds"]["split"] += 1
            with self.assertRaisesRegex(PreparationError, "seed/source/settings/IDs differ"):
                _load_or_reserve_partition(9045, development, path)
            self.assertEqual((path / "partition.json").read_bytes(), before)
            altered = json.loads(before)
            altered["ids"]["smoke"][0] = altered["ids"]["confirmation"][0]
            (path / "partition.json").write_text(json.dumps(altered))
            with self.assertRaisesRegex(PreparationError, "integrity"):
                _load_or_reserve_partition(9045, config(), path)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "fitted_model.json").write_text("{}")
            with self.assertRaisesRegex(PreparationError, "reservation is missing"):
                _load_or_reserve_partition(9045, config(), path)

    def test_source_checksum_and_schema_fail_without_silent_replacement(self):
        rows = [
            "20,100,10,0,0,40,1,1,0,0,0,1,1,0",
            "45,200,12,0,0,50,0,0,1,1,1,0,0,1",
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.csv"
            data = (",".join([*FEATURE_NAMES, "income"]) + "\n" + "\n".join(rows) + "\n").encode()
            path.write_bytes(data)
            spec = SourceFile(path.name, 2, len(data), hashlib.sha256(data).hexdigest())
            x, y = read_adult_csv(path, spec)
            self.assertEqual(x.shape, (2, 13))
            np.testing.assert_array_equal(y, [0, 1])
            corrupted = data.replace(b"age,", b"AGE,", 1)
            path.write_bytes(corrupted)
            with patch("urllib.request.urlopen") as download:
                with self.assertRaisesRegex(PreparationError, "Source identity mismatch"):
                    fetch_source(spec, Path(folder))
                download.assert_not_called()
            self.assertEqual(path.read_bytes(), corrupted)
            # Even a verified file hash cannot silently change the feature meaning/order.
            altered_spec = SourceFile(path.name, 2, len(corrupted), hashlib.sha256(corrupted).hexdigest())
            with self.assertRaisesRegex(PreparationError, "column order"):
                read_adult_csv(path, altered_spec)

    def test_scaler_is_fit_on_training_rows_and_positive_class_is_preserved(self):
        train_x, train_y = training_fixture()
        artifact = fit_train_pipeline(train_x, train_y, 2026110502)
        np.testing.assert_allclose(artifact["scaler"]["mean"], train_x.mean(axis=0), atol=1e-12)
        self.assertEqual(artifact["scaler"]["n_samples_seen"], len(train_x))
        self.assertTrue(artifact["training"]["positive_class_orientation_checked"])
        self.assertEqual(artifact["model"]["classes"], [0, 1])
        evaluation = train_x[:4] + 20.0
        saved_mean = np.asarray(artifact["scaler"]["mean"]).copy()
        transformed = transform_with_saved_scaler(evaluation, artifact)
        self.assertGreater(float(np.abs(transformed.mean(axis=0)).min()), 5.0)
        np.testing.assert_array_equal(artifact["scaler"]["mean"], saved_mean)
        self.assertEqual(artifact["training"]["evaluation_records_used_for_fitting"], 0)

    def test_one_fitted_model_is_reused_and_mismatches_stop(self):
        train_x, train_y = training_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            first, reused = _load_or_fit_model(train_x, train_y, config(), path)
            self.assertFalse(reused)
            development = config()
            development["phase"], development["n_inputs"] = "development", 128
            with patch("xai_metric_behavior.exp01r_data.fit_train_pipeline") as fit:
                second, reused = _load_or_fit_model(train_x, train_y, development, path)
                self.assertTrue(reused)
                self.assertEqual(first["model_sha256"], second["model_sha256"])
                fit.assert_not_called()
                changed = config()
                changed["seeds"]["model"] += 1
                with self.assertRaisesRegex(PreparationError, "differ"):
                    _load_or_fit_model(train_x, train_y, changed, path)
                altered = json.loads((path / "fitted_model.json").read_text())
                altered["model"]["weights"][0] += 1.0
                (path / "fitted_model.json").write_text(json.dumps(altered))
                with self.assertRaisesRegex(PreparationError, "integrity"):
                    _load_or_fit_model(train_x, train_y, config(), path)
                fit.assert_not_called()

    def test_development_cannot_silently_fit_another_model(self):
        train_x, train_y = training_fixture()
        settings = config()
        settings["phase"], settings["n_inputs"] = "development", 128
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(PreparationError, "reuse its reservation"):
                _load_or_reserve_partition(9045, settings, Path(folder))
            with self.assertRaisesRegex(PreparationError, "Run smoke first"):
                _load_or_fit_model(train_x, train_y, settings, Path(folder))

    def test_completeness_alone_does_not_pass_coordinate_errors(self):
        reference = np.array([[1.0, -2.0, 3.0]])
        cancelling_errors = reference + np.array([[0.1, -0.1, 0.0]])
        checks = ig_diagnostics(cancelling_errors, reference, np.array([0.0]),
                               np.array([2.0]), atol=1e-7, rtol=1e-7)
        self.assertTrue(checks[0]["completeness_gate_passed"])
        self.assertFalse(checks[0]["coordinate_gate_passed"])
        self.assertFalse(checks[0]["passed"])

    def test_actual_captum_ig_includes_bias_in_output_but_not_contributions(self):
        from captum.attr import IntegratedGradients

        weights = np.array([1.2, -0.5, 2.0])
        x = np.array([[2.0, 1.0, -1.0], [-0.25, 2.0, 3.0]])
        model = FrozenLogOdds(weights, bias=3.25)
        inputs = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        actual, delta = IntegratedGradients(model, multiply_by_inputs=True).attribute(
            inputs, baselines=torch.zeros_like(inputs), n_steps=50,
            method="gausslegendre", return_convergence_delta=True,
        )
        ig = actual.detach().numpy()
        reference = x * weights
        output_difference = (model(inputs) - model(torch.zeros_like(inputs))).detach().numpy().ravel()
        np.testing.assert_allclose(model(inputs).detach().numpy().ravel(), reference.sum(axis=1) + 3.25)
        checks = ig_diagnostics(ig, reference, delta.detach().numpy(), output_difference,
                               atol=1e-7, rtol=1e-7)
        self.assertTrue(all(row["passed"] for row in checks))
        for shift in (1, 2):
            permuted = np.roll(ig, shift, axis=1)
            np.testing.assert_array_equal(np.sort(permuted, axis=1), np.sort(ig, axis=1))
            self.assertFalse(np.array_equal(permuted, ig))


if __name__ == "__main__":
    unittest.main()
