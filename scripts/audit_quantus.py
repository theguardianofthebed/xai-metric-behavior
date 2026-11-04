"""Auxiliary source and NumPy-reference audit of Quantus 0.6.0.

EXP-01 reports official public Quantus scores. These fixed fixtures separately
check source identity, utility semantics and the auxiliary NumPy reference.
They do not replace the main Quantus run or its full recorded-trace checks.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

QUANTUS_COMMIT = "85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a"
EXPECTED_BLOBS = {
    "metrics/faithfulness/faithfulness_correlation.py": "fe38df61fe0f09a94bb33b91b3cb772c0d97689f",
    "metrics/faithfulness/faithfulness_estimate.py": "4cdf7aa209e34ee97b702927b3527194f2599c4f",
    "functions/perturb_func.py": "15d1c07746bef0d6e20da64218422224feeafb2f",
    "functions/similarity_func.py": "e408818c523107047cfd60c79cb7f3bd4e4a6e5a",
    "metrics/base.py": "521592ce2e06ed3b6a441f2fdcd5bf8e1ec2c0ec",
    "helpers/asserts.py": "401f5f568cf136814b2ddec6a58e0b26d79e2d82",
    "helpers/utils.py": "3f072e30c2e99e0352f9f969a9ce1ec98d058b67",
    "helpers/model/pytorch_model.py": "5c61a8335a67d617c1790e932f0a5455e0129143",
}


@contextmanager
def saved_numpy_seed(seed):
    """Seed Quantus' legacy RNG locally and restore the caller's state."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class RecordingReplacement:
    """Record inputs/masks; delegate replacement to the unchanged function."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.inputs = []
        self.indices = []
        self.outputs = []

    def __call__(self, arr, indices, **kwargs):
        self.inputs.append(np.array(arr, copy=True))
        self.indices.append(np.array(indices, copy=True))
        result = self.delegate(arr=arr, indices=indices, **kwargs)
        self.outputs.append(np.array(result, copy=True))
        return result


class RecordingPearson:
    """Record actual U,V passed by Quantus; delegate its score calculation."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.u = None
        self.v = None

    def __call__(self, a, b, **kwargs):
        self.u, self.v = np.array(a, copy=True), np.array(b, copy=True)
        return self.delegate(a=a, b=b, **kwargs)


class NumpyScalarModel:
    """Small duck-typed adapter for the explicit evaluate_batch audit only."""

    def __init__(self, weights):
        self.weights = weights

    def shape_input(self, x, shape, channel_first=True, batched=True):
        return np.asarray(x).reshape(shape[0], -1)

    def predict(self, x):
        return (np.asarray(x).reshape(len(x), -1) @ self.weights)[:, None]


def _check(name, operation):
    try:
        details = operation()
        return {"name": name, "status": "passed", **(details or {})}
    except Exception as exc:
        return {"name": name, "status": "failed", "exception": type(exc).__name__, "reason": str(exc)}


def run_audit():
    """Return a JSON-safe audit; missing required packages are failures."""
    report = {
        "audit": "exp01-quantus-0.6.0-reference-audit-v2",
        "role": "auxiliary fixed-fixture/source verification; official Quantus is the primary experiment scorer",
        "quantus_source_commit": QUANTUS_COMMIT,
        "python": platform.python_version(),
        "checks": [],
        "limitations": [
            "Finite controlled fixtures check implementation compatibility, not general metric validity.",
            "No development or held-out confirmation experiment is executed by this script.",
            "No Quantus validation, scoring, RNG function, or installed source is monkeypatched.",
            "The dyadic fixture is exactly representable in float32/float64; production non-dyadic precision is checked by quantus_backend and its tests.",
        ],
    }
    if util.find_spec("quantus") is None:
        return {**report, "status": "failed", "reason": "Required quantus==0.6.0 is not installed."}
    try:
        import quantus
        from quantus.functions.perturb_func import batch_baseline_replacement_by_indices
        from quantus.functions.similarity_func import correlation_pearson
        from quantus.helpers import asserts
        from xai_metric_behavior.core import LinearOracle, checked_pearson, contribution_vectors, perturbation_trace
    except Exception as exc:
        return {**report, "status": "failed", "reason": f"Import failed: {type(exc).__name__}: {exc}"}

    report["packages"] = {name: metadata.version(name) for name in ("quantus", "numpy", "scipy")}
    checks = report["checks"]

    def source_check():
        if metadata.version("quantus") != "0.6.0":
            raise ValueError("This audit applies only to quantus==0.6.0.")
        package_root = Path(quantus.__file__).parent
        observed = {}
        for relative, expected in EXPECTED_BLOBS.items():
            data = (package_root / relative).read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            observed[relative] = actual
            if actual != expected:
                raise ValueError(f"Pinned source mismatch: {relative}: {actual} != {expected}")
        return {"verified_git_blobs": observed}

    checks.append(_check("pinned_source_identity", source_check))
    if checks[-1]["status"] == "failed":
        return {**report, "status": "failed"}

    # Dyadic input values are exactly representable in float32 and float64.
    # Quantus' public Torch wrapper converts input using torch.Tensor(x).
    x = np.array([1.5, 0.75, -0.5, 1.25, -1.75] + [0.125 * (j + 1) for j in range(15)])
    weights = np.array([2.5, -2.0, 1.5, 1.0, -0.5] + [0.0] * 15)
    baseline = np.zeros(20)
    oracle = LinearOracle(weights)
    candidates = contribution_vectors(x, weights, baseline)

    def utility_check():
        arr = np.array([[-2.0, 3.0, 4.0]])
        indices = np.array([[1]])
        zero = batch_baseline_replacement_by_indices(arr, indices, perturb_baseline=0.0)
        black = batch_baseline_replacement_by_indices(arr, indices, perturb_baseline="black")
        np.testing.assert_array_equal(arr, [[-2.0, 3.0, 4.0]])
        np.testing.assert_array_equal(zero, [[-2.0, 0.0, 4.0]])
        np.testing.assert_array_equal(black, [[-2.0, -2.0, 4.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            undefined = correlation_pearson(np.zeros(4), np.arange(4.0))
        if not np.isnan(undefined):
            raise AssertionError("Constant-vector Pearson should be NaN, not zero.")
        return {"literal_zero": zero.tolist(), "black_per_input_minimum": black.tolist(),
                "constant_pearson": None, "warnings": [str(w.message) for w in caught]}

    checks.append(_check("replacement_and_undefined_pearson", utility_check))

    def validator_check():
        try:
            asserts.assert_value_smaller_than_input_size(x[None, :], 5, "subset_size")
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected pinned 2-D validator defect was not reproduced.")
        asserts.assert_value_smaller_than_input_size(x[None, None, :], 5, "subset_size")
        return {"expected_upstream_failure": message, "accepted_shape": [1, 1, 20]}

    checks.append(_check("unpatched_shape_validation", validator_check))

    options = dict(abs=False, normalise=False, perturb_baseline=0.0, return_aggregate=False,
                   disable_warnings=True, display_progressbar=False)

    def parity_case(kind, name, k, public_model=None):
        replacement = RecordingReplacement(batch_baseline_replacement_by_indices)
        comparator = RecordingPearson(correlation_pearson)
        attribution = candidates[name]
        common = {**options, "perturb_func": replacement, "similarity_func": comparator}
        if kind == "FC":
            metric = quantus.FaithfulnessCorrelation(nr_runs=25, subset_size=k, **common)
        else:
            metric = quantus.FaithfulnessEstimate(features_in_step=1, **common)
        x_batch, a_batch = x[None, None, :], attribution[None, None, :]
        with saved_numpy_seed(20261104), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if public_model is None:
                actual = metric.evaluate_batch(model=NumpyScalarModel(weights), x_batch=x_batch,
                                               y_batch=np.array([0]), a_batch=a_batch)[0]
            else:
                actual = metric(model=public_model, x_batch=x_batch, y_batch=np.array([0]),
                                a_batch=a_batch, channel_first=True, softmax=False, device="cpu")[0]
        masks = np.stack([indices[0] for indices in replacement.indices])
        permutations = np.stack([np.r_[mask, np.setdiff1d(np.arange(20), mask)] for mask in masks])
        # Every retained prefix is exactly the mask used by the actual Quantus execution.
        trace = perturbation_trace(oracle, x, baseline, {name: attribution}, permutations, masks.shape[1])
        u = np.asarray(trace["U"])
        if u.ndim == 2:
            u = u[:, 0]
        v = np.asarray(trace["V"])
        expected = checked_pearson(u, v).score
        np.testing.assert_allclose(comparator.u[0], u, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(comparator.v[0], v, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, equal_nan=True)
        for arr in replacement.inputs:
            np.testing.assert_array_equal(arr[0], x)
        for mask, arr in zip(masks, replacement.outputs):
            expected_input = x.copy()
            expected_input[mask] = 0.0
            np.testing.assert_array_equal(arr[0], expected_input)
        if kind == "FE":
            population = checked_pearson(attribution, weights * x).score
            np.testing.assert_allclose(actual, population, atol=1e-12, rtol=1e-12)
            np.testing.assert_array_equal(np.sort(masks[:, 0]), np.arange(20))
        return {"metric": kind, "candidate": name, "k": masks.shape[1], "draws": len(masks),
                "quantus_score": float(actual), "numpy_reference_score": float(expected),
                "score_absolute_difference": float(abs(actual - expected)),
                "max_U_difference": float(np.max(np.abs(comparator.u[0] - u))),
                "max_V_difference": float(np.max(np.abs(comparator.v[0] - v))),
                "warnings": [str(w.message) for w in caught]}

    for kind, sizes in (("FC", (1, 5, 10, 19)), ("FE", (1,))):
        for k in sizes:
            for name in candidates:
                checks.append(_check(f"batch_{kind}_{name}_k{k}", lambda kind=kind, name=name, k=k:
                                     parity_case(kind, name, k)))

    if util.find_spec("torch") is not None:
        import torch
        report["packages"]["torch"] = metadata.version("torch")

        class TorchScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weights", torch.tensor(weights, dtype=torch.float64))

            def forward(self, inputs):
                flat = inputs.reshape(inputs.shape[0], -1).to(dtype=self.weights.dtype)
                return (flat @ self.weights)[:, None]

        public_model = TorchScalarModel().eval()
        for kind, sizes in (("FC", (1, 5, 10, 19)), ("FE", (1,))):
            for k in sizes:
                for name in candidates:
                    checks.append(_check(f"public_{kind}_{name}_k{k}", lambda kind=kind, name=name, k=k:
                                         parity_case(kind, name, k, public_model)))

        def full_api_2d_failure():
            metric = quantus.FaithfulnessCorrelation(nr_runs=25, subset_size=5, **options)
            try:
                metric(model=public_model, x_batch=x[None, :], a_batch=candidates["exact"][None, :],
                       y_batch=np.array([0]), channel_first=True, softmax=False, device="cpu")
            except ValueError as exc:
                if "subset_size" not in str(exc):
                    raise
                return {"expected_upstream_failure": str(exc)}
            raise AssertionError("Plain 2-D public FC unexpectedly succeeded; re-audit the source.")

        checks.append(_check("public_2d_expected_failure", full_api_2d_failure))
    else:
        checks.append({"name": "public_torch_API", "status": "failed", "reason": "Required torch is not installed."})

    failed = [c for c in checks if c["status"] == "failed"]
    skipped = [c for c in checks if c["status"] == "skipped"]
    report["status"] = "failed" if failed else "partial" if skipped else "passed"
    report["counts"] = {"attempted": len(checks), "passed": sum(c["status"] == "passed" for c in checks),
                        "failed": len(failed), "skipped": len(skipped)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "quantus_audit.json")
    args = parser.parse_args()
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Quantus audit: {report['status']} — {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
