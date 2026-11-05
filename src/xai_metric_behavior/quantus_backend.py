"""Primary EXP-01 scores from the unmodified Quantus 0.6.0 public API.

Wrappers only observe official replacement and Pearson calls. The NumPy reference
is an independent check; it never supplies a returned experimental score.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import quantus
import torch
from quantus.functions.perturb_func import batch_baseline_replacement_by_indices
from quantus.functions.similarity_func import correlation_pearson
from quantus.helpers.model.pytorch_model import PyTorchModel

from .core import checked_pearson

METRIC_VARIANT = "exp01-quantus-public-0.6.0-v1"
QUANTUS_SOURCE_COMMIT = "85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a"
_QUANTUS_LOCK = threading.RLock()
_EXPECTED_BLOBS = {
    "metrics/faithfulness/faithfulness_correlation.py": "fe38df61fe0f09a94bb33b91b3cb772c0d97689f",
    "metrics/faithfulness/faithfulness_estimate.py": "4cdf7aa209e34ee97b702927b3527194f2599c4f",
    "functions/perturb_func.py": "15d1c07746bef0d6e20da64218422224feeafb2f",
    "functions/similarity_func.py": "e408818c523107047cfd60c79cb7f3bd4e4a6e5a",
    "metrics/base.py": "521592ce2e06ed3b6a441f2fdcd5bf8e1ec2c0ec",
    "helpers/asserts.py": "401f5f568cf136814b2ddec6a58e0b26d79e2d82",
    "helpers/utils.py": "3f072e30c2e99e0352f9f969a9ce1ec98d058b67",
    "helpers/model/pytorch_model.py": "5c61a8335a67d617c1790e932f0a5455e0129143",
}


@lru_cache(maxsize=1)
def backend_provenance() -> dict[str, Any]:
    """Hash the installed source actually executing; no network is needed."""
    package = Path(quantus.__file__).resolve().parent
    versions = {name: importlib.metadata.version(name) for name in ("quantus", "torch", "numpy", "scipy")}
    if versions["quantus"] != "0.6.0":
        raise RuntimeError("EXP-01 requires Quantus 0.6.0; do not silently change the evaluator")
    sources, verified_blobs = {}, {}
    for name, expected in _EXPECTED_BLOBS.items():
        data = (package / name).read_bytes()
        sources[name] = hashlib.sha256(data).hexdigest()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned Quantus source mismatch: {name}; re-audit before running")
        verified_blobs[name] = actual
    payload = {"metric_variant": METRIC_VARIANT, "versions": versions, "source_sha256": sources}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "execution_sha256": fingerprint,
            "upstream_commit": QUANTUS_SOURCE_COMMIT, "verified_git_blobs": verified_blobs,
            "public_api": ["quantus.FaithfulnessCorrelation.__call__", "quantus.FaithfulnessEstimate.__call__"],
            "perturb_func": "quantus.functions.perturb_func.batch_baseline_replacement_by_indices",
            "similarity_func": "quantus.functions.similarity_func.correlation_pearson",
            "precision": "CPU float64; scoped torch.set_default_dtype(torch.float64); original dtype restored in finally",
            "rng": "Scoped legacy NumPy MT19937; original state restored in finally; serialized by a process-local RLock",
            "thread_contract": "Run this experiment serially; unrelated threads must not mutate NumPy global RNG or Torch default dtype.",
            "model_cache_lifecycle": "Clear pinned PyTorchModel introspection caches when each locked public-call context exits; official returns and extracted traces are unchanged.",
            "score_source": "Actual public metric(...) return, never replaced by a reference calculation"}


def evaluator_seed_words(root: int, phase: int, input_id: int, repetition: int) -> np.ndarray:
    return np.random.SeedSequence([root, phase, input_id, repetition]).generate_state(4, dtype=np.uint32)


def _release_quantus_model_caches() -> None:
    """Release wrapper references held by the pinned upstream metadata caches.

    Quantus 0.6.0 decorates these instance methods with unbounded ``lru_cache``.
    Their ``self`` keys retain every fresh wrapper, our model, and its prediction
    logs after a metric call. Clearing cache entries does not change either
    method, the official calculation, or the extracted trace. The pinned source
    audit verifies the private methods we depend on. Call only while holding
    ``_QUANTUS_LOCK``, after the public-call body (also on exceptions).
    """
    PyTorchModel._get_last_softmax_layer_index.cache_clear()
    PyTorchModel._get_model_with_linear_top.cache_clear()


@contextmanager
def quantus_call_context(seed_words: np.ndarray) -> Iterator[None]:
    """Quantus uses the legacy global RNG and torch.Tensor uses default dtype.

    Set both through supported APIs, restore even on failure, serialize our calls.
    Release cached wrapper references before releasing the lock. Local model
    variables remain available to extract exception traces after context exit.
    No Quantus method, sampler, validator, or input array is patched.
    """
    with _QUANTUS_LOCK:
        state, dtype = np.random.get_state(), torch.get_default_dtype()
        try:
            np.random.seed(np.asarray(seed_words, dtype=np.uint32))
            torch.set_default_dtype(torch.float64)
            yield
        finally:
            try:
                _release_quantus_model_caches()
            finally:
                np.random.set_state(state)
                torch.set_default_dtype(dtype)


class ScalarLogitModel(torch.nn.Module):
    """One-column scalar logit. Record what the official adapter really sees."""
    def __init__(self, weights: np.ndarray, bias: float = 0.0):
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(np.array(weights, dtype=np.float64), dtype=torch.float64))
        if not np.isfinite(bias):
            raise ValueError("the scalar model bias must be finite")
        self.bias = float(bias)
        self.inputs: list[np.ndarray] = []
        self.outputs: list[np.ndarray] = []
        self.input_dtypes: list[str] = []
        self.eval()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.input_dtypes.append(str(value.dtype))
        if value.dtype != torch.float64 or value.device.type != "cpu":
            raise RuntimeError("the official model adapter must deliver CPU float64 inputs")
        flat = value.reshape(value.shape[0], -1)
        self.inputs.append(flat.detach().cpu().numpy().copy())
        result = (flat * self.weights).sum(dim=1, keepdim=True)
        # Preserve the original EXP-01 arithmetic when bias is exactly zero.
        # EXP-01-R must expose the learned log-odds, including its intercept.
        if self.bias != 0.0:
            result = result + self.bias
        self.outputs.append(result.detach().cpu().numpy().copy())
        return result


class _ReplacementLog:
    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def __call__(self, arr: np.ndarray, indices: np.ndarray, **kwargs: Any) -> np.ndarray:
        result = batch_baseline_replacement_by_indices(arr=arr, indices=indices, **kwargs)
        self.calls.append({"before": np.array(arr, copy=True), "indices": np.array(indices, copy=True),
                           "after": np.array(result, copy=True)})
        return result


class _PearsonLog:
    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def __call__(self, a: np.ndarray, b: np.ndarray, **kwargs: Any) -> Any:
        result = correlation_pearson(a=a, b=b, **kwargs)
        self.calls.append({"U": np.array(a, copy=True), "V": np.array(b, copy=True),
                           "returned": np.asarray(result).copy()})
        return result


@dataclass
class PublicScore:
    score: float
    valid: bool
    reason: str
    trace: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)
    exception: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)
    analysis_eligible: bool = False
    exception_stage: str | None = None


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_trace(model: ScalarLogitModel, replacement: _ReplacementLog, pearson: _PearsonLog,
                    x: np.ndarray, a: np.ndarray, baseline: np.ndarray, expected_draws: int,
                    subset_size: int, score: float, tolerance: float,
                    near_constant: bool = False) -> tuple[dict[str, Any], dict[str, Any], str]:
    if len(replacement.calls) != expected_draws or len(pearson.calls) != 1:
        raise RuntimeError("unexpected public Quantus perturbation/Pearson call count")
    if len(model.inputs) != expected_draws + 1:
        raise RuntimeError("unexpected public Quantus prediction call count")
    if not np.array_equal(model.inputs[0], x[None]):
        raise RuntimeError("the public model adapter changed the canonical input")
    subsets = np.concatenate([call["indices"] for call in replacement.calls], axis=0)
    if subsets.shape != (expected_draws, subset_size):
        raise RuntimeError("unexpected native subset shape")
    for row in subsets:
        if len(np.unique(row)) != subset_size or np.any(row < 0) or np.any(row >= len(x)):
            raise RuntimeError("native subset is not a valid sample without replacement")
    actual_after = []
    for draw, call in enumerate(replacement.calls):
        expected = x.copy()
        expected[subsets[draw]] = baseline[subsets[draw]]
        if not np.array_equal(call["before"], x[None]):
            raise RuntimeError("replacement was cumulative or its original input changed")
        if not np.array_equal(call["after"], expected[None]):
            raise RuntimeError("official replacement did not apply the declared literal baseline")
        if not np.array_equal(model.inputs[draw + 1], expected[None]):
            raise RuntimeError("the model adapter changed a perturbed input")
        actual_after.append(call["after"][0])
    u, v = pearson.calls[0]["U"].reshape(-1), pearson.calls[0]["V"].reshape(-1)
    original = float(model.outputs[0][0, 0])
    outputs = np.array([item[0, 0] for item in model.outputs[1:]])
    expected_u, expected_v = a[subsets].sum(axis=1), original - outputs
    if not np.array_equal(u, expected_u) or not np.array_equal(v, expected_v):
        raise RuntimeError("logged Pearson U/V differ from actual attributions and model responses")
    reference = checked_pearson(u, v)
    # A library NearConstantInputWarning means roundoff can dominate Pearson.
    # Keep the official value and diagnostic replay error, but do not assert
    # numerical agreement for this ill-conditioned case or analyze it as an
    # ordinary finite estimator. Shape, replacements and U/V checks still apply.
    if not near_constant and bool(np.isfinite(score)) != reference.valid:
        raise RuntimeError("public score validity disagrees with its logged Pearson inputs")
    if not near_constant and reference.valid and abs(score - reference.score) > tolerance:
        raise RuntimeError("public score disagrees with independent trace replay")
    if not np.allclose(outputs, np.sum(np.array(actual_after) * model.weights.numpy(), axis=1) + model.bias,
                       atol=tolerance, rtol=0):
        raise RuntimeError("actual model responses differ from the declared scalar logit")
    trace = {"subsets": subsets, "U": u, "V": v, "original_output": original,
             "perturbed_outputs": outputs, "model_inputs": np.concatenate(model.inputs, axis=0),
             "input_dtypes": model.input_dtypes, "mask_sha256": _array_hash(subsets),
             "uv_sha256": _array_hash(u, v), "model_input_sha256": _array_hash(*model.inputs)}
    checks = {"n_predictions": len(model.inputs), "n_perturbations": len(replacement.calls),
              "n_pearson_calls": len(pearson.calls), "dtype": "torch.float64", "input_exactly_preserved": True,
              "original_replacement_verified": True,
              "trace_replay_absolute_error": abs(score - reference.score) if reference.valid and np.isfinite(score) else None,
              "score_parity_status": "not_asserted_near_constant" if near_constant else "passed",
              "near_constant_warning": near_constant,
              "n_unchanged_draws": sum(np.array_equal(item, x) for item in actual_after)}
    reason = ("near_constant_input" if np.isfinite(score) else "near_constant_nonfinite") if near_constant else (
        "valid" if reference.valid else reference.reason)
    return trace, checks, reason


def public_score(metric_name: str, x: np.ndarray, weights: np.ndarray, attribution: np.ndarray,
                 seed_words: np.ndarray, *, subset_size: int | None = None, budget: int | None = None,
                 baseline: float = 0.0, tolerance: float = 1e-10, bias: float = 0.0) -> PublicScore:
    """Execute one input/explanation using public metric(...), logging all steps.

    FC's own np.random.choice selects masks. FE's own signed sort selects singleton
    order. Exceptions and nonfinite returns remain explicit, with no fallback.
    """
    backend_provenance()
    if baseline != 0.0:
        raise ValueError("use literal numeric zero, not a named baseline preset")
    x, a = np.asarray(x, dtype=np.float64), np.asarray(attribution, dtype=np.float64)
    if x.ndim != 1 or a.shape != x.shape:
        raise ValueError("single-input vectors of matching shape are required")
    if metric_name not in ("FC", "FE"):
        raise ValueError("metric_name must be FC or FE")
    if metric_name == "FC" and (subset_size is None or budget is None):
        raise ValueError("FC requires subset_size and budget")
    replacement, pearson = _ReplacementLog(), _PearsonLog()
    model = ScalarLogitModel(weights, bias=bias)
    recorded: list[Any] = []
    result = PublicScore(float("nan"), False, "not_attempted")
    stage = "official_call"
    try:
        with quantus_call_context(seed_words), warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            settings = dict(abs=False, normalise=False, perturb_baseline=0.0,
                            perturb_func=replacement, similarity_func=pearson,
                            return_aggregate=False, disable_warnings=True, display_progressbar=False)
            if metric_name == "FC":
                metric = quantus.FaithfulnessCorrelation(nr_runs=budget, subset_size=subset_size, **settings)
                draws, k = int(budget), int(subset_size)
            else:
                metric = quantus.FaithfulnessEstimate(features_in_step=1, **settings)
                draws, k = len(x), 1
            # Keep this actual public invocation: preprocess/validation/wrapping run.
            returned = metric(model=model, x_batch=x.reshape(1, 1, -1), y_batch=np.array([0]),
                              a_batch=a.reshape(1, 1, -1), channel_first=True,
                              softmax=False, device="cpu", batch_size=1)
            result.score = float(returned[0])
            near_constant = any(item.category.__name__ == "NearConstantInputWarning" for item in recorded)
            stage = "trace_validation"
            result.trace, result.checks, result.reason = _validate_trace(
                model, replacement, pearson, x, a, np.zeros_like(x), draws, k, result.score, tolerance,
                near_constant=near_constant)
            result.valid = bool(np.isfinite(result.score))
            result.analysis_eligible = result.valid and not near_constant
    except Exception as error:
        result.valid = False
        result.exception = f"{type(error).__name__}: {error}"
        result.exception_stage = stage
        result.reason = f"exception:{type(error).__name__}"
        result.trace = {"model_inputs": np.concatenate(model.inputs, axis=0) if model.inputs else np.empty((0, len(x))),
                        "input_dtypes": model.input_dtypes,
                        "subsets": np.concatenate([item["indices"] for item in replacement.calls], axis=0) if replacement.calls else np.empty((0, 0), dtype=int),
                        "U": pearson.calls[0]["U"].reshape(-1) if pearson.calls else np.array([]),
                        "V": pearson.calls[0]["V"].reshape(-1) if pearson.calls else np.array([])}
    finally:
        result.warnings = [{"category": item.category.__name__, "message": str(item.message)} for item in recorded]
    return result


def verify_pairing(reference: dict[str, Any], current: dict[str, Any], *, budget: int, subset_size: int) -> None:
    """Compare masks observed in two real public calls, not just their seeds."""
    expected = reference["subsets"][:budget, :subset_size]
    if not np.array_equal(current["subsets"], expected):
        raise RuntimeError("actual Quantus masks violate the declared k/budget/explanation prefix pairing")
