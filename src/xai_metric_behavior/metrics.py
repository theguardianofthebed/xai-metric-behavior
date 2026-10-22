from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .core import flatten_numeric


@dataclass(frozen=True)
class MetricSpec:
    name: str
    property_name: str
    score_direction: str


SPECS = {
    "faithfulness_correlation": MetricSpec("Faithfulness Correlation","faithfulness","higher"),
    "monotonicity_correlation": MetricSpec("Monotonicity Correlation","faithfulness","higher"),
    "sensitivity_n": MetricSpec("Sensitivity-N","faithfulness","higher"),
    "complexity": MetricSpec("Complexity","compactness","lower"),
    "sparseness": MetricSpec("Sparseness","compactness","higher"),
    "effective_complexity": MetricSpec("Effective Complexity","compactness","lower"),
    "avg_sensitivity": MetricSpec("Avg Sensitivity","robustness","lower"),
    "max_sensitivity": MetricSpec("Max Sensitivity","robustness","lower"),
    "local_lipschitz": MetricSpec("Local Lipschitz Estimate","robustness","lower"),
}

def _patch_quantus_tabular_input_size_assert():
    """
    Compatibility patch for Quantus 0.6.0.

    Quantus 0.6.0 incorrectly performs an additional x.shape[2:]
    input-size check for 2D tabular arrays shaped (batch, features),
    which makes the input size appear as 1.

    This patch only corrects the input-size validation logic.
    It does not modify any metric computation.
    """
    from quantus.helpers import asserts

    # Avoid patching multiple times.
    if getattr(
        asserts.assert_value_smaller_than_input_size,
        "_xai_metric_behavior_patched",
        False,
    ):
        return

    def fixed_assert_value_smaller_than_input_size(
        x,
        value,
        value_name,
    ):
        x = np.asarray(x)

        if x.ndim == 2:
            # Tabular:
            # (batch, features)
            input_size = int(np.prod(x.shape[1:]))

        else:
            # Image / higher-dimensional data:
            # batch and channel are excluded.
            input_size = int(np.prod(x.shape[2:]))

        if value >= input_size:
            raise ValueError(
                f"'{value_name}' must be smaller than input size. "
                f"[{value} >= {input_size}]"
            )

    fixed_assert_value_smaller_than_input_size._xai_metric_behavior_patched = True

    asserts.assert_value_smaller_than_input_size = (
        fixed_assert_value_smaller_than_input_size
    )

def build_metric(metric_id, params):
    try:
        import quantus
    except ImportError as e:
        raise ImportError("Install requirements-experiment.txt first.") from e

    _patch_quantus_tabular_input_size_assert()
    p = dict(params)
    p.pop("softmax", None)
    cls = {
        "faithfulness_correlation": quantus.FaithfulnessCorrelation,
        "monotonicity_correlation": quantus.MonotonicityCorrelation,
        "sensitivity_n": quantus.SensitivityN,
        "complexity": quantus.Complexity,
        "sparseness": quantus.Sparseness,
        "effective_complexity": quantus.EffectiveComplexity,
        "avg_sensitivity": quantus.AvgSensitivity,
        "max_sensitivity": quantus.MaxSensitivity,
        "local_lipschitz": quantus.LocalLipschitzEstimate,
    }[metric_id]
    return cls(**p)


def evaluate_metric(metric_id, params, *, model, x_batch, y_batch,
                    a_batch=None, explain_func=None, device="cpu"):
    metric = build_metric(metric_id, params)
    kwargs = dict(
        model=model,
        x_batch=np.asarray(x_batch),
        y_batch=np.asarray(y_batch),
        device=device,
        batch_size=min(64, len(x_batch)),
    )
    if a_batch is not None:
        kwargs["a_batch"] = np.asarray(a_batch)
    if explain_func is not None:
        kwargs["explain_func"] = explain_func
    if "softmax" in params:
        kwargs["softmax"] = bool(params["softmax"])
    return flatten_numeric(metric(**kwargs))
