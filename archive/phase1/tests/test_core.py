import numpy as np
import pandas as pd
import torch

from xai_metric_behavior.core import (
    make_oracle_data,oracle_attributions,relocate_to_decoys,
    support_size_explanation,concentration_explanation,behavior_profiles
)
from xai_metric_behavior.models import OracleLinearClassifier


def test_oracle_exactness():
    d = make_oracle_data(128,20,[2.5,-2,1.5,1,-0.5],seed=1)
    m = OracleLinearClassifier(20,d.coefficients)
    with torch.no_grad():
        pred = m(torch.tensor(d.x)).argmax(1).numpy()
    assert np.array_equal(pred,d.y)
    a = oracle_attributions(d.x,d.y,d.coefficients)
    target_logit = np.where(d.y==1,d.z,-d.z)
    assert np.allclose(a.sum(1),target_logit,atol=1e-5)


def test_faithfulness_intervention_preserves_l1_and_support():
    rng = np.random.default_rng(0)

    a = np.zeros((10, 20), dtype=np.float32)
    a[:, :5] = rng.normal(size=(10, 5))

    original_l1 = np.abs(a).sum(axis=1)
    original_l0 = np.count_nonzero(a, axis=1)

    for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        out = relocate_to_decoys(
            a,
            level,
            informative_features=5,
            decoy_start=5,
        )

        # Total attribution magnitude must remain unchanged.
        assert np.allclose(
            np.abs(out).sum(axis=1),
            original_l1
        )

        # Number of non-zero attribution values must remain unchanged.
        assert np.array_equal(
            np.count_nonzero(out, axis=1),
            original_l0
        )


def test_compactness_generators():
    for k in [1,2,5,20]:
        a=support_size_explanation(20,k)
        assert np.isclose(a.sum(),1)
        assert np.count_nonzero(a)==k
    for beta in [0,.2,2]:
        a=concentration_explanation(20,beta)
        assert np.isclose(a.sum(),1)
        assert np.count_nonzero(a)==20


def test_response_alignment():
    s=pd.DataFrame({
        "phase":["x"]*3,"property":["f"]*3,"intervention":["i"]*3,
        "level":[0,.5,1],"metric_id":["m"]*3,"metric_name":["M"]*3,
        "score_direction":["higher"]*3,"quality_direction":[-1]*3,
        "score_mean":[1,.5,0]
    })
    p=behavior_profiles(s)
    assert p.iloc[0]["response_alignment"]>.99
