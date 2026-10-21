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


def test_faithfulness_intervention_preserves_l1():
    rng=np.random.default_rng(0)
    a=np.zeros((10,20),dtype=np.float32)
    a[:,:5]=rng.normal(size=(10,5))
    for level in [0,.1,.5,1]:
        out=relocate_to_decoys(a,level,5,5)
        assert np.allclose(np.abs(out).sum(1),np.abs(a).sum(1))


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
