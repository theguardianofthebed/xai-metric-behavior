from __future__ import annotations
from pathlib import Path
import importlib.metadata as md
import json
import platform
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from .core import (
    make_oracle_data, oracle_attributions, relocate_to_decoys,
    support_size_explanation, concentration_explanation,
    summarize, behavior_profiles, config_hash,
)
from .models import OracleLinearClassifier, MLPClassifier, train_mlp
from .explainers import ControlledUnstableOracleExplainer, CaptumTabularExplainer
from .metrics import SPECS, evaluate_metric


def _rows(phase, prop, intervention, level, metric_id, scores, quality_direction):
    s = SPECS[metric_id]
    return [{
        "phase": phase,
        "property": prop,
        "intervention": intervention,
        "level": float(level),
        "metric_id": metric_id,
        "metric_name": s.name,
        "score_direction": s.score_direction,
        "quality_direction": int(quality_direction),
        "score_index": i,
        "score": float(v),
    } for i, v in enumerate(scores)]


def _version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _save(run_dir, cfg, raw, summary, profiles, extra=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(run_dir/"raw_scores.csv", index=False)
    summary.to_csv(run_dir/"summary.csv", index=False)
    profiles.to_csv(run_dir/"behavior_profiles.csv", index=False)
    with (run_dir/"resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = {
        "config_hash": config_hash(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {x:_version(x) for x in [
            "torch","captum","quantus","scikit-learn","numpy","pandas","scipy","PyYAML"
        ]},
    }
    if extra:
        meta.update(extra)
    with (run_dir/"run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return run_dir


def run_oracle(cfg):
    seed = cfg["experiment"]["random_seed"]
    dc = cfg["dataset"]
    data = make_oracle_data(
        dc["n_samples"], dc["n_features"], dc["coefficients"], seed
    )
    model = OracleLinearClassifier(dc["n_features"], data.coefficients).eval()
    device = cfg["experiment"].get("device","cpu")
    model.to(device)
    a0_all = oracle_attributions(data.x, data.y, data.coefficients)
    rows = []

    fc = cfg["faithfulness"]
    if fc.get("enabled", True):
        n = min(fc["metric_sample_size"], len(data.x))
        x, y, a0 = data.x[:n], data.y[:n], a0_all[:n]
        for level in fc["corruption_levels"]:
            a = relocate_to_decoys(
                a0, level, dc["informative_features"], dc["decoy_start"]
            )
            if not np.allclose(np.abs(a).sum(1), np.abs(a0).sum(1), atol=1e-5):
                raise AssertionError("Faithfulness intervention changed L1 mass.")
            for mid, params in fc["metrics"].items():
                scores = evaluate_metric(
                    mid, params, model=model, x_batch=x, y_batch=y,
                    a_batch=a, device=device
                )
                rows += _rows(
                    "P1-A Oracle","faithfulness","oracle_to_decoy_relocation",
                    level,mid,scores,-1
                )

    cc = cfg["compactness"]
    if cc.get("enabled", True):
        xdummy = np.zeros((1,dc["n_features"]),dtype=np.float32)
        ydummy = np.zeros(1,dtype=np.int64)
        for k in cc["support_sizes"]:
            a = support_size_explanation(dc["n_features"],k)[None,:]
            for mid, params in cc["metrics"].items():
                scores = evaluate_metric(
                    mid,params,model=None,x_batch=xdummy,y_batch=ydummy,
                    a_batch=a,device=device
                )
                rows += _rows(
                    "P1-A Oracle","compactness","support_size",k,mid,scores,-1
                )
        for beta in cc["concentration_betas"]:
            a = concentration_explanation(dc["n_features"],beta)[None,:]
            for mid, params in cc["metrics"].items():
                scores = evaluate_metric(
                    mid,params,model=None,x_batch=xdummy,y_batch=ydummy,
                    a_batch=a,device=device
                )
                rows += _rows(
                    "P1-A Oracle","compactness","concentration_beta",
                    beta,mid,scores,1
                )

    rc = cfg["robustness"]
    if rc.get("enabled", True):
        n = min(rc["metric_sample_size"], len(data.x))
        x, y = data.x[:n], data.y[:n]
        for level in rc["instability_levels"]:
            explainer = ControlledUnstableOracleExplainer(
                data.coefficients, dc["n_features"], dc["informative_features"],
                level, rc["omega"], rc["unstable_mass_ratio"]
            )
            a = explainer(model,x,y)
            for mid, params in rc["metrics"].items():
                scores = evaluate_metric(
                    mid,params,model=model,x_batch=x,y_batch=y,
                    a_batch=a,explain_func=explainer,device=device
                )
                rows += _rows(
                    "P1-A Oracle","robustness","controlled_instability",
                    level,mid,scores,-1
                )

    raw = pd.DataFrame(rows)
    summ = summarize(raw)
    prof = behavior_profiles(summ)
    out = Path(cfg["experiment"]["output_dir"])/cfg["experiment"]["name"]
    return _save(out,cfg,raw,summ,prof)


def run_learned(cfg):
    seed = cfg["experiment"]["random_seed"]
    dc = cfg["dataset"]
    lc = cfg["learned_xai"]
    data = make_oracle_data(
        dc["n_samples"], dc["n_features"], dc["coefficients"], seed
    )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data.x))
    ntr = int(len(idx)*lc["train_fraction"])
    nva = int(len(idx)*lc["val_fraction"])
    tr,va,te = idx[:ntr],idx[ntr:ntr+nva],idx[ntr+nva:]

    model = MLPClassifier(dc["n_features"],tuple(lc["hidden_dims"]))
    hist = train_mlp(
        model,data.x[tr],data.y[tr],data.x[va],data.y[va],
        epochs=lc["epochs"],batch_size=lc["batch_size"],
        learning_rate=lc["learning_rate"],seed=seed,
        device=cfg["experiment"].get("device","cpu")
    )
    device = cfg["experiment"].get("device","cpu")
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(data.x[te],device=device)).argmax(1).cpu().numpy()
    test_acc = float((pred==data.y[te]).mean())

    n = min(cfg["faithfulness"]["metric_sample_size"],len(te))
    x,y = data.x[te][:n],pred[:n]
    rows = []
    for method in lc["methods"]:
        exp = CaptumTabularExplainer(
            method,lc["integrated_gradients_steps"],
            lc["gradient_shap_samples"],device
        )
        a = exp(model,x,y)
        for mid,params in cfg["faithfulness"]["metrics"].items():
            scores = evaluate_metric(
                mid,params,model=model,x_batch=x,y_batch=y,
                a_batch=a,device=device
            )
            rows += _rows(
                "P1-B Learned XAI","faithfulness",
                f"xai_method::{method}",0.0,mid,scores,0
            )

    raw = pd.DataFrame(rows)
    summ = summarize(raw)
    prof = pd.DataFrame()
    out = Path(cfg["experiment"]["output_dir"])/(cfg["experiment"]["name"]+"_learned_xai")
    return _save(out,cfg,raw,summ,prof,{
        "training":{"final_val_accuracy":hist[-1]["val_accuracy"],"test_accuracy":test_acc}
    })
