from __future__ import annotations
import numpy as np
import torch

from .core import oracle_attributions


class OracleExplainer:
    def __init__(self, coefficients):
        self.coefficients = np.asarray(coefficients, dtype=np.float32)

    def __call__(self, model, inputs, targets, **kwargs):
        return oracle_attributions(inputs, targets, self.coefficients)


class ControlledUnstableOracleExplainer:
    def __init__(self, coefficients, n_features, informative_features,
                 level, omega=16.0, unstable_mass_ratio=0.35, seed=314159):
        self.coefficients = np.asarray(coefficients, dtype=np.float32)
        self.informative_features = informative_features
        self.level = float(level)
        self.omega = float(omega)
        self.ratio = float(unstable_mass_ratio)
        rng = np.random.default_rng(seed)
        self.A = rng.normal(size=(n_features, informative_features)).astype(np.float32)

    def __call__(self, model, inputs, targets, **kwargs):
        x = np.asarray(inputs, dtype=np.float32)
        base = oracle_attributions(x, targets, self.coefficients)
        raw = np.sin(self.omega * (x @ self.A)).astype(np.float32)
        raw_l1 = np.maximum(np.abs(raw).sum(1, keepdims=True), 1e-8)
        base_l1 = np.maximum(
            np.abs(base[:, :self.informative_features]).sum(1, keepdims=True), 1e-8
        )
        unstable = raw/raw_l1 * base_l1 * self.ratio
        out = base.copy()
        out[:, :self.informative_features] += self.level*unstable
        return out


class CaptumTabularExplainer:
    def __init__(self, method, ig_steps=32, gs_samples=20, device="cpu"):
        self.method = method
        self.ig_steps = ig_steps
        self.gs_samples = gs_samples
        self.device = device

    def __call__(self, model, inputs, targets, **kwargs):
        try:
            from captum.attr import Saliency, IntegratedGradients, DeepLift, GradientShap
        except ImportError as e:
            raise ImportError("Install requirements-experiment.txt first.") from e

        model = model.to(self.device).eval()
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        zero = torch.zeros_like(x)

        if self.method == "saliency":
            a = Saliency(model).attribute(x, target=y, abs=False)
        elif self.method == "integrated_gradients":
            a = IntegratedGradients(model).attribute(
                x, baselines=zero, target=y, n_steps=self.ig_steps
            )
        elif self.method == "deeplift":
            a = DeepLift(model).attribute(x, baselines=zero, target=y)
        elif self.method == "gradient_shap":
            refs = torch.cat([zero[:1], zero[:1] + 1e-4], dim=0)
            a = GradientShap(model).attribute(
                x, baselines=refs, target=y,
                n_samples=self.gs_samples, stdevs=0.0
            )
        else:
            raise ValueError(self.method)
        return a.detach().cpu().numpy().astype(np.float32)
