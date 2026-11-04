# XAI Metric Behavior

**When and why do quantitative XAI metrics change their judgments?**

EXP-01 keeps explanations fixed and changes the evaluator's **subset size** $k$ and **sampling budget** $B$. Under the declared additive model, population Faithfulness Correlation (FC) is independent of $k$. A finite estimate can fluctuate, be undefined, or be numerically unstable. The experiment separates these effects using a known model, two fixed nonexact candidates, and an exact anchor.

**Quantus 0.6.0 public FC/FE calls are the measurement engine.** The NumPy formula and exhaustive checks supply analytical references; trace replay checks that the official calculations obey the declared contract. Parity is an implementation guard, not a result about explanation quality.

> **쉽게 말하면:** 같은 설명을 그대로 두고, 평가기가 한 번에 바꾸는 feature 수와 반복 횟수만 바꿉니다. Quantus가 계산한 점수와 정확히 구할 수 있는 기준값을 비교하여 sampling 오차와 실제 기준값의 변화를 구분합니다.

## Run on CPU

Use Python 3.12 in a fresh virtual environment. The CPU PyTorch installation below is for Linux/Colab; no GPU, model training or dataset download is needed.

```bash
git clone --branch exp01-atlas-protocol https://github.com/theguardianofthebed/xai-metric-behavior.git
cd xai-metric-behavior
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m xai_metric_behavior run --config configs/exp01_smoke.yaml --output results/exp01-smoke
```

The package installation includes the required Quantus and scientific dependencies. Use a new output directory for every run. Read the manifest, failure counts and plots together. Full development and confirmation commands are in the [experiment report](docs/EXP-01.md#6-what-physically-implements-the-design).

[Open in Colab](https://colab.research.google.com/github/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/notebooks/EXP-01.ipynb) or use the [local notebook](notebooks/EXP-01.ipynb). The notebook creates an isolated CPU environment and calls the same CLI package; its kernel only inspects saved outputs, with no second metric implementation.

## Read the project

| Document | Purpose |
|---|---|
| [EXP-01](docs/EXP-01.md) | Seven compact sections: question, contract, design, latest smoke, limitations, implementation/run steps, and insight. |
| [Framework](docs/FRAMEWORK.md) | Four research stages, evidence rules, and later experiments. |
| [Quantus contract and checks](docs/QUANTUS_AUDIT.md) | Official-call settings, recorded traces, compatibility decisions, and the role of parity. |
| [Current verification](results/verification/) | Fresh Quantus execution evidence; full development and confirmation remain unexecuted. |
| [Atlas reference](docs/ATLAS_REFERENCE.md) | Full revision-4 source context, including the conditional derivation. |

The [live XAI Framework Atlas](https://timecapsule-traceability-atlas.iamlim0619.chatgpt.site) is the research map. **EXP-01-R remains an Atlas plan only**; this repository does not implement its proposed real-data experiment.

## Current evidence and boundaries

**Completed Quantus rerun verification is pending.** The [failure report](docs/EXP-01.md#5-limitations--failure-report) records the first attempt's near-constant case: its official finite score is preserved and excluded from analysis, with the warning and denominator retained. Earlier NumPy-engine results are preserved in [archive/exp01-numpy](archive/exp01-numpy/); the original Phase 1 scaffold remains in [archive/phase1](archive/README.md). Neither is relabeled as a Quantus experiment.

**Derived**, **implemented**, **observed**, **proposed**, and **unverified** are different evidence states. The initial synthetic Faithfulness study does not establish a universal metric or transfer to real data. Compactness and Robustness/Continuity need their own measurement contracts.
