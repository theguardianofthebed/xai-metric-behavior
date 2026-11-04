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

[Open in Colab](https://colab.research.google.com/github/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/notebooks/EXP-01.ipynb) or use the [local notebook](notebooks/EXP-01.ipynb). The notebook creates an isolated CPU environment and calls the same CLI package; its kernel only inspects saved outputs, with no second metric implementation. If the old first cell failed at `venv`, **reopen the updated notebook** and rerun that cell: it repairs a partial environment using `--without-pip` and target-only pip bootstrap. An already open Colab copy retains the old cell code.

## Run EXP-01-R on Adult data

[Open EXP-01-R in Colab](https://colab.research.google.com/github/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/notebooks/EXP-01-R.ipynb) in a fresh CPU runtime. For an existing isolated CLI environment:

```bash
python -m pip install -e '.[exp01r]'
python -m xai_metric_behavior run-real --config configs/exp01r_smoke.yaml --output results/exp01r-smoke
```

This downloads hash-verified OpenXAI author CSVs, fits one model on training data, generates actual Captum IG, and measures official Quantus FC/FE. `rq_answer.txt` and the RQ figures answer whether k changes the population target or its finite estimate. Preserve the fitted model/partition cache when proceeding to development. [Compact contract, dataset provenance and limits](docs/EXP-01-R.md).

## Read the project

| Document | Purpose |
|---|---|
| [EXP-01](docs/EXP-01.md) | Seven compact sections: question, contract, design, latest smoke, limitations, implementation/run steps, and insight. |
| [Framework](docs/FRAMEWORK.md) | Four research stages, evidence rules, and later experiments. |
| [Quantus contract and checks](docs/QUANTUS_AUDIT.md) | Official-call settings, recorded traces, compatibility decisions, and the role of parity. |
| [Current verification](results/verification/) | Fresh Quantus execution evidence; full development and confirmation remain unexecuted. |
| [Atlas reference](docs/ATLAS_REFERENCE.md) | Full revision-4 source context, including the conditional derivation. |

The [live XAI Framework Atlas](https://timecapsule-traceability-atlas.iamlim0619.chatgpt.site) is the research map. **EXP-01-R** adds [real Adult data, a learned model and Captum IG](docs/EXP-01-R.md), using the same official Quantus measurement path. Its saved outputs directly compare population references with finite scores; confirmation remains reserved.

## Current evidence and boundaries

**Latest Quantus smoke completed:** 576 FC attempts produced **559 finite returns (555 analysis-eligible, 4 warned near-constant)** and **17 undefined**, with **0 exceptions**. All 27 tests and 34 audit checks passed. The updated notebook passed all five code cells from a fresh Python 3.13.11 environment and reproduced key CLI outputs byte-for-byte. Missing-`ensurepip` creation/recovery checks also passed; hosted Colab/Jupyter UI remains unverified. [Setup verification](results/verification/colab_setup_validation_2026-11-05.json). [Results and limits](docs/EXP-01.md#4-latest-result--smoke)

The first attempt's near-constant case is preserved in the [failure report](docs/EXP-01.md#5-limitations--failure-report). Earlier [NumPy evidence](archive/exp01-numpy/) and the [Phase 1 scaffold](archive/README.md) remain historical. Full development and confirmation have not run.

**Derived**, **implemented**, **observed**, **proposed**, and **unverified** are different evidence states. The initial synthetic Faithfulness study does not establish a universal metric or transfer to real data. Compactness and Robustness/Continuity need their own measurement contracts.
