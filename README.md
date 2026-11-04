# XAI Metric Behavior

**When and why do quantitative XAI metrics change their judgments?**

This project studies metric behavior through controlled experiments: write down what a score measures, derive a prediction, retain the calculations behind the score, and test the prediction under reserved conditions. A changed score is not automatically evidence of a worse explanation or an invalid metric.

The current starting point is **EXP-01: separate subset size from finite-sampling effects**. Its source is the [XAI Framework Atlas](https://timecapsule-traceability-atlas.iamlim0619.chatgpt.site), captured at revision 4 in this repository. The previous Phase 1 scaffold is preserved in [archive/phase1](archive/README.md).

## What EXP-01 asks

Faithfulness Correlation (FC) compares a subset's attribution sum with the model-output change caused by replacing those features. We vary:

- **Subset size, $k$:** how many features the evaluator replaces in one draw.
- **Sampling budget, $B$:** how many draws form one input's estimated correlation.

The model, input, explanation, baseline, and signed scalar target remain fixed within a comparison. In the declared additive setting, the **population** FC is mathematically independent of $k$. Finite estimates can still differ in bias, variance, and frequency of undefined results. Exact enumeration and paired repeated draws separate these effects. See the [design and implementation report](docs/EXP-01.md) for the assumptions and derivation.

> **쉽게 말하면:** 같은 설명을 평가하더라도, 평가기가 한 번에 몇 개의 feature를 바꾸고 몇 번 반복하느냐에 따라 추정 점수가 흔들릴 수 있습니다. EXP-01은 모든 가능한 subset을 고려한 값과 실제로 일부만 뽑아 계산한 값을 구분합니다. 단순한 모델을 쓰는 이유는 이 두 값의 차이를 정확하게 설명하기 위해서입니다.

## Run

Use Python 3.12 and check out the `exp01-atlas-protocol` branch:

```bash
git clone --branch exp01-atlas-protocol https://github.com/theguardianofthebed/xai-metric-behavior.git
cd xai-metric-behavior
python -m pip install -e .
python -m unittest discover -s tests -v
python -m xai_metric_behavior run --config configs/exp01_smoke.yaml --output results/exp01-smoke
```

The experiment generates its own synthetic data and uses an explicit model formula; no dataset download, model training, or GPU is required. Use a new output directory for each run. The [notebook](notebooks/EXP-01.ipynb) walks through the same implementation; you can also [open it in Colab](https://colab.research.google.com/github/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/notebooks/EXP-01.ipynb). Full development and confirmation instructions are in the design report; confirmation requires a frozen prediction specification.

## Reading guide

| Read | Purpose |
|---|---|
| [Framework](docs/FRAMEWORK.md) | Research motivation, four stages, evidence rules, and the experiment roadmap. |
| [EXP-01 design and implementation](docs/EXP-01.md) | Original Atlas protocol, design choices, code roles, execution, and failure report. |
| [Quantus audit](docs/QUANTUS_AUDIT.md) | Pinned implementation semantics and the status of parity checks. |
| [Atlas reference](docs/ATLAS_REFERENCE.md) | Full source context, organized by stable Atlas element ID. |
| [Historical archive](archive/README.md) | Previous code and documents, separated from the current experiment. |

## Verification

18 tests and 34 Quantus parity checks passed. The smoke run retained 547 valid and 29 undefined scores out of 576 attempts, with zero execution exceptions. Notebook code cells also reproduced the CLI scores. This environment blocked Jupyter socket binding, so the notebook UI remains unverified. Full development and reserved confirmation have not been run. See the [verification and failure report](docs/EXP-01.md#failure-report).

## Evidence boundary

**Derived** means a mathematical consequence under stated assumptions; **implemented** means code exists; **observed** requires an identified run; **proposed** and **unverified** remain separate. Current execution evidence belongs in the EXP-01 report, not in historical pilot numbers or Atlas status labels.

The first study concerns feature-attribution Faithfulness. Compactness and Robustness/Continuity remain later extensions needing their own measurement definitions. This repository does not claim a universal metric, a universal ranking of explanations, or verified novelty of metric meta-evaluation.
