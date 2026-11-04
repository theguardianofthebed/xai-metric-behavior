# EXP-01 — Quantus FC: same explanation, different estimates

Based on the [Atlas revision-9 contract](atlas_quantus_contract_2026-11-04.json), fixed before the corrected rerun. [Bhatt et al.](https://doi.org/10.24963/ijcai.2020/417) defines the subset-correlation family; [HYP-01](ATLAS_REFERENCE.md#hyp-01) is the project's conditional derivation.

### 1. Question / RQ

**With the explanation fixed, does replacing more features at once (k) change population FC, or only the estimate from B sampled groups?**

This addresses [RQ-01](ATLAS_REFERENCE.md#rq-01) within [RQ-00](ATLAS_REFERENCE.md#rq-00). Population FC considers every allowed size-k subset for one input. Quantus estimates it from B draws. HYP-01 predicts population invariance; finite estimates can vary, be undefined, or be numerically unstable.

### 2. Measurement Contract — what is fixed or varied?

| Item | Rule |
|---|---|
| Model / data | 20 independent standard-normal features; $g(x)=2.5x_1-2x_2+1.5x_3+x_4-0.5x_5$. Five active features; known weights give exact contributions without training. |
| Explanation meaning: C | Signed zero-relative contributions $v_j=w_jx_j$. Preserve signs; no attribution normalization or probability transform. |
| Primary evaluator | **Quantus 0.6.0 public FC / singleton FE calls**; abs=False, normalise=False, return_aggregate=False, softmax=False. |
| Evaluator rules: theta | Uniform size-k subsets; replace selected features with numeric **0.0 in the original input**. Official Pearson compares attribution sums U with actual output changes V. |
| Sample sizes | k = features per draw; B = draws per score; R = independent evaluator repeats per input; N = independent inputs. Keep $1\leq k<20$. |
| Randomness: xi | Seed native Quantus sampling before each paired call. Verify recorded candidate, k and B prefixes; no injected masks. |
| Reporting | Average eligible repeats within inputs, then weight represented inputs equally. Compare jointly eligible cases; bootstrap inputs. Retain raw returns, warnings and reasons. |

The official path uses `(N,1,d)` inputs, flattened by the model, and scoped float64 arithmetic with state restoration. The [Quantus guide](QUANTUS_AUDIT.md) explains these compatibility choices.

```math
U_S=\sum_{j\in S}a_j,\qquad V_S=g(x)-g(x_{S\leftarrow0}),\qquad
\widehat{FC}_B=\mathrm{Corr}_{r=1}^{B}(U_{S_r},V_{S_r}).
```

### 3. Proposed experiment design and flow

**Exact** is the true contribution vector, checking unit correlation on nondegenerate samples. **A** moves contribution 1 to unused feature 6; **B** also moves contribution 2 to unused feature 7 (1-based numbering). Values, sum, sparsity and norms stay fixed; locations change. Two nonexact candidates reveal sampling effects that the exact anchor can hide. Fix them before inspecting reversals.

```text
[Fix contract; prepare inputs and Exact / A / B]
                    ↓
[Analytical + enumerable population reference]
                    ↓
[Public Quantus → recorded masks, U/V, scores]
                    ↓
[Check traces/pairing; summarize errors, A−B and failures]
                    ↓
[Development review → freeze → held-out confirmation]
```

**Derived:** $FC_{\mathrm{pop}}(k)=\mathrm{Corr}_j(a_j,v_j)$ for positive coordinate variances and $1\leq k<20$. Additivity makes subset responses sums of effects; uniform sampling gives covariance and both variances a common factor that cancels in Pearson.

Enumeration checks the teaching fixture and first 20-feature input at configured k, using tolerance $10^{-10}$. Official singleton FE checks every input. The A/B population margin is $v_2^2/\sum_j(v_j-\bar v)^2\geq0$; record its input-dependent size, including practical near-ties.

**NumPy's auxiliary role:** replay traces to check the adapter; Quantus's return remains the reported score. An upstream near-constant warning defers only numerical score parity, retaining its discrepancy. Structural checks remain strict. Analytical/enumeration references address the population prediction.

**Planned development:** N=128, R=20, k={1,5,10,19}, B={25,100,400}. **Confirmation:** N=512, R=20, reserved k={2,7,13}, the same budgets, fresh streams. Counts are planning choices, not a power calculation. Freeze predictions, margins, failure/analysis rules, code/configuration, backend versions/source hashes and development provenance before confirmation.

### 4. Latest result — Smoke

**Completed 4 November 2026:** run `0221db61739ed30a`, from clean source [`ce08d31`](https://github.com/theguardianofthebed/xai-metric-behavior/commit/ce08d3181f9325b82e91b9bb1b1585e10c787ab7). N=4, R=4, k={1,5,10,19}, B={5,25,100}. [Run artifacts](../results/verification/smoke/).

| Check | Observed result |
|---|---|
| FC outcomes | **576 attempts = 559 official finite + 17 undefined.** The finite returns comprise **555 eligible + 4 warned near-constant** values. **0 exceptions.** |
| Population reference | **16/16 enumeration checks passed**; maximum discrepancy $5.33\times10^{-15}$. Scope: teaching fixture and first 20-feature input at configured k. |

**One recorded example:** input_id=1, repetition=0, candidate A, k=1. Population FC is **0.46939**; B={5,25,100} gives approximately **{1.00000,1.00000,0.49760}**. The explanation stayed fixed. This is one draw-plan outcome; general budget trends require development. [Raw scores](../results/verification/smoke/scores.csv)

### 5. Limitations / failure report

| Status / issue | Meaning and handling |
|---|---|
| **Observed — near-constant input** | The first attempt's replay guard stopped at input 2 / repeat 1 / A / k=19 / B=5: U varied by only $8.88\times10^{-16}$. Quantus returned **0.31623** with a warning; NumPy replay gave **0.25820**. Preserve the official value and mark it ineligible. [Failure record](../archive/exp01-quantus-first-attempt/). |
| **Observed — 17 undefined returns** | Constant sampled U or V gives no defined Pearson correlation. Retain NaN and the reason, not a score of zero. |
| **Verified — adapter/trace checks** | All **588 FC/FE calls** passed structural checks; **576 FC calls** passed recorded-draw pairing checks. Detailed parity classification is in the [audit](QUANTUS_AUDIT.md). |
| **Fixed — Colab setup failure** | Cell 1 failed at `venv`, before Quantus. Reproduced locally without `ensurepip`; setup now creates a pipless environment, bootstraps target pip and repairs partial creation. The supplied traceback did not include the underlying stderr. |
| **Fresh setup verified; hosted UI unverified** | New Python 3.13.11 environment: all five cells and smoke passed; three key files match canonical bytes. Missing-`ensurepip` creation/recovery/reuse also passed. [Record](../results/verification/colab_setup_validation_2026-11-05.json). Hosted Colab/Jupyter UI and notebook clone remain unverified. |

**Eligibility:** preserve every official return; exclude undefined values, exceptions and upstream `NearConstantInputWarning` cases from statistical summaries. A finite return can still be ineligible. The rule was added after the first attempt; samples, scores and the $10^{-10}$ tolerance were preserved.

Summaries condition on eligibility, which can differ by setting. Rank contrasts require both candidates at both settings to be eligible. Use all-attempt denominators; reason incidences may overlap, while total ineligible is their union. Four inputs do not establish general failure rates, real-data transfer or held-out success. Intervals are descriptive/pointwise; 0.01 is an operational tie threshold, not an equivalence bound. **Full development and confirmation remain unexecuted.**

An enumeration discrepancy under verified premises requires investigating the calculation/derivation; a finite-score shift alone does not refute HYP-01. Interactions, nonuniform subsets, nonlinear output transforms and subset-dependent baselines need separate contracts.

### 6. What physically implements the design?

| Component | Role |
|---|---|
| [`quantus_backend.py`](../src/xai_metric_behavior/quantus_backend.py) | Public FC/FE calls; delegating wrappers record actual calculations and preserve official scores. |
| `core.py` | Known model/candidates, analytical references, enumeration and auxiliary replay. |
| `analysis.py`; `plotting.py` | Paired eligibility, input-level uncertainty and diagnostic figures. |
| `configs/`; `experiment.py` | Settings, provenance, traces/failures and confirmation gate. |
| CLI; notebook | Same engine; the notebook runs the CLI in an isolated CPU environment and inspects outputs. |

**Verification:** 27/27 tests and 34/34 audit checks passed. Notebook execution is a verification replay of this smoke, not another research result. [Verification record](../results/verification/summary.json)

**Local CLI:** Python 3.12, CPU, from the repository root (Linux). No training, external dataset or GPU is needed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m xai_metric_behavior run --config configs/exp01_smoke.yaml --output results/exp01-smoke
```

[Clone instructions](../README.md#run-on-cpu) · [Open in Colab](https://colab.research.google.com/github/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/notebooks/EXP-01.ipynb). **Colab:** reopen the updated notebook in a CPU runtime and run its cells in order. Cell 1 creates or repairs `.venv` without requiring host `ensurepip`; kernel pip >=22.3 manages only the target environment. The notebook kernel is not upgraded. Use a new output directory per run. Keep the complete directory: manifest, raw scores, eligibility/failures, public-call records and traces. [Output map](../results/README.md)

**Plot axes:** population FC versus k shows the first input; bias/RMSE versus B uses eligible scores. `analysis_ineligible_fraction.png` separates undefined and numerical-instability fractions, each divided by all attempts.

After smoke, run development (the full grid can take much longer):

```bash
python -m xai_metric_behavior run --config configs/exp01_development.yaml --output results/exp01-development
```

Review precision and fix the prediction before freezing and opening reserved results:

```bash
python -m xai_metric_behavior freeze --config configs/exp01_confirmation.yaml --development-run results/exp01-development --output protocol-freeze.json
python -m xai_metric_behavior run --config configs/exp01_confirmation.yaml --freeze protocol-freeze.json --output results/exp01-confirmation
```

Keep the freeze with its outputs. A revised prediction after viewing confirmation needs fresh reserved conditions. **EXP-01-R remains an Atlas plan only.**

### 7. What insight do we get?

**Separate explanation differences from evaluator sampling effects.** A nonexact explanation can receive finite FC near 1; a finite return can also be numerically unstable. Read k, B, uncertainty, warnings and eligibility together. This invariance study does not use directional DMS/Response Alignment.

**Research Learning Summary:** 필요성 → 설명의 차이와 평가기의 sampling 효과를 구분한다. 직관적인 예시/이해 → 같은 A의 점수가 B에 따라 약 1에서 0.498로 바뀌어도 설명은 그대로다. 공식적인 정의 → population FC는 전체 subset 분포, finite FC는 B개 draw의 Pearson이다. 근거/그게 되는 이유 → additive 조건에서 population Pearson의 공통 계수는 상쇄되지만 finite sample은 변동·퇴화할 수 있다. Near-constant warning이 있는 값도 보관하고, 분석에서는 제외한다.
