# EXP-01-R — Quantus FC on Adult data

[Pre-run Atlas protocol](atlas_exp01r_protocol_2026-11-05.json). The data-source change and numerical gates were specified before the first real-data smoke.

### 1. Question / RQ

**On real Adult records and a fitted additive model, does changing subset size k change population FC, or the finite score estimated from B sampled groups?**

This extends EXP-01 to real inputs and Captum explanations under HYP-01, addressing RQ-01/RQ-03. Income labels are prediction targets, not explanation ground truth.

### 2. Measurement Contract — what is fixed or varied?

**Dataset:** OpenXAI's processed **Adult Income** has **36,177 train / 9,045 test records**. Six numeric inputs cover age, census weight (`fnlwgt`), education level, capital gain/loss and weekly work hours. Seven binary inputs encode sex, workclass, marital status, occupation, relationship, race and country. Label: **1 = annual income >$50,000; 0 = otherwise**.

**Previous XAI use:** Agarwal et al., *OpenXAI* (NeurIPS 2022), evaluated Adult explanations including IG with logistic regression: **Table 3, p. 9**, faithfulness results. Our Quantus contract is a new experiment; those scores are not reproduced. [Paper](https://papers.nips.cc/paper/2022/file/65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf).

| Item | Frozen smoke rule |
|---|---|
| Data source | Author CSV snapshot `3665d80`, referenced by OpenXAI's historical loader. Verify SHA256, 13-column order and labels. Source-change reason below. |
| Preprocessing / model | StandardScaler fitted on train only, over all 13 encoded inputs. Fit one scikit-learn 1.9.1 logistic regression: lbfgs, L2 (`l1_ratio=0`), C=1, max_iter=2000, tol=1e-8. Save and reuse the fitted pipeline. |
| Scalar target / C | Fixed high-income log-odds $g(x)=\beta^\top x+c$, including the fitted intercept. Numeric zero baseline in standardized coordinates equals the training means in raw encoded coordinates. |
| Explanations | Actual Captum 0.9.0 IG: float64, zero baseline, multiply_by_inputs=True, 50 Gauss-Legendre steps. Signed, without attribution normalization. |
| Primary evaluator / theta | Quantus 0.6.0 public FC / singleton FE. Zero replacement in the original input, official Pearson, no softmax/aggregation. Same recorded shape/precision controls as EXP-01. |
| Randomness / replication | N=inputs, R=evaluator repeats, B=draws per score. Seed native Quantus calls and verify actual paired masks across candidates/k/B. Split/model/evaluator/analysis seeds: 2026110501/02/03/04. |

### 3. Proposed experiment design and flow

**IG** is the calibration anchor. **A/B** shift its coordinates by one/two positions, preserving signed values and norms. These fixed controls are not separate XAI methods. Compare each input's population A/B margin; neither ordering is assumed. Practical tie threshold: 0.01.

Check actual IG against $v_j=\beta_jx_j$ and completeness, using pre-run absolute/relative tolerances **1e-7 / 1e-7**. Stop and retain failures; never substitute analytical v for Captum output.

```text
[Verify Adult files; reserve test rows; fit on training data]
                         ↓
[Save fixed pipeline; generate IG and shifts A/B]
                         ↓
[Check IG/reference and permitted exact enumeration]
                         ↓
[Quantus FC/FE → actual masks, U/V, scores and failures]
                         ↓
[Population vs finite-score graph + error/budget report]
                         ↓
[Later: development review → freeze → reserved confirmation]
```

**Derived:** $FC_{\mathrm{pop}}(k)=\mathrm{Corr}_j(a_j,v_j)$ for positive variances and $1\leq k<13$. Enumerate actual replacements for the first input at allowed k; tolerance **1e-10**. NumPy supplies reference/trace checks; Quantus supplies reported scores.

**Grid:** smoke N=4, R=4, B={5,25,100}; development N=128, R=20, B={25,100,400}; k={1,5,10,12}. A pre-run permutation reserves **512 test rows and k={2,7,11}**, then assigns separate smoke/development rows. Counts are planning choices, not power estimates. Confirmation stays blocked until a separate pipeline/code/prediction freeze is implemented.

### 4. Latest result — Smoke

**Observed:** run `e6e61e8b575a4f12`, clean source [`1b8967b`](https://github.com/theguardianofthebed/xai-metric-behavior/commit/1b8967b1de1a5300cced2ff01b5167799227cc24). All **576 FC + 12 FE** calls were eligible; **0 undefined, near-constant warnings or exceptions**.

**Direct RQ answer:** the first input's exhaustive population check agreed with the coordinate prediction across all four allowed k values: **12/12 comparisons**, maximum error **1.39e-16**. Finite Quantus scores varied despite the fixed population target:

| Draws B | A/B RMSE from population reference | Strict A/B order reversals |
|---|---:|---:|
| 5 | 0.475097 | 23/64 |
| 25 | 0.157345 | 10/64 |
| 100 | 0.076164 | 5/64 |

RMSE uses the same **16/16 input-repeat pairs** across every k/B/control, with equal input weights. Reversals pool 4 inputs × 4 repeats × 4 k settings; these are correlated comparisons, not 64 independent people. Finite ties remain separate in the CSVs.

[Population versus finite scores](https://github.com/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/results/verification/exp01r-smoke/figures/rq_population_vs_estimates.png): x=k, y=FC; dashed=population, points=actual repeats. [Error versus budget](https://github.com/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/results/verification/exp01r-smoke/figures/rq_error_vs_budget.png): x=B, y=RMSE, lower is closer to the reference. [Readable answer and complete run](https://github.com/theguardianofthebed/xai-metric-behavior/tree/exp01-atlas-protocol/results/verification/exp01r-smoke).

### 5. Limitations / failure report

| Status / issue | Handling |
|---|---|
| **Observed — download failure** | Dataverse V3 returned HTTP 403. The pinned author snapshot was chosen before measurement; V3 byte identity remains **unverified**. |
| **Observed — numerical gates passed** | Actual IG/reference maximum coordinate error 1.10e-9; all 4 inputs passed. All 588 official-call traces and 576 FC pairing checks passed. |
| **Quantus applicability** | Zero failures here does not guarantee later validity. Retain exceptions, undefined Pearson and near-constant warnings; never replace cases. |
| **Encoded-category/OOD limit** | Mean replacement and IG's straight path can leave valid categories or joint support. This tests encoded-coordinate replacement, not causal removal. |
| **Sampling / split limits** | Four inputs; one model. Published train/test splits share some encoded patterns. Finite Pearson need not be unbiased or improve monotonically with B. |

**Development/confirmation unexecuted; 512 reserved rows and k={2,7,11} unevaluated.** Broad real-data reliability and training-seed robustness remain untested.

### 6. What physically implements the design?

| Component | Purpose |
|---|---|
| `exp01r_data.py` | Verified downloads, train-only transform/model, reserved rows, actual Captum IG and numerical gate. |
| `quantus_backend.py` | Official FC/FE with fitted intercept; actual perturbation/Pearson traces and checks. |
| `exp01r.py` + analysis | Run the grid; retain failures; compare population references, finite errors and A/B margins. |
| CLI + `EXP-01-R.ipynb` | Same engine, isolated CPU installation, direct RQ graphs and readable result output. |

In [Colab](https://colab.research.google.com/github/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/notebooks/EXP-01-R.ipynb): fresh CPU runtime, `PROFILE="smoke"`, run five code cells in order. Or use branch `exp01-atlas-protocol` in an isolated environment:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[exp01r]'
python -m xai_metric_behavior run-real --config configs/exp01r_smoke.yaml --output results/exp01r-smoke
```

Use a new output directory. Preserve `data/cache/openxai-adult` for development: it holds the fitted model and reserved IDs. Published artifacts include model, partition, explanations, scores and traces; full CSVs remain cached.

**Verified:** 42 tests passed. All five actual notebook code cells passed in an external Python 3.13.11 kernel, reusing the checkout/environment/cache; six key outputs matched the canonical smoke byte-for-byte. Hosted Colab UI and a fresh EXP-01-R clone/install were not tested. [Verification](https://github.com/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/results/verification/exp01r_validation.json).

### 7. What insight do we get?

**A changed sampled score need not mean a changed population target.** The population graph answers the k question; errors/failures show what limited B delivers. IG alone would hide much of this behavior because its defined FC is expected to be 1; shifted controls expose sampling effects.

**Research Learning Summary:** Need → test real inputs. Intuition → different sampled groups score one fixed explanation. Definition → finite FC versus population reference. Reason → additive log-odds enables an independent numerical check.
