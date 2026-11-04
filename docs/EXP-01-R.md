# EXP-01-R — Quantus FC on Adult data

[Pre-run Atlas protocol](atlas_exp01r_protocol_2026-11-05.json). The data-source change and numerical gates were specified before the first real-data smoke.

### 1. Question / RQ

**On real Adult records and a fitted additive model, does changing subset size k change population FC, or the finite score estimated from B sampled groups?**

This extends EXP-01 to real inputs and Captum explanations, addressing RQ-01/RQ-03 under HYP-01's assumptions. Income labels are prediction targets, not ground-truth explanations of why a person earns that income.

### 2. Measurement Contract — what is fixed or varied?

**Dataset:** OpenXAI's processed **Adult Income** has 45,222 person records: 36,177 train and 9,045 test. Six numeric inputs describe age, census sampling weight (`fnlwgt`), education level, capital gain/loss and weekly work hours. Seven binary inputs encode sex, workclass, marital status, occupation, relationship, race and country. The label is **1 = annual income >$50,000; 0 = otherwise**.

**Previous XAI use:** Agarwal et al., *OpenXAI: Towards a Transparent Evaluation of Model Explanations* (NeurIPS 2022), evaluated attribution methods including IG on Adult with logistic regression, reporting faithfulness results in **Table 3, p. 9**. Our Quantus measurement contract is a new experiment, not a reproduction of those scores. [Paper](https://papers.nips.cc/paper/2022/file/65398a0eba88c9b4a1c38ae405b125ef-Paper-Datasets_and_Benchmarks.pdf).

| Item | Frozen smoke rule |
|---|---|
| Data source | Author-published CSV snapshot `3665d80`, linked by OpenXAI's official historical loader. Verify exact SHA256, 13-column order and labels. Dataverse V3 download returned 403; byte identity with V3 is **unverified**. |
| Preprocessing / model | StandardScaler fitted on train only, over all 13 encoded inputs. Fit one scikit-learn 1.9.1 logistic regression: lbfgs, L2 (`l1_ratio=0`), C=1, max_iter=2000, tol=1e-8. Save and reuse the fitted pipeline. |
| Scalar target / C | Fixed high-income log-odds $g(x)=\beta^\top x+c$, including the fitted intercept. Numeric zero baseline in standardized coordinates equals the training means in raw encoded coordinates. |
| Explanations | Actual Captum 0.9.0 IG: float64, zero baseline, multiply_by_inputs=True, 50 Gauss-Legendre steps. Preserve signs; no abs or attribution normalization. |
| Primary evaluator / theta | Quantus 0.6.0 public FC / singleton FE. Zero replacement in the original input, official Pearson, no softmax/aggregation. Same recorded shape/precision controls as EXP-01. |
| Randomness / replication | N=inputs, R=evaluator repeats, B=draws per score. Seed native Quantus calls and verify actual paired masks across candidates/k/B. Split/model/evaluator/analysis seeds: 2026110501/02/03/04. |

### 3. Proposed experiment design and flow

**IG** is the calibration anchor. **A/B** cyclically shift its coordinates by one/two positions, preserving its signed values and norms. They are fixed nonexact controls, not separate XAI methods. A dense model gives no universal A>B ordering: compute each input's population margin and retain either sign and practical ties (0.01).

Check actual IG against $v_j=\beta_jx_j$ coordinate by coordinate, and check completeness. Freeze absolute/relative tolerances at **1e-7 / 1e-7** before execution; stop and save discrepancies on failure. Never substitute analytical v for Captum's output.

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

**Derived:** $FC_{\mathrm{pop}}(k)=\mathrm{Corr}_j(a_j,v_j)$ for positive variances and $1\leq k<13$. Dependence among real features does not break this per-input additive identity. Enumerate actual replacements for the first selected input at allowed k; compare with tolerance **1e-10**. NumPy is only the reference/trace check; reported scores come from Quantus.

**Grid:** smoke N=4, R=4, B={5,25,100}; development N=128, R=20, B={25,100,400}; k={1,5,10,12}. Before measurement, reserve **512 separate test rows and k={2,7,11}**. A fixed test-row permutation assigns reserved rows first, then smoke and development rows. Never replace failed cases. Counts are planning choices, not power estimates. Confirmation remains blocked until its separate model/pipeline/code-bound freeze is implemented and reviewed.

### 4. Latest result — Smoke

**Implementation in progress; no EXP-01-R measurement result yet.** The protocol above is fixed before the first real-data smoke. EXP-01's synthetic smoke is separate.

### 5. Limitations / failure report

| Status / issue | Handling |
|---|---|
| **Observed — Dataverse access failed** | HTTP 403 blocked the planned V3 download. Use the explicitly pinned author snapshot referenced by official OpenXAI code; do not claim V3 byte identity. |
| **Preprocessing / model gate** | Train-only scaler, fixed positive-class orientation and saved pipeline. Stop on convergence or IG/adapter disagreement; keep the failure record. |
| **Quantus applicability** | Preserve validation exceptions, undefined Pearson and near-constant warnings separately. Do not change signs or pad vectors to bypass validation. |
| **Encoded-category/OOD limit** | Mean replacement and IG's straight path can leave valid categories or joint support. This tests encoded-coordinate replacement, not causal removal. |
| **Statistical scope** | Use joint eligibility and equal input weighting; report every attempted call and exclusions. Finite Pearson need not be unbiased or improve monotonically with B. |

One fitted model does not test training-seed variation. Four smoke inputs cannot establish real-data reliability or held-out success. Near-zero population A/B margins give weak discrimination. Full development and confirmation remain unexecuted.

### 6. What physically implements the design?

| Component | Purpose |
|---|---|
| `exp01r_data.py` | Verified downloads, train-only transform/model, reserved rows, actual Captum IG and numerical gate. |
| `quantus_backend.py` | Official FC/FE with fitted intercept; actual perturbation/Pearson traces and checks. |
| `exp01r.py` + analysis | Run the grid; retain failures; compare population references, finite errors and A/B margins. |
| CLI + `EXP-01-R.ipynb` | Same engine, isolated CPU installation, direct RQ graphs and readable result output. |

Run from the `exp01-atlas-protocol` branch after publication:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[exp01r]'
python -m xai_metric_behavior run-real --config configs/exp01r_smoke.yaml --output results/exp01r-smoke
```

Use an isolated Python environment and a new output directory. [Design / run guide](https://github.com/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/docs/EXP-01-R.md). Downloaded dataset files stay in the local cache; model, partition, manifest, explanations, raw scores and traces document the run.

### 7. What insight do we get?

**Separate a fixed population target from sampling-driven score changes on real inputs.** The population/reference graph answers the k question; finite errors and failure counts answer what a limited B actually delivers. An IG-only run would hide much of this behavior because its defined FC is expected to be 1. The shifted controls expose estimation effects without assuming either is universally better.

**Research Learning Summary:** Need → test the mechanism on real inputs. Intuition → Quantus samples different feature groups for one fixed Adult explanation. Definition → compare finite FC with the additive population reference. Reason → fixed log-odds preserves additivity; real data adds preprocessing, numerical and applicability checks.
