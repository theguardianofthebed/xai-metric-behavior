# EXP-01 — Separate subset size from finite-sampling effects

**Question:** Does changing the number of replaced features change the underlying Faithfulness Correlation, or only the behavior of its finite estimate?

This report connects the Atlas design to the runnable experiment. The original protocol below is copied from **EXP-01 Details, Atlas revision 4**. Implementation choices and execution evidence are separate additions; the complete [source context](ATLAS_REFERENCE.md#exp-01) and [framework](FRAMEWORK.md) remain available.

### Atlas protocol

Start with a deterministic additive scalar target, including a sparse linear case. Declare a baseline before subset selection and sample uniformly among all subsets of each fixed size. Keep `1 <= k < d`. Hold the explanation fixed while varying `k` and the within-input perturbation budget `B`. The number of evaluated inputs is a separate design dimension.

For small feature dimension, enumerate all subsets to obtain a reference population score. Separately specify repeated random-sampling estimates for increasing budgets, with independently recorded seeds and the same subset realizations across explanation variants when pairing is intended. Include exact contributions and a fixed corrupted explanation. Exact contributions can yield sample correlation one on every nondegenerate draw, so they alone cannot diagnose estimator noise. If ranking reversals are an endpoint, preregister two distinct nonexact explanations in addition to the exact anchor, for example separate relocations of informative features into different unused coordinates. Select and fix their population margin during development, then retain the pair for confirmation. An exact-versus-corrupted pair cannot show a strict preference above the exact score of one in this contract; it is a calibration control.

```math
g(x)=2x_1+x_2,\quad x=(3,4,5),\quad b=0,
\quad v=(6,4,0),\quad a_{\mathrm{relocated}}=(0,4,6).
```

The formal prediction is HYP-01: population FC equals coordinate correlation for every admissible `k`. Singleton FE supplies the related deterministic check. Finite Pearson estimates may exhibit budget- and subset-size-dependent bias, variance, or undefined cases. These possibilities must be measured separately instead of averaging undefined outcomes into a nominal score.

Planned outputs are population score versus subset size; estimator error versus sample budget, stratified by subset size; and the fraction of undefined estimates. Within-input replicates characterize estimation uncertainty. Across-input summaries use inputs as sampling units and retain the per-input pairing.

**Falsification criterion:** A reproducible exact-enumeration discrepancy from HYP-01, with its premises verified, would contradict the stated derivation or reveal a calculation error. A finite-sample discrepancy alone does not do so. Nonuniform subset selection, nonlinear target transformations, interaction terms, or subset-dependent baselines belong to different conditions and cannot be pooled into this test.

### Implementation supplement

#### 1. The measurement contract

| Choice | Concrete rule | Why this choice answers the question |
|---|---|---|
| Input data | Independently generated $x\sim\mathcal N(0,I_{20})$. | Supplies repeatable test cases; no downloaded labels or external dataset is needed. The theorem does not require Gaussian inputs. |
| Model | $g(x)=w^\top x$, with $w=(2.5,-2,1.5,1,-0.5,0,\ldots,0)$. | Known feature effects allow an independent reference. Five used and fifteen unused coordinates support controlled relocation. |
| Output target | The same scalar $g$ throughout; no class switching, sigmoid, or softmax. | A nonlinear output transform would change the additive relationship being tested. No training or GPU is needed. |
| Baseline | Literal numeric zero, fixed before subset selection. | The attribution reference and evaluator replacement represent the same comparison. |
| Explanation | Signed contributions $v_j=w_j(x_j-b_j)$ and two fixed relocations. No absolute-value conversion or normalization. | Preserves the contribution meaning and makes the candidate differences explicit. |
| Subsets | Uniform size-$k$ subsets, distinct coordinates within each draw; repeated draws may repeat a subset. | This is the sampling law in HYP-01. The explanation does not select the subsets. |
| Comparison | Pearson correlation within one input. Larger values indicate stronger association under this contract. | Matches the declared FC mechanism; correlation does not establish numerical agreement or universal explanation quality. |
| Aggregation | Average repetitions within each input, then weight inputs equally. Retain paired eligibility and failures. | Keeps input variation separate from evaluator randomness. Pooling all subset pairs would be another estimand. |

These choices implement [SET-01–03](ATLAS_REFERENCE.md#set-01), [FW-01](ATLAS_REFERENCE.md#fw-01), and [MET-FC](ATLAS_REFERENCE.md#met-fc). They do not inherit the archived classifier target, attribution preprocessing, aggregation defaults, or compatibility patch.

#### 2. Why use three explanations?

The model always uses the same input coordinates. Only the attribution vector changes. Feature numbers below are **1-based**; Python indices are one smaller.

| Candidate | Construction | Role |
|---|---|---|
| Exact | $a=v$. | Calibration anchor: every nondegenerate estimate should equal one, up to numerical precision. |
| A | Move contribution 1 into unused coordinate 6; set coordinate 1 to zero. | First nonexact explanation. |
| B | Make the same move as A and move contribution 2 into unused coordinate 7. | Second nonexact explanation with a predefined additional error. |

These swaps preserve the signed value multiset, attribution sum, nonzero count, and L1/L2 norms. They change which coordinates receive the contributions. The rules are fixed before simulation; candidates are not selected because they happen to show reversals.

[EXP-05](ATLAS_REFERENCE.md#exp-05) supplies the relocation identity. Let $D_v=\sum_j(v_j-\bar v)^2>0$. For this nested pair:

```math
FC_{\rm pop}(A)-FC_{\rm pop}(B)=\frac{v_2^2}{D_v}\geq 0.
```

This is a **derived** population margin, not an observed result. Its size varies by input; it is zero when $v_2=0$ and can be practically small otherwise. Store the per-input margin. The exact-versus-nonexact comparison calibrates the evaluator; A-versus-B can expose finite-sampling ordering changes. The exact anchor cannot be substantively beaten by another defined Pearson score.

#### 3. The prediction and its independent check

For a fixed input and explanation:

```math
U_S=\sum_{j\in S}a_j,\qquad
V_S=g(x)-g(x_{S\leftarrow b}),\qquad
\widehat{FC}_B=\operatorname{Corr}_{r=1}^{B}(U_{S_r},V_{S_r}).
```

Additivity gives $V_S=\sum_{j\in S}v_j$. Uniform fixed-size selection gives subset covariance equal to the centered coordinate cross-product sum times $k(d-k)/[d(d-1)]$. Both subset variances use the same factor times their respective centered squared sums. Pearson divides covariance by the two standard deviations, so the factor cancels:

```math
FC_{\rm pop}(k)=\operatorname{Corr}_{j=1}^{d}(a_j,v_j),\qquad 1\leq k<d.
```

This is the project's [HYP-01 derivation](ATLAS_REFERENCE.md#hyp-01), not a theorem attributed to the metric's definition paper. It applies to arbitrary supplied $a$ under the assumptions, including the two nonexact candidates and additive nonlinear models. It excludes constant vectors, $k=0$, and $k=d$.

**Independent checks matter.** Computing the same coordinate formula twice would not test the subset evaluator. The implementation also computes actual model changes for singleton replacements and enumerable subset cases, then compares those correlations with the analytical reference. The exhaustive audit covers the three-feature teaching fixture and the first generated 20-feature input at the run's configured subset sizes; it saves the subset counts and discrepancy. Singleton FE is checked for every generated input. Large random-input runs use the analytical reference after those checks; they do not enumerate every subset for every input.

For the original three-feature teaching example, the complete subset values are:

| $k$ | Subsets in order | Attribution sums $U$ | Model changes $V$ |
|---|---|---|---|
| 1 | $\{1\},\{2\},\{3\}$ | $(0,4,6)$ | $(6,4,0)$ |
| 2 | $\{1,2\},\{1,3\},\{2,3\}$ | $(4,6,10)$ | $(10,6,4)$ |

Both population correlations are $-13/14$. These are hand-derived teaching values, not simulation observations. Sampling only some of these subsets can produce another estimate or no defined correlation.

#### 4. Separate the sample sizes and preserve pairing

| Symbol | Counts what? | Interpretation |
|---|---|---|
| $k$ | Features replaced in one draw. | A subset setting, not the explanation's support size. |
| $B$ | Draws forming one estimated correlation. | The within-input sampling budget. |
| $R$ | Independently generated draw plans for one input. | Measures evaluator Monte Carlo variation. |
| $N$ | Independently generated inputs. | Replication units for conclusions about the input distribution. |

For every draw, generate an independent uniform feature permutation. Its first $k$ entries give a uniform size-$k$ subset. Reuse that permutation across subset sizes and explanations; use the first $B$ draws for each budget. Thus, each setting retains its intended sampling law while the comparisons share randomness. Budgets and subset-size estimates are dependent by design. Inputs and evaluator draw plans use separately recorded random streams. See [CTRL-PAIRING](ATLAS_REFERENCE.md#ctrl-pairing).

| Run | Subset sizes | Budgets | Purpose |
|---|---|---|---|
| Smoke | $1,5,10,19$ | $5,25,100$ | Check the complete pipeline with $N=4$, $R=4$. |
| Development | $1,5,10,19$ | $25,100,400$ | Examine estimator behavior with $N=128$, $R=20$. |
| Confirmation | Reserved $2,7,13$ | $25,100,400$ | Test the frozen prediction on fresh inputs/draws with $N=512$, $R=20$. |

The larger counts retain the Atlas planning amounts. They are **computational starting points, not a power calculation**. The smoke run does not establish research reliability. Development results should inform whether precision is adequate before confirmation is frozen. A newly chosen seed at an old setting checks repeatability; reserved $k$ values test new settings. Do not inspect the confirmation outcomes and then select the candidate pair or revise the prediction.

#### 5. What the analysis can conclude

The report keeps the following distinct:

- **Population discrepancy:** exhaustive subset FC minus the analytical reference. The numerical tolerance checks arithmetic and implementation; it is not a practical equivalence margin.
- **Finite-estimator error:** estimated FC minus its per-input population reference. Summaries include conditional bias, RMSE, variability, and validity counts. More budget need not improve every individual realization.
- **Pairwise margin:** estimated A score minus estimated B score. A practical tie tolerance of **0.01 FC units** is an operational reporting choice, not a validated threshold of explanation quality or an equivalence claim.
- **Condition contrast:** compare matched settings using repetitions jointly valid across the relevant arms. Report their exact eligibility set and exclusion counts. Four-arm eligibility is used when comparing A-versus-B judgments across two settings.

The primary across-input summaries weight inputs equally after the within-input paired calculations. Bootstrap intervals resample whole inputs while retaining their paired condition/repetition summaries. These are descriptive, pointwise intervals, not a multiple-testing correction or formal equivalence procedure. Subset draws are not treated as independent inputs. Valid-only means describe a **conditional finite-score estimand**: the average among comparisons for which the required scores can be computed. Condition-dependent failures can change who is represented. Report validity alongside the estimates and retain the analytical population contrast for context. An uncertainty interval spanning zero is unresolved evidence, not proof of a population tie. [ANA-OUTCOMES](ATLAS_REFERENCE.md#ana-outcomes), [ANA-UNCERTAINTY](ATLAS_REFERENCE.md#ana-uncertainty)

A finite-budget score shift alone does not refute HYP-01. A reproducible exact-enumeration discrepancy beyond numerical tolerance, with the premises verified, requires checking the implementation and derivation. Interactions or nonuniform selection deliberately violate the premises and belong to boundary controls. **DMS/Response Alignment is not used here:** subset size and budget do not define a monotone explanation-quality intervention, and the population prediction is invariance.

#### 6. What each file does

| File or module | Purpose and reason for keeping it separate |
|---|---|
| `notebooks/EXP-01.ipynb` | Walks through the design, runs the same engine, and inspects artifacts/plots. It does not contain a second scoring implementation. |
| `configs/exp01_*.yaml` | Makes model, budgets, seeds, tolerances, and run stage reviewable before execution. |
| `core.py` | Defines the oracle, exact/relocated explanations, draw plans, actual replacement traces, checked Pearson, enumeration, and singleton FE. These are the quantities tested by the hypothesis. |
| `analysis.py` | Summarizes errors and paired contrasts while preserving validity and the input/repetition hierarchy. |
| `experiment.py` | Connects the declared configuration to the run, writes artifacts, and freezes/checks confirmation provenance. |
| `plotting.py` | Displays population scores, estimator error, and undefined outcomes with explicit axes. |
| `__main__.py` | Provides the `run` and `freeze` command-line entry points. |
| `tests/test_exp01.py` | Checks mathematical identities, pairing, degeneracy, and boundary cases; passing these checks is not evidence of real-data transfer. |
| `scripts/audit_quantus.py` | Separately checks the pinned Quantus implementation; see [the audit](QUANTUS_AUDIT.md) for what was actually inspected or run. |

The Python modules are in `src/xai_metric_behavior/`. The main evaluator is an explicit, traceable implementation of the declared FC/FE calculation. Any claim that it agrees with a particular Quantus release must be supported by the separate versioned parity checks; it is not implied by using the same metric name.

**Saved outputs:**

| Output | How to use it |
|---|---|
| `manifest.json`, `resolved_config.yaml` | Identify the code/configuration, versions, streams, run status, and resolved measurement rules. |
| `inputs.npz` | Recover the generated inputs and explanation/reference data. |
| `population_scores.csv`, `exact_enumeration.csv` | Inspect analytical references, singleton checks, and independent enumerable comparisons. |
| `scores.csv` | Read every attempted per-input, per-repetition, per-condition score and its validity. |
| `input_summary.csv`, `score_summary.csv` | Inspect within-input behavior and equal-weight input summaries. |
| `pairwise_margins.csv` | Inspect A-minus-B margins and point judgments for each setting/repetition. |
| `contrast_records.csv`, `contrast_summary.csv` | Recover jointly valid repetition membership, matched condition differences, and input-level intervals. |
| `failures.csv` | Count undefined scores and other recorded failures by reason; do not replace them with zero. |
| `traces/input_XXXX.npz` | Reconstruct sampled permutations, subset prefixes, attribution sums $U$, output changes $V$, and perturbed outputs. |
| `figures/` | Inspect plots; consult the CSVs and trace data for exact values. |

The trace stores the largest budget once. Smaller budgets are prefixes of the same sequence; this saves space while retaining every realized draw. Run directories are preserved separately and existing outputs are not silently overwritten.

#### 7. Execute the experiment

Before this PR is merged, use the `exp01-atlas-protocol` branch. From the repository root, with Python 3.12:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m xai_metric_behavior run --config configs/exp01_smoke.yaml --output results/exp01-smoke
```

Start by reading the smoke manifest, failure counts, population check, and plots. The default experiment requires only CPU libraries; Quantus is a separate optional audit dependency. The notebook offers the same sequence with explanatory cells.

Run development after the smoke check:

```bash
python -m xai_metric_behavior run --config configs/exp01_development.yaml --output results/exp01-development
```

Review the population discrepancies, finite-estimator errors, validity rates, and paired margins. Keep the configuration and output directory together. A failure row may be an expected zero-variance estimate; read its reason before diagnosing a software problem. Use a new directory when rerunning.

**Confirmation is a separate prospective step.** After reviewing development precision and fixing the intended prediction, create the frozen specification:

```bash
python -m xai_metric_behavior freeze --config configs/exp01_confirmation.yaml --development-run results/exp01-development --output protocol-freeze.json
python -m xai_metric_behavior run --config configs/exp01_confirmation.yaml --output results/exp01-confirmation --freeze protocol-freeze.json
```

The freeze binds the confirmation configuration, code, and identified development run. Keep it with the later outputs. If scientific code or the protocol changes after freezing, resolve the mismatch and create a new prospective specification before examining the reserved results. If a prediction is revised after inspecting confirmation, record that revision and reserve fresh conditions for its next confirmation. A filesystem gate records workflow provenance; it cannot prove that a person has never examined the held-out settings elsewhere.

**Reading the plots:**

| Figure | Horizontal axis | Vertical axis and scope |
|---|---|---|
| `population_vs_subset_size.png` | Subset size $k$. | Population FC for the **first sampled input only**; analytical lines and exact-enumeration markers. It is a fixture check, not an average over inputs. |
| `estimator_error_vs_budget.png` | Budget $B$, on a logarithmic scale; separate curves for $k$. | Conditional bias in the top row and conditional RMSE in the bottom row, for A and B. Inputs receive equal weight. Shading shows pointwise input-cluster intervals for bias. |
| `undefined_fraction.png` | Budget $B$; separate curves for $k$. | Undefined scores divided by all attempted scores in that condition, shown separately for exact, A, and B. |

A smaller estimation error is evidence about estimator precision under this contract. Neither an unchanged population curve nor a small error establishes universal metric validity.

### Failure report

<!-- EXECUTION_STATUS_BEGIN -->

Execution evidence is being collected during implementation. Replace this block with identified run paths, actual check counts, observed discrepancies, and any environment/implementation failures before treating the report as complete.

| Area | Evidence status |
|---|---|
| Mathematical prediction | **Derived** in Atlas HYP-01, conditional on the stated assumptions. |
| Engine and verification checks | Implementation verification pending final recorded execution. |
| Smoke/development outcomes | No outcome claimed in this draft. |
| Quantus compatibility | See the separate audit for inspected source and executed/blocked parity status. |
| Confirmation | **Unexecuted**; reserved conditions remain separate from development. |

<!-- EXECUTION_STATUS_END -->

**Failure interpretation rules:** a constant sampled attribution sum or model response makes Pearson undefined; this is an outcome of the finite estimator, not zero faithfulness. An exact-enumeration disagreement under verified premises is a calculation/derivation discrepancy to investigate. Missing packages or execution errors are implementation/environment failures. An inconclusive interval is a statistical limit. Record these separately, retaining attempted/valid/nonfinite/exception counts and reasons. A failure does not automatically establish that the metric is invalid.

## Research Learning Summary

### Population value versus finite estimate

- **필요성:** subset 크기에 따른 측정값 변화와 적은 추출 횟수 때문에 생긴 추정 오차를 구분해야 한다.
- **직관적인 예시/이해:** 모든 가능한 subset을 계산한 점수는 같아도, 그중 일부만 뽑으면 점수가 다르게 나올 수 있다.
- **공식적인 정의:** $FC_{\rm pop}=\operatorname{Corr}_S(U_S,V_S)$; $\widehat{FC}_B$는 $B$개 추출 결과의 Pearson correlation이다.
- **근거/그게 되는 이유:** additive model과 uniform fixed-size sampling에서는 공분산과 두 분산에 같은 계수가 곱해져 population correlation에서 상쇄된다. 유한 표본의 비선형 추정 오차까지 사라지는 것은 아니다.

### Exact anchor and two nonexact candidates

- **필요성:** 정확한 설명만 사용하면 sampling noise나 순위 변화가 제대로 드러나지 않을 수 있다.
- **직관적인 예시/이해:** exact 설명은 정의되는 경우 항상 1을 받는다. 서로 다르게 잘못 배치한 A와 B를 추가해야 유한 표본의 판단 변화를 볼 수 있다.
- **공식적인 정의:** A는 $v_1$을, B는 $v_1,v_2$를 unused coordinates로 이동한다. Population margin은 $v_2^2/D_v$다.
- **근거/그게 되는 이유:** 이동은 attribution 값의 분포와 norm을 유지하지만 실제 feature effect와의 대응을 바꾼다. 후보 선택과 margin의 근거는 확인용 결과를 보기 전에 고정한다.

### Pairing and confirmation

- **필요성:** 비교 대상마다 우연히 다른 subset을 골라 생기는 차이와 새로운 조건에서의 예측 성공을 구분해야 한다.
- **직관적인 예시/이해:** 같은 feature 순서를 공유해 각 $k$에 맞는 앞부분을 사용하고, 확인용 subset 크기는 별도로 남겨 둔다.
- **공식적인 정의:** 반복마다 독립 uniform permutation을 만들고 $k$와 $B$에 따라 prefix를 사용한다. Confirmation은 frozen prediction과 reserved conditions를 연결한다.
- **근거/그게 되는 이유:** permutation prefix는 각 조건의 uniform subset 법칙을 유지한다. Fresh seed는 반복 가능성을, reserved setting은 예측 범위를 확인하며 두 증거는 다르다.
