# EXP-01 — Quantus FC: same explanation, different estimates

This report follows the [Atlas revision-9 Quantus contract](atlas_quantus_contract_2026-11-04.json), captured before the corrected rerun. [Bhatt et al.](https://doi.org/10.24963/ijcai.2020/417) defines the subset-correlation family; [HYP-01](ATLAS_REFERENCE.md#hyp-01) is the project's conditional derivation.

### 1. Question / RQ

**If the explanation stays fixed, does replacing more features at once (k) change population FC, or only the finite estimate obtained from B sampled groups?**

This addresses [RQ-01](ATLAS_REFERENCE.md#rq-01) within [RQ-00](ATLAS_REFERENCE.md#rq-00). Population FC considers every allowed size-k subset for one input; the Quantus estimate uses B sampled subsets. HYP-01 predicts population invariance under the additive contract; finite estimates can fluctuate or be undefined.

### 2. Measurement Contract — what is fixed or varied?

| Item | Rule |
|---|---|
| Model / data | 20 independent standard-normal features; $g(x)=2.5x_1-2x_2+1.5x_3+x_4-0.5x_5$. Five active features; the fixed scalar formula gives exact contributions without training. |
| Explanation meaning: C | Signed zero-relative contributions $v_j=w_jx_j$. Preserve signs; no attribution normalization or probability transform. |
| Primary evaluator | **Quantus 0.6.0 public FC / singleton FE calls**. Explicit abs=False, normalise=False, return_aggregate=False, softmax=False. |
| Evaluator rules: theta | Uniform size-k subsets; replace selected features by numeric **0.0 in the original input**. FC forms attribution sums U and output changes V; official baseline replacement and Pearson functions perform replacement and comparison. |
| What changes | k = features per draw; B = draws per FC score. Smoke: k = {1,5,10,19}, B = {5,25,100}, N = 4 inputs, R = 4 repeats. |
| Randomness: xi | Seed Quantus before each paired call; record actual subsets. Verify identical draws across candidates, nested k prefixes and B prefixes. Preserve native sampling; do not inject masks. |
| Reporting | Average repeats within each input, then weight inputs equally. Compare jointly eligible repeats; resample inputs. Retain official finite/nonfinite returns, warnings and eligibility reasons. |

Use `(N,1,d)` input plus model flattening for the pinned tabular compatibility issue. Preserve float64 model inputs and arithmetic through a scoped runtime setting, with restoration afterward. These are declared adapter choices; [the Quantus contract](QUANTUS_AUDIT.md) explains why they are necessary.

```math
U_S=\sum_{j\in S}a_j,\qquad V_S=g(x)-g(x_{S\leftarrow0}),\qquad
\widehat{FC}_B=\mathrm{Corr}_{r=1}^{B}(U_{S_r},V_{S_r}).
```

### 3. Proposed experiment design and flow

**Exact:** the true contribution vector. **A:** move contribution 1 to unused feature 6. **B:** also move contribution 2 to unused feature 7. Feature numbers are 1-based. Values, sum, sparsity and norms stay fixed; locations change. Defined exact-anchor scores should be 1, hiding estimator noise. A/B expose it; fix them before inspecting reversals.

```text
[Fix contract; prepare inputs and Exact / A / B]
                         ↓
[Derive population reference; enumerate selected cases]
                         ↓
[Call Quantus FC / FE; record actual masks, U, V, scores]
                         ↓
[Check paired draws + NumPy trace parity; retain failures]
                         ↓
[Compare estimator error and A−B gaps]
                         ↓
[Later: development review → freeze → held-out confirmation]
```

**Derived:** $FC_{\mathrm{pop}}(k)=\mathrm{Corr}_j(a_j,v_j)$ for positive coordinate variances and $1\leq k<20$. Additivity gives subset responses as sums of effects; uniform fixed-size sampling gives covariance and both variances a common factor that cancels in Pearson.

Enumeration checks the teaching fixture and first 20-feature input at configured k; tolerance $10^{-10}$ checks numerical consistency. Official singleton FE checks every input. The A/B population margin is $v_2^2/\sum_j(v_j-\bar v)^2\geq0$; its size varies by input and can be practically small.

**Auxiliary NumPy checks:** replay recorded subsets/U/V to check the adapter; Quantus's return remains the score. Strict score parity is deferred only for an upstream near-constant warning, retaining its diagnostic discrepancy. Structural checks stay strict. Analytical/enumeration references address the population prediction.

Development: N = 128, R = 20, k = {1,5,10,19}, B = {25,100,400}. Confirmation uses N = 512, R = 20, B = {25,100,400}, reserved k = {2,7,13}, and fresh streams. These are planning counts, not a power calculation. Before confirmation, freeze the prediction, margins, analysis/failures, code/configuration, backend execution hash (versions/source files), and development provenance.

### 4. Latest result — Smoke

**Quantus rerun pending.** New counts, discrepancies and the run identity will come from [fresh verification artifacts](../results/verification/). The earlier NumPy-primary smoke is superseded and preserved in [archive/exp01-numpy](../archive/exp01-numpy/); it is not evidence that this Quantus implementation has run.

### 5. Limitations / failure report

| Status / issue | Meaning and handling |
|---|---|
| **Observed — near-constant input; rule updated** | The first attempt stopped at input 2 / repeat 1 / A / k=19 / B=5: U varied by only $8.88\times10^{-16}$. Quantus returned **0.31623** with a warning; NumPy trace replay gave **0.25820**. Keep the official value and mark it analysis-ineligible. [Record](../archive/exp01-quantus-first-attempt/). |
| **Expected possible outcome — undefined Pearson** | Constant sampled U or V has no defined correlation. Preserve NaN and the reason; do not score it as zero. |
| **Adapter checks required** | Verify channel shape, literal-zero baseline, float64 inputs, native draw coupling and unchanged official functions. |
| **Unverified — hosted notebook UI** | Prepare a Colab front end to the same engine and state exactly how it was exercised. |

**Eligibility:** retain every official score; exclude undefined values, exceptions, and upstream `NearConstantInputWarning` cases from statistical summaries. A finite return can still be ineligible. This rule was added after the first attempt; it preserves samples, scores and the $10^{-10}$ tolerance.

Summaries are conditional on eligibility, which can differ by setting. Compare ranks only where both candidates at both settings are eligible. Report all-attempt denominators; reason counts can overlap, while total ineligible counts the union. Four inputs do not establish general failure rates, real-data transfer or held-out success. Intervals are descriptive/pointwise; 0.01 is an operational tie threshold, not an equivalence bound. Development and confirmation remain unexecuted.

An enumeration discrepancy under verified premises requires investigating the calculation/derivation; a finite-score shift alone does not refute HYP-01. Interactions, nonuniform subsets, nonlinear output transforms and subset-dependent baselines require separate contracts.

### 6. What physically implements the design?

| Component | What it does and why |
|---|---|
| [`quantus_backend.py`](../src/xai_metric_behavior/quantus_backend.py) | Calls official FC/FE and records actual inputs/returns through delegating wrappers. It supplies the reported metric scores. |
| `core.py` | Known model, candidates, analytical references, enumeration and auxiliary trace replay. It supplies checks, not replacement scores. |
| `analysis.py`; `plotting.py` | Preserve paired eligibility and input-level uncertainty; display eligible estimator error and ineligible outcomes. |
| `configs/`; `experiment.py` | Declare settings; save manifests, traces and failures; enforce the confirmation freeze. |
| CLI; `notebooks/EXP-01.ipynb` | Same engine; the notebook runs the CLI in an isolated CPU environment and only inspects its outputs. |

**Execute on CPU:** use Python 3.12 in a fresh environment, from the repository root. Quantus and CPU PyTorch are required; no training, external dataset or GPU is needed. Install CPU PyTorch first (Linux/Colab), then the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m xai_metric_behavior run --config configs/exp01_smoke.yaml --output results/exp01-smoke
```

The [README](../README.md#run-on-cpu) includes the clone command. [Open in Colab](https://colab.research.google.com/github/theguardianofthebed/xai-metric-behavior/blob/exp01-atlas-protocol/notebooks/EXP-01.ipynb) for setup and a readable smoke run. Use a new output directory per run.

Read `manifest.json` for provenance, `scores.csv` / `failures.csv` for FC outcomes, `metric_calls.jsonl` for every official FC/FE call, and `traces/` for actual subsets/U/V. [Output map](../results/README.md). Keep the whole run directory.

**Plot axes:** population FC versus k shows the first input; bias/RMSE versus B uses eligible scores. `analysis_ineligible_fraction.png` shows undefined and numerical-instability fractions in separate panels, each divided by all attempts. Inspect all three.

After smoke, run development (the full grid can take much longer):

```bash
python -m xai_metric_behavior run --config configs/exp01_development.yaml --output results/exp01-development
```

Review precision and fix the prediction before freezing and opening reserved results:

```bash
python -m xai_metric_behavior freeze --config configs/exp01_confirmation.yaml --development-run results/exp01-development --output protocol-freeze.json
python -m xai_metric_behavior run --config configs/exp01_confirmation.yaml --freeze protocol-freeze.json --output results/exp01-confirmation
```

Keep the freeze with its outputs. A revised prediction after seeing confirmation needs fresh reserved conditions. **EXP-01-R is an Atlas plan only; no real-data implementation is included.**

### 7. What insight do we get?

**Separate explanation differences from evaluator sampling effects.** A finite FC of 1 can occur for a nonexact explanation; its meaning depends on k, B, uncertainty and numerical eligibility. This is an invariance study, so DMS/Response Alignment is not used.

**Research Learning Summary:** 필요성 → 같은 설명의 점수 변화와 sampling 오차를 구분한다. 직관적인 예시/이해 → Quantus가 확인하는 feature 묶음과 횟수만 바꿔 채점한다. 공식적인 정의 → population FC는 모든 허용 subset을, finite FC는 B개 draw를 사용한다. 근거/그게 되는 이유 → additive model의 population Pearson에서는 공통 계수가 상쇄되지만 finite sample은 변동하거나 상수가 될 수 있다. Trace parity는 구현 점검이며, near-constant warning이 있는 finite 값도 보관하되 통계 분석에서는 제외한다.
