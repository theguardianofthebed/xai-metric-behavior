# Quantus contract and implementation checks

**Quantus 0.6.0 public `FaithfulnessCorrelation` and `FaithfulnessEstimate` calls supply EXP-01's reported metric scores.** [`quantus_backend.py`](../src/xai_metric_behavior/quantus_backend.py) records the public calls; project code supplies the known model, explanations, references and analysis. It never silently substitutes a NumPy score for a failed official call.

This guide describes the pinned implementation contract. **Completed-rerun verification is pending**; [current verification](../results/verification/) will report it. The first attempt is preserved in [its own archive](../archive/exp01-quantus-first-attempt/). Earlier NumPy-primary tests and audit results remain in [archive/exp01-numpy](../archive/exp01-numpy/).

## 1. Explicit metric settings

The upstream source target is Quantus 0.6.0 at commit [`85bcf2c`][quantus]. The adapter verifies Quantus's version and eight selected source files against that commit's Git blobs, then records their SHA-256 hashes and package versions. This does not verify every byte of every dependency.

| Choice | EXP-01 setting and reason |
|---|---|
| Attribution processing | `abs=False`, `normalise=False`: retain signed scalar contributions. |
| Replacement | Official `batch_baseline_replacement_by_indices`, numeric `perturb_baseline=0.0`; each draw starts from the original input. |
| Comparison | Official `correlation_pearson`; FC compares subset attribution sums with actual model-output differences. |
| Output and aggregation | `softmax=False`, `return_aggregate=False`; a fixed one-output scalar model and per-input results. Target index 0 selects that sole output, not a classification decision. |
| FC / FE settings | FC uses `subset_size=k`, `nr_runs=B`. FE uses `features_in_step=1`, replacing each feature individually; it is not cumulative deletion. |
| Generic startup banner | `disable_warnings=True` skips the parameter banner and its delay. Runtime warnings and undefined outcomes remain recorded. |

FE sorts observations by attribution, but singleton replacement changes only their order, so the paired Pearson correlation has the same analytical reference. Its docstring mentions Spearman while the executable default is Pearson; the contract sets Pearson explicitly. See the pinned [FC][fc] and [FE][fe] sources.

## 2. Why the adapter is necessary

**Environment.** Quantus [package initialization][init] imports optional ML frameworks when installed. The notebook uses an isolated CPU environment so Colab's preinstalled TensorFlow/Captum and already-loaded NumPy do not enter the experiment interpreter. It installs the full package there and runs the same CLI.

**Shape.** Pinned FC validation rejects valid `(N,d)` tabular inputs because a later size check uses the empty trailing shape. Pass `(N,1,d)` and flatten within the scalar model. This preserves the same d features and uses the original validator. [Validator][asserts]

**Precision.** Quantus's [PyTorch wrapper][torch-wrapper] constructs `torch.Tensor(x)`, whose default floating dtype is normally float32. Converting to double only inside the model cannot recover input bits already lost. The adapter temporarily selects float64 through the supported [PyTorch setting][dtype], uses float64 model arithmetic, verifies the actual received inputs, and restores the prior setting. This runtime choice preserves the declared generated input; it does not change the metric formula.

**Baseline.** The named `"black"` baseline uses the input minimum in this version; it is not a synonym for zero. Numeric `0.0` keeps replacement aligned with the contribution reference and avoids an unintended sampling dependency. [Baseline handling][utils]

**Pairing.** Quantus samples native size-k subsets without replacement within each draw. With pinned NumPy 2.3.5 and unchanged batch/order, its legacy RNG path uses a permutation prefix for `choice(..., replace=False, p=None)`. Replaying the same seed therefore supports coupling under the deterministic zero-baseline contract. The decisive check is still the recorded masks: candidates must share draws, smaller k must match the appropriate feature prefix, and smaller B must match the draw prefix. Save and restore the caller's RNG state; do not inject masks or replace the official sampler. [NumPy source][numpy-choice]

The scoped RNG/dtype controls belong to serial execution; they are not a guarantee for concurrent calls sharing process-global state.

## 3. Traces, references and parity answer different questions

Recording wrappers delegate to the official replacement and Pearson functions. They retain the selected coordinates, original/replaced inputs, U/V passed to Pearson, and the returned score. Actual model calls provide prediction traces. Quantus remains responsible for the metric calculation and returned value.

| Check | What it establishes |
|---|---|
| Analytical / exhaustive reference | Under HYP-01's assumptions, does subset correlation agree with coordinate correlation? Exhaustive scope is declared, not every input. |
| Same-trace NumPy replay | Check saved subsets, replacements and U/V strictly; assert ordinary score agreement. Warned near-constant scores retain a diagnostic discrepancy without that assertion. |
| Pairing / source / precision guards | Did this run retain the pinned calculation, intended random coupling and input values? |
| Fresh smoke | Does the pipeline preserve official returns, numerical eligibility, warnings and failures? It is not held-out confirmation. |

Parity is an **implementation guard**, not a new empirical result about explanation quality or an independent proof of the population identity. Keep reported Quantus scores unchanged even when an auxiliary check fails; record the discrepancy and classify the run accordingly. Undefined Pearson values remain nonfinite outcomes with reasons, not replacement scores of zero.

**Observed near-constant case and revised rule.** The first official attempt (source `02cd3d2`, input 2, repeat 1, A, k=19, B=5) stopped at the project's strict replay guard. U had a spread of only `8.881784197001252e-16`; Quantus returned `0.31622776601683805` with `NearConstantInputWarning`, while NumPy replay gave `0.2581988897471609`. Near-zero spread makes Pearson sensitive to rounding; neither finite calculation should be treated as a reliable estimate. [SciPy's warning semantics][pearson-warning]

Before rerunning, the contract now separates `valid` (official finite return) from `analysis_eligible`. The recorded warning makes the estimate ineligible and marks `numerical_instability`. Its official score, warning, actual subsets/U/V and diagnostic discrepancy are retained; `score_parity_status=not_asserted_near_constant`. All structural checks remain strict. Ordinary score parity still uses `1e-10`; there is no score replacement, resampling, or increased tolerance. The regression test reproduces the archived call.

Summary columns explicitly say `*_given_eligible` or `*_given_joint_eligible`. Report official finite, undefined, numerical-instability, exception and total-ineligible counts against all attempts. Reason incidence can overlap across arms; total ineligible is the union, not their sum.

## 4. Function names do not guarantee the same meaning

The professor's requested utility inspection matters because changing one helper can change the contract. The [perturbation][perturb] and [similarity][similarity] sources distinguish:

| Function family | Interpretation |
|---|---|
| Baseline replacement | The selected coordinates receive a declared value; EXP-01 uses the batch variant and literal zero. |
| Gaussian / uniform noise | Adds noise instead of deleting to a fixed reference; different intervention. |
| `no_perturbation` | Leaves every response difference zero, so this Pearson contract is undefined. |
| Pearson / Spearman / Kendall | Linear association, rank association, and ordering agreement are different comparisons. |
| `cosine` | Returns SciPy cosine **distance**: identical directions give 0, not maximum similarity. |
| Euclidean / Manhattan / Lipschitz | Distance or normalized-change calculations; they are not interchangeable names for FC. |

## 5. Recheck and read the evidence

Install the required environment through the [run guide](EXP-01.md#6-what-physically-implements-the-design), then run the checks separately from the research grids:

```bash
python -m unittest discover -s tests -v
python scripts/audit_quantus.py --output results/quantus-audit.json
```

Use a fresh audit output path. Inspect package/source identity, check scope, failures and skipped checks; a missing dependency or skipped path is not a pass. New run counts and notebook verification will be recorded in [the current summary](../results/verification/summary.json). Full development and reserved confirmation remain unexecuted.

**Research Learning Summary:** 필요성 → 공식 metric과 우리가 기록한 계산이 일치하는지 확인한다. 직관적인 예시/이해 → Quantus의 계산 과정을 옆에서 기록하고 같은 기록으로 다시 계산한다. 공식적인 정의 → public metric의 반환값은 결과, trace parity는 그 기록의 재현 점검이다. 근거/그게 되는 이유 → wrapper 오류는 탐지할 수 있지만, 두 계산의 일치만으로 설명 품질이나 일반적인 metric 신뢰성이 증명되지는 않는다.

[quantus]: https://github.com/understandable-machine-intelligence-lab/Quantus/tree/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a
[fc]: https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/faithfulness/faithfulness_correlation.py
[fe]: https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/faithfulness/faithfulness_estimate.py
[asserts]: https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/helpers/asserts.py
[torch-wrapper]: https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/helpers/model/pytorch_model.py
[dtype]: https://docs.pytorch.org/docs/2.10/generated/torch.set_default_dtype.html
[utils]: https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/helpers/utils.py
[numpy-choice]: https://github.com/numpy/numpy/blob/v2.3.5/numpy/random/mtrand.pyx
[perturb]: https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/functions/perturb_func.py
[similarity]: https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/functions/similarity_func.py

[init]: https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/__init__.py

[pearson-warning]: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.pearsonr.html
