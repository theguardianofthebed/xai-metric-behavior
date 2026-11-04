# Experiment outputs

Each run writes to a **new directory**. Keep its configuration, code/dependency provenance, inputs, returned metric scores, actual draw traces, failures and summaries together.

## Current evidence

[`verification/`](verification/) contains the latest **completed Quantus smoke**, run `0221db61739ed30a`, from clean source [`ce08d31`](https://github.com/theguardianofthebed/xai-metric-behavior/commit/ce08d3181f9325b82e91b9bb1b1585e10c787ab7). Its 576 FC attempts include 559 finite returns (555 eligible, 4 near-constant), 17 undefined and 0 exceptions. All 12 singleton FE returns are eligible.

The test/audit records and unchanged notebook-cell replay verify implementation; they are not additional research results. The three key notebook output files matched canonical smoke byte-for-byte. Hosted Colab and interactive Jupyter remain unverified. See [summary.json](verification/summary.json) for precise scope/provenance. Full development and confirmation remain unexecuted.

Normal smoke/development/confirmation output directories are ignored by Git because full traces can be large. Publish complete versioned run directories when sharing a study; a plot alone cannot reconstruct the calculation. Keep the frozen prediction file with confirmation outputs.

## Read a run

| Files | Purpose |
|---|---|
| `manifest.json`, `resolved_config.yaml` | Run status, measurement engine, resolved rules, versions, hashes and random streams. |
| `scores.csv`, `failures.csv` | Every official FC score; `valid` means finite, while `analysis_eligible` excludes undefined/exception and warned near-constant cases. Preserve `numerical_instability` and reasons. |
| `metric_calls.jsonl` | Every FC/FE public attempt: returned score, checks, warnings, hashes and pairing outcome. |
| `traces/`, `inputs.npz` | Generated inputs; FC `input_*.npz` records `subsets_k*`, U/V and outputs. Separate `FE_input_*.npz` files retain singleton order and actual model inputs. |
| `population_scores.csv`, `exact_enumeration.csv` | Auxiliary analytical references, singleton checks and declared exhaustive comparisons. |
| `input_summary.csv`, `score_summary.csv` | Eligible within-input/equal-weight summaries; separate finite, eligible, undefined, numerical-instability, exception and total-ineligible counts. |
| `pairwise_margins.csv`, `contrast_records.csv`, `contrast_summary.csv` | A/B margins, joint eligibility and paired condition comparisons. |
| `figures/` | Population check, eligible estimation error and `analysis_ineligible_fraction.png`; separate undefined/numerical-instability panels use all attempts. |

Reason counts are incidences and may overlap; `n_ineligible` counts their union. Conditional summaries use `*_given_eligible` or `*_given_joint_eligible`, with pairing preserved.

## Historical evidence

The previous NumPy-primary verification is preserved unchanged in [archive/exp01-numpy/verification](../archive/exp01-numpy/verification/). Its results describe that earlier implementation, not the new Quantus run. Do not combine or relabel the two evidence sets. The original Phase 1 scaffold remains separately archived.

The [first Quantus attempt](../archive/exp01-quantus-first-attempt/) is also unchanged: its near-constant warning exposed the overly strict auxiliary parity guard. It is a failure record, not the latest completed smoke.
