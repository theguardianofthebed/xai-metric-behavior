# Experiment outputs

Each run writes to a **new directory**. Keep its configuration, code/dependency provenance, inputs, returned metric scores, actual draw traces, failures and summaries together.

## Current evidence

`verification/` is reserved for the **fresh Quantus-primary** implementation check: the official smoke run, test/audit records and summary. These are implementation-verification artifacts, not a full development study or held-out confirmation. Fresh execution is pending while the migration is prepared.

Normal smoke/development/confirmation output directories are ignored by Git because full traces can be large. Publish complete versioned run directories when sharing a study; a plot alone cannot reconstruct the calculation. Keep the frozen prediction file with confirmation outputs.

## Read a run

| Files | Purpose |
|---|---|
| `manifest.json`, `resolved_config.yaml` | Run status, measurement engine, resolved rules, versions, hashes and random streams. |
| `scores.csv`, `failures.csv` | Every attempted public Quantus FC score and any undefined/exception outcome. |
| `metric_calls.jsonl` | Every FC/FE public attempt: returned score, checks, warnings, hashes and pairing outcome. |
| `traces/`, `inputs.npz` | Generated inputs; FC `input_*.npz` records `subsets_k*`, U/V and outputs. Separate `FE_input_*.npz` files retain singleton order and actual model inputs. |
| `population_scores.csv`, `exact_enumeration.csv` | Auxiliary analytical references, singleton checks and declared exhaustive comparisons. |
| `input_summary.csv`, `score_summary.csv` | Within-input summaries and equal-weight input summaries. |
| `pairwise_margins.csv`, `contrast_records.csv`, `contrast_summary.csv` | A/B margins, joint eligibility and paired condition comparisons. |
| `figures/` | Population check, estimation error and undefined fraction; numerical values remain in the tables. |

## Historical evidence

The previous NumPy-primary verification is preserved unchanged in [archive/exp01-numpy/verification](../archive/exp01-numpy/verification/). Its results describe that earlier implementation, not the new Quantus run. Do not combine or relabel the two evidence sets. The original Phase 1 scaffold remains separately archived.
