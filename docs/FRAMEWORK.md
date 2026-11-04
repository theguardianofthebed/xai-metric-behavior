# Research framework

## Question and motivation

> Under what controlled conditions do quantitative XAI evaluation metrics change their judgments, and can those changes be explained by the metric's measurement mechanism rather than by explanation quality itself?

This is [RQ-00](ATLAS_REFERENCE.md#rq-00), preserved from the current Atlas. Two explanations can receive different rankings because the comparison changed, because a random evaluator used different draws, or because another assumption failed. The research goal is to distinguish those reasons and make a testable prediction.

Prior work already studies evaluator reliability, configuration sensitivity, and failure mechanisms. [MetaQuantus](https://openreview.net/forum?id=j3FK00HyfU) and [Sanity Checks for Saliency Metrics](https://doi.org/10.1609/aaai.v34i04.6064) are direct precedents. The proposed contribution must be a narrower, controlled explanation of a particular behavior. Merely drawing a parameter-sweep curve or obtaining perfect linear-model calibration is insufficient as a novelty claim. The Atlas preserves the [prior-work qualifications](ATLAS_REFERENCE.md#ref-fidelity), including distinctions between reviewed publication versions.

## Four stages and why each exists

| Stage | What we prepare or calculate | Why it is needed | Atlas source |
|---|---|---|---|
| Measurement contract and predictions | Define the explanation's meaning, scalar target, reference baseline, evaluator, settings, randomness, and predicted response. | A score can only be interpreted relative to the calculation it represents. | [FW-01](ATLAS_REFERENCE.md#fw-01) |
| Oracle worlds and reference bank | Generate known model functions, inputs, exact baseline-relative contributions, and fixed nonexact candidates. | Establish the intended differences independently of the metric being tested. | [FW-02](ATLAS_REFERENCE.md#fw-02) |
| Metric evaluation with traces | Calculate per-input scores and save subsets, attribution sums, output changes, and failures. | A final score alone cannot locate the calculation responsible for a change. | [FW-03](ATLAS_REFERENCE.md#fw-03) |
| Diagnosis and held-out confirmation | Compare observations with predictions, remove a suspected cause where possible, and test reserved conditions. | An explanation fitted after seeing a curve is weaker evidence than a prediction that survives a new condition. | [FW-04](ATLAS_REFERENCE.md#fw-04) |

The research question motivates the researcher's choice of measurement rules; it does not mathematically determine one unique contract. During development, revise unsuitable rules openly. Freeze the applicable prediction before confirmation.

## The measurement contract in plain language

| Object | Meaning | EXP-01 example |
|---|---|---|
| $C$ | What the explanation means and which property is being examined. | Signed contributions to a fixed scalar output relative to zero. |
| $\theta$ | How the evaluator is configured. | Uniform size-$k$ subsets, $B$ draws, zero replacement, Pearson comparison. |
| $\xi$ | Which random choices actually occurred. | The recorded list of sampled subsets in one repetition. |

The **explanation reference** is the baseline used to define a contribution. The **reference explanation** is the resulting attribution vector. The evaluator's replacement baseline is another explicit choice. Matching their names is not enough: check the actual values, target, signs, units, and expected relationship. Pearson association, numerical agreement, and rank agreement are different comparison goals. [RQ-02](ATLAS_REFERENCE.md#rq-02) gives a worked example.

## Scores, rankings, and uncertainty

For higher-is-better scores, a pairwise margin is the difference between two explanations evaluated under the same setting:

```math
D_\theta = Q_\theta(A)-Q_\theta(B).
```

A setting can change both scores while retaining the same preference. A strict reversal requires the margin to change sign beyond a predefined practical tie tolerance. Exact ties, practical ties, uncertain comparisons, and undefined scores are reported separately. A preferred score is not itself a proof that an explanation is better under the declared reference definition. [RQ-01](ATLAS_REFERENCE.md#rq-01) and [ANA-OUTCOMES](ATLAS_REFERENCE.md#ana-outcomes) define these distinctions.

Inputs and evaluator draws are different replication units. Many draws from one input do not become many independent inputs. Pair comparable evaluations before scoring, preserve each setting's sampling law, and retain common eligibility when comparing arms. Constant-vector correlations remain undefined. An interval that includes zero does not establish equivalence; practical invariance needs justified bounds and appropriate precision. See [ANA-UNCERTAINTY](ATLAS_REFERENCE.md#ana-uncertainty) and [CTRL-PAIRING](ATLAS_REFERENCE.md#ctrl-pairing).

## Experiment roadmap

| Experiment | Controlled question | Current role |
|---|---|---|
| **EXP-01** | Does subset size change the population score, the finite estimator, or both? | Current implementation target; [design and execution evidence](EXP-01.md). |
| EXP-02 | What changes when explanation reference and evaluator baseline differ? | Proposed; do not assume every mismatch lowers correlation. |
| EXP-03 | When do exact contribution allocations stop matching subset-deletion responses under interactions? | Proposed; distinguish additive nonlinear functions from interactions. |
| EXP-04 | What information survives sign reversal or scaling? | Proposed representation controls. |
| EXP-05 | How does relocating attribution change FC while preserving its value distribution? | Proposed full experiment; its fixed relocation construction supplies EXP-01's candidate pair. |

DMS/Response Alignment describe an expected monotone quality direction. **EXP-01 predicts population invariance**, so it uses exact discrepancies, estimator error, margins, and uncertainty instead. Forcing a monotonic-alignment score onto a flat prediction would answer the wrong question. [ANA-ALIGNMENT](ATLAS_REFERENCE.md#ana-alignment) retains the historical definitions and limitations.

## Evidence and scope

Use **proposed**, **derived**, **implemented**, **observed**, and **unverified** precisely. Atlas statuses describe the state of the specification; they do not prove scientific truth or execution. Archived Phase 1 values are historical context, not observations from this protocol.

An oracle is a model whose behavior is known well enough to derive a reference. It is not real-world causal truth. Evidence from the initial synthetic additive family does not automatically transfer to image explanations, probability outputs, feature interactions, alternative subset laws, or every metric grouped under Faithfulness. Compactness and Robustness/Continuity require their own contracts. [CTRL-SCOPE](ATLAS_REFERENCE.md#ctrl-scope)

The full [Atlas reference](ATLAS_REFERENCE.md) retains the original reasoning and qualifications. The [JSON snapshot](atlas_snapshot_2026-11-04.json) preserves the design baseline used for this implementation; it is not an automatically synchronized copy of future Atlas edits.
