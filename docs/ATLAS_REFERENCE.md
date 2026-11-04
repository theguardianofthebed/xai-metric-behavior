# Atlas source reference

This is the readable design snapshot of **XAI Metric Behavior — Framework & Experiments**, revision **4**, retrieved on **4 November 2026**. All 45 element summaries and author-written English sections are preserved. Heading levels are shifted only to nest them within this document. Original evidence labels and future tense are retained; statements that a protocol was unexecuted describe this source snapshot. Current implementation and run evidence are added separately in [EXP-01](EXP-01.md).

The [JSON snapshot](atlas_snapshot_2026-11-04.json) also retains all 162 connections, settings, stable IDs, and source references. This is a fixed design baseline, not a live synchronization feed. The [framework guide](FRAMEWORK.md) offers a shorter reading path.

## Index

| Category | Elements |
|---|---|
| Research Question | [RQ-00](#rq-00) · [RQ-01](#rq-01) · [RQ-02](#rq-02) · [RQ-03](#rq-03) |
| Framework | [FW-00](#fw-00) · [FW-01](#fw-01) · [FW-02](#fw-02) · [FW-03](#fw-03) · [FW-04](#fw-04) |
| Experimental Setting | [SET-01](#set-01) · [SET-02](#set-02) · [SET-03](#set-03) · [SET-04](#set-04) |
| Hypothesis | [HYP-01](#hyp-01) · [HYP-02](#hyp-02) · [HYP-03](#hyp-03) |
| Metric | [MET-FC](#met-fc) · [MET-FE](#met-fe) · [MET-MC](#met-mc) · [MET-RESIDUAL](#met-residual) |
| Experiment | [EXP-01](#exp-01) · [EXP-02](#exp-02) · [EXP-03](#exp-03) · [EXP-04](#exp-04) · [EXP-05](#exp-05) |
| Control | [CTRL-PAIRING](#ctrl-pairing) · [CTRL-PROVENANCE](#ctrl-provenance) · [CTRL-SCOPE](#ctrl-scope) |
| Analysis & Validation | [ANA-OUTCOMES](#ana-outcomes) · [ANA-UNCERTAINTY](#ana-uncertainty) · [ANA-DIAGNOSIS](#ana-diagnosis) · [ANA-CONFIRM](#ana-confirm) · [ANA-ALIGNMENT](#ana-alignment) |
| Reference Paper | [REF-BHATT](#ref-bhatt) · [REF-ALVAREZ](#ref-alvarez) · [REF-NGUYEN](#ref-nguyen) · [REF-YEH](#ref-yeh) · [REF-SHAPLEY](#ref-shapley) · [REF-METAQUANTUS](#ref-metaquantus) · [REF-TOMSETT](#ref-tomsett) · [REF-BINDER](#ref-binder) · [REF-LATEC](#ref-latec) · [REF-FIDELITY](#ref-fidelity) · [REF-MORRIS](#ref-morris) · [REF-LAKENS](#ref-lakens) |

<a id="rq-00"></a>

## RQ-00 — When and why do XAI metric judgments change?

Identify controlled conditions that change a metric's scores or rankings, and test whether its measurement mechanism explains those changes.

### Overview

- **Who:** The researcher evaluates selected quantitative XAI metrics.
- **What:** A question about the evaluator's behavior.
- **When:** Before choosing experiments and before interpreting their results.
- **Where:** Initially, synthetic tabular feature attribution with known model functions.
- **Why:** A changed score can reflect explanation differences, measurement choices, or finite sampling.
- **How:** Specify the construct, manipulate named factors, predict responses, inspect traces, and confirm on held-out conditions.

### Details

#### Research question
Under what controlled conditions do quantitative XAI evaluation metrics change their judgments, and can those changes be explained by the metric's measurement mechanism rather than by explanation quality itself?

#### Scope
The first study concerns feature-attribution Faithfulness under explicitly defined model targets, references, perturbations, and comparators. A score is a measurement conditional on this specification. The quality of an explanation must not be defined by the very metric being investigated.

#### Observable answers
Identify numerical score changes, pairwise ranking reversals, estimator variability or bias, and undefined outcomes. A successful diagnosis makes a quantitative or ordinal prediction, identifies the intermediate quantities responsible, survives an effect-removing control when applicable, and predicts behavior under conditions not used to develop it.

#### Claim boundary
A configuration-sensitive metric is not automatically invalid. Newness of this research contribution remains unverified; existing meta-evaluation and mechanism analyses must be compared directly.

### Evidence/Justification

**Research status:** proposed question, not an empirical conclusion.

**Project source:** the current conversation and the supplied 29 October 2026 project status define this working question.

**Prior work:** MetaQuantus evaluates evaluators; Tomsett et al. examine saliency-metric reliability; Binder et al. provide mechanism-oriented criticism of randomization checks. Their existence prevents claiming that evaluator diagnosis itself is new.

**Required future evidence:** immutable experiment artifacts, trace-based predictions, repeated estimates, and held-out tests. No new notebook results are asserted here.

### Easy explanation

We want to understand when an XAI score changes and why. Keep the model, input, and explanation fixed when the question concerns evaluator settings, then deliberately change one declared condition.

A changed number is the starting observation. We still need to determine whether the evaluator asked a different question, used a noisier estimate, or behaved differently from our prediction. The intended result states the conditions under which a particular mechanism explains the response.

**Source references:** `SRC-DESIGN`, `SRC-STATUS`, `REF-METAQUANTUS`, `REF-TOMSETT`, `REF-BINDER`.


<a id="rq-01"></a>

## RQ-01 — What counts as a changed judgment?

Track changes in one explanation's score separately from changes in two explanations' order or tie status.

### Overview

- **Who:** The researcher chooses the explanation pair; the analysis compares its scores.
- **What:** One explanation's score change and the score gap between two explanations.
- **When:** After evaluating the same candidates under each specified setting.
- **Where:** Within one metric, with the model and input(s) held fixed for the comparison.
- **Why:** Scores can change even when the metric continues to prefer the same candidate.
- **How:** Record each score, subtract B's score from A's within each setting, and compare the resulting preferences.

### Details

#### What are A, B, theta, and Q?

A and B are **two candidate explanations** of the same model output for the same input. They are not two evaluator settings. Theta labels an evaluator setting, such as a replacement baseline or subset size. Q_theta(A) is the metric's score for A under that setting, written so that a larger value is preferred.

When results are averaged across inputs, use the same predefined input set and aggregation rule for the comparison; ANA-OUTCOMES specifies how validity restrictions are handled.

#### Two different comparisons

A **score change** compares one explanation with itself across settings: did A receive another number when theta changed?

A **ranking change** concerns the relative ordering of A and B: A above B, B above A, or a tie. A **strict ranking reversal** specifically switches from A above B to B above A, or vice versa. Moving into or out of a tie is another ordering change; a comparison unresolved by uncertainty is a separate reporting outcome.

#### Pairwise margin means the score gap for one pair

“Pairwise” means “comparing two explanations at a time.” “Margin” means the signed gap between their scores **under the same setting**.

```math
\Delta_\theta(A,B)=Q_\theta(A)-Q_\theta(B).
```

- A positive margin means A receives the preferred score.
- A negative margin means B receives the preferred score.
- A zero margin means an exact numerical tie.
- The absolute size of the margin says how far apart these scores are, in this metric's score units.

#### One illustrative score table

These are invented teaching values for a higher-is-better metric, not experimental results.

| Evaluator setting | Score of A | Score of B | Pairwise margin: A minus B | Metric preference |
|---|---:|---:|---:|---|
| Setting 1 | 0.8 | 0.6 | +0.2 | A |
| Setting 2 | 0.5 | 0.4 | +0.1 | A |
| Setting 3 | 0.3 | 0.5 | -0.2 | B |

From Setting 1 to Setting 2, A's score changes, but A remains preferred: a score change without a ranking reversal. From Setting 2 to Setting 3, the margin changes from positive to negative: a pairwise ranking reversal.

#### Choosing the pair and interpreting the result

Choose A and B before confirmation because their difference answers the research question. An exact explanation versus a specified relocation is a calibration pair; two specified nonexact explanations can probe ranking changes. In the matched additive FC setting, the exact explanation already attains the largest defined correlation, so use two nonexact candidates when studying possible strict reversals.

“Metric preference” means the metric assigns the preferred number. Establishing that A is actually better under our explanation definition requires an independently justified reference relation.

For a lower-is-better raw metric M, set Q = -M before using the margin rule; retain M for reporting. Define any practical tie tolerance in advance. A small or uncertain gap may not support a clear preference, and an undefined score makes the corresponding margin undefined. Report these cases and uncertainty alongside the score gaps.

### Evidence/Justification

**Definition source:** an operational outcome definition of this project.

**Justification:** pairwise margins expose whether a numerical response changes the relative judgment. Prior saliency-metric reliability work motivates examining rankings as well as raw values.

**Evidence not yet available:** observed reversal frequencies from the new controlled experiments.

**Teaching clarification:** the score-gap definition is arithmetic. The three-setting table contains invented values solely to distinguish score changes from pairwise reversals; it is not an observation from this project.

### Easy explanation

A **pairwise margin** is simply **A's score minus B's score under the same evaluator setting**. “Pairwise” means we compare two explanations at a time.

In the table above, Setting 1 gives A = 0.8 and B = 0.6. The margin is +0.2, so the metric puts A ahead by 0.2.

Now distinguish two questions:

- **Did A's number change?** From Setting 1 to Setting 2, A falls from 0.8 to 0.5.
- **Did the preferred explanation change?** No: B falls to 0.4, so A still leads. The new margin is +0.1.

Only in Setting 3 does B lead: 0.3 - 0.5 = -0.2. That switch from a positive to a negative margin is a strict reversal.

We choose A and B before evaluating them. Their margin reports the metric's preference, not an independently established quality difference. Ties, uncertain comparisons, and undefined scores are reported explicitly.

**Source references:** `SRC-DESIGN`, `REF-TOMSETT`.


<a id="rq-02"></a>

## RQ-02 — Are the explanation and evaluator answering the same question?

Check what contribution the explanation claims and what model response the evaluator compares it with, starting with their baseline references.

### Overview

- **Who:** The researcher specifies the explanation's meaning and the evaluator's comparison.
- **What:** Whether both sides refer to quantities whose relationship we have justified.
- **When:** Before interpreting a score and whenever a relevant evaluator setting changes.
- **Where:** The output being explained, reference baseline, attribution representation, and response comparison.
- **Why:** A fixed explanation can be evaluated against a different model response after a setting change.
- **How:** Write both quantities explicitly, compare matched and deliberately mismatched conditions, and predict the resulting response.

### Details

#### Start with “contribution compared with what?”

In our baseline-relative experiments, an explanation describes a feature's contribution **relative to a specified comparison input**. That input is the **explanation reference**, also called the reference baseline b.

The baseline is a chosen comparison point. It is not automatically a perfect explanation, a typical input, an absence of information, or a real-world causal removal.

#### Two similar phrases with different meanings

| Phrase | Meaning here |
|---|---|
| Explanation reference | The comparison input or baseline b used to define what a contribution means. |
| Reference explanation | An attribution E* derived under the declared model, output, baseline, and explanation convention. |

Thus, a reference explanation is calculated **using** an explanation reference. They are different objects.

#### One contribution example

For this teaching example only, take a one-feature model g(x) = 2x and the current input x = 3. The model output is 6. Define the feature's contribution relative to baseline b as the output difference:

```math
E^{*,b}(3)=g(3)-g(b)=6-2b.
```

| Explanation reference | Reference output | Exact contribution under that reference |
|---|---:|---:|
| b = 0 | g(0) = 0 | 6 - 0 = 6 |
| b = 1 | g(1) = 2 | 6 - 2 = 4 |

The predictor and current input stayed the same. The reference changed the question: contribution relative to output 0, or contribution relative to output 2? Both answers are correct for their specified questions.

#### What the evaluator compares

In a baseline-replacement evaluation, the evaluator replaces selected input features and measures the resulting output change. Its **replacement baseline** is a separate choice from the explanation reference.

Keep the supplied explanation fixed at E = 6, which was derived relative to b = 0:

- With evaluator baseline 0, the measured response is 6. The evaluator compares 6 with 6.
- With evaluator baseline 1, the measured response is 4. The evaluator compares the same explanation value 6 with 4.

The second comparison uses different references. E = 6 has not ceased to be exact for its original reference; the evaluator now measures another response.

There are two distinct ways to restore this example's agreement: keep E = 6 and restore evaluator baseline 0, or keep evaluator baseline 1 and recompute the explanation as E = 4. The first changes the evaluator; the second changes the explanation.

**Scope of the example:** this illustrates contribution meanings and numerical agreement, not a correlation score. A single feature cannot supply a defined coordinate Pearson correlation. In a larger evaluation, a numerical mismatch can still coexist with correlation one.

#### What a compatibility check asks

Write down the two sides before interpreting their comparison:

1. **Explanation side:** What output, reference, units, signs, and allocation rule does E represent?
2. **Evaluator side:** What output change or other quantity does the evaluator actually compute?
3. **Expected relationship:** Why should these quantities agree numerically, correlate, or have the same order under this metric?

A shared baseline is one part of this check. Matching baselines alone does not guarantee that every attribution convention reproduces every deletion response. EXP-03 investigates this issue under feature interactions. Also distinguish a signed contribution from a magnitude-only relevance value, and one scalar output target from another.

#### Research use

EXP-02 crosses explanation references with evaluator baselines. Holding an explanation fixed while changing only the evaluator identifies the changed comparison; recomputing a matched explanation supplies a separate control. Derive the expected metric response in each condition. A mismatch alone does not predict that correlation must decline, and a high score establishes only the declared relation.

### Evidence/Justification

**Research status:** proposed compatibility question.

**Conceptual grounding:** Sundararajan and Najmi distinguish different model-explanation operationalizations of Shapley values. Bhatt et al. define a correlation between attribution sums and output changes relative to a reference baseline.

**Derivation plan:** state the exact quantities on both sides before deriving expectations. Do not infer a direction of correlation change solely from a reference mismatch.

**Worked example:** g(x) = 2x is an illustrative mathematical construction. Its contributions follow the declared definition E*(x; b) = g(x) - g(b). The example demonstrates reference semantics and numerical comparison; it reports no measured metric result.

**Formal reference:** M. Sundararajan and A. Najmi, “The many Shapley values for model explanation,” in *Proc. 37th Int. Conf. Machine Learning*, PMLR, vol. 119, 2020, pp. 9269–9278. [Primary paper](https://proceedings.mlr.press/v119/sundararajan20b.html). The paper supplies context for alternative explanation definitions; the teaching example is the project's own calculation.

### Easy explanation

##### What is an explanation reference?

It is the **comparison input used when saying how much a feature contributes**. Ask: “Contributes compared with what?”

Use the example above: g(x) = 2x and x = 3, so the model output is 6.

- Compared with input 0, whose output is 0, the contribution is **6**.
- Compared with input 1, whose output is 2, the contribution is **4**.

The current model output is still 6. The two contribution values answer questions with different starting points.

##### Why does this matter during evaluation?

Suppose we supply the explanation value **6**, meaning “contribution relative to input 0.” If the evaluator replaces the input with **1**, it measures an output change of **4**. It is comparing values derived from different reference points.

The explanation has stayed the same. The evaluator has changed the comparison it makes. We must understand that change before interpreting the resulting metric score.

##### Two phrases to keep separate

The **explanation reference** is the baseline input, such as 0 or 1. The **reference explanation** is the attribution derived using that input, such as 6 or 4.

RQ-02 asks us to make the meaning of both sides explicit and predict what their comparison should show. This small example explains those meanings; it does not calculate FC or imply that a baseline mismatch always lowers correlation.

**Source references:** `SRC-DESIGN`, `REF-SHAPLEY`, `REF-BHATT`.


<a id="rq-03"></a>

## RQ-03 — Can a mechanism prediction survive new conditions?

Test whether our proposed reason for a metric's behavior predicts what happens under reserved conditions.

### Overview

- **Who:** The researcher proposes and tests a reason for the metric's behavior.
- **What:** Predictions about the metric's behavior.
- **When:** After exploration but before confirmation data are inspected.
- **Where:** Held-out inputs, settings, model coefficients, and applicable model families.
- **Why:** An after-the-fact explanation can fit an observation without predicting anything new.
- **How:** Freeze assumptions and predictions, then evaluate them on new conditions and fresh repetitions.

### Details

#### What does “account” mean here?

An **account** is a proposed explanation of why the metric behaves as it does. A **mechanism account** identifies the calculation steps thought to produce the behavior, the assumptions under which that reasoning applies, and a prediction we can check. We also call it a **mechanism hypothesis**.

This is an explanation **of the metric's behavior**. The XAI explanation E is the attribution vector being evaluated. These are separate uses of the word “explanation.”

#### Question

Can this mechanism hypothesis predict the metric's response under conditions that were not used to develop or revise it?

For example, under HYP-01's additive-model and uniform-subset assumptions, the proposed reasoning predicts that population FC is unchanged across admissible subset sizes. A reserved subset size can test that prediction. Finite random estimates need separate uncertainty analysis; the population claim does not say that every sampled score will be identical.

#### Confirmation logic

Predictions may concern a population score, an estimator's error or variability, an ordering of two explanations, a trace residual, or a failure boundary. Define what would support or contradict the particular prediction. Use a control that removes the suspected cause when that can be done while preserving other relevant conditions.

Fresh random seeds at the same condition provide evidence about repeatability. Conditions withheld from hypothesis development test the prediction's scope. Pairing within a comparison is compatible with fresh randomness across repetitions.

#### Allowed conclusion

State the tested conditions, supported mechanism, observed response, and remaining discrepancy. Evidence from a synthetic family supports the stated scope; it does not establish universal metric validity.

### Evidence/Justification

**Research status:** proposed confirmation criterion.

**Justification:** this is a design commitment for falsifiable mechanism testing. Morris et al. provide methodological guidance for explicitly defining simulation aims, data generation, estimands, methods, and performance measures. MetaQuantus and Binder et al. establish relevant prior evaluator and mechanism analyses.

**Future artifact:** a prediction register fixed before the confirmation run, linked to actual outcomes and failures.

**Terminology clarification:** “account” is ordinary research language for a proposed explanation, not a new metric or experimental output. The subset-size illustration refers to the project's conditional HYP-01 derivation and remains a prospective confirmation example.

### Easy explanation

Here, **account** means **our proposed explanation of why the metric behaves that way**. You can read “mechanism account” as “a reason for the metric's behavior that makes a testable prediction.”

For example, we may propose that the additive model and uniform sampling rule explain why the population FC score stays unchanged across subset sizes. We then test that prediction at reserved sizes, keeping its assumptions explicit.

The XAI explanation E is the attribution vector being scored. Our account is the reasoning about the metric that scores it.

Using a fresh seed at an old setting checks repeatability. Trying a reserved setting checks whether the proposed reasoning predicts something beyond the conditions used to develop it.

**Source references:** `SRC-DESIGN`, `REF-MORRIS`, `REF-METAQUANTUS`, `REF-BINDER`.


<a id="fw-00"></a>

## FW-00 — Research map and reading guide

Start here: the research question guides our choice of measurement rules; the remaining stages prepare, run, and interpret controlled comparisons.

### Overview

- **Who:** The researcher and collaborators.
- **What:** A navigation and governance map for this research atlas.
- **When:** Throughout design, execution, and interpretation.
- **Where:** This atlas and future versioned research artifacts.
- **Why:** Keep conceptual understanding, formal claims, and evidence distinct.
- **How:** Read Summary and Overview first, Details for the protocol, Evidence for support, and Easy explanation for interpretation.

### Details

#### What the first arrow means

**RQ -> Measurement contract** means: “Given what we want to learn, the researcher chooses and documents how it will be measured.”

The research question gives the purpose. It does not automatically produce a unique contract or set numerical parameters. We choose and justify the explanation meaning, metric, output target, baseline, quantities held fixed, and settings varied so that the comparison can answer the question.

For a question about subset size, for example, the contract states that subset size is varied while the model, input, explanation, and other relevant settings remain fixed. That is an experimental design decision motivated by the question.

#### Reading path

The arrows below show the order of design and investigation.

```text
Research question: What do we want to learn?
       |
       | Researcher chooses and documents measurement rules
       v
Measurement contract: What is measured, fixed, and varied?
       |
       v
Oracle and reference bank: Prepare controlled test cases
       |
       v
Metric evaluation: Calculate scores and keep traces
       |
       v
Diagnosis: Compare results with mechanism predictions
       |
       v
Held-out confirmation: Test predictions on reserved conditions
```

During exploration, an unsuitable contract may lead us to revise the design or narrow the question. Freeze the applicable contract and predictions before confirmation.

#### First scope

Begin with synthetic tabular feature-attribution Faithfulness. EXP-01 studies subset sampling and estimation; EXP-02 studies reference compatibility; EXP-03 studies interactions. EXP-04 and EXP-05 provide representation and placement controls. The framework permits later Compactness and Robustness studies after metric-specific definitions and reference assumptions are established.

#### What a box status means

**In-Progress:** this item's research specification or review is still being refined. It does not mean an experiment is running.
**Finalized:** this item's text or protocol has been explicitly frozen for its stated purpose. It does not imply that a hypothesis is true, a method is universally valid, or an experiment passed.
**Deprecated:** the item is superseded and retained for provenance.

Scientific evidence is stated separately in Evidence/Justification using proposed, derived, observed, implemented, and unverified distinctions. Initial planning items remain In-Progress. Finalization never propagates automatically through links.

#### Navigation

Connections express a named direction. A reference citation supplies context or a definition unless the note explicitly identifies a demonstrated result. Warning rules identify missing documentation, missing traces, and inconsistent finalization states; they do not validate scientific truth.

### Evidence/Justification

**Source:** this organization implements the user's explicit request to map the current research framework and its detailed experiments in Project Atlas.

**Status of content:** the new experiments are proposed. The previous attribution-relocation pilot is historical and must not be treated as execution of this design.

**Scope of authority:** this atlas does not modify the experiment repository or assert its current implementation. Future code claims require a fresh repository inspection and exact artifact links.

**Workflow clarification:** the RQ-to-contract arrow expresses a researcher-made design step. It is not a mathematical implication or a claim that one research question determines one unique measurement contract.

### Easy explanation

The arrow **RQ -> Measurement contract** means:

**“We have a research question. Now we choose and write down measurement rules that let us investigate it.”**

The RQ tells us what we want to learn. The researcher makes the design choices. The question does not automatically select a metric, baseline, or sample budget, and more than one justified contract may address it.

If we ask what happens when subset size changes, the contract states which subset sizes we compare and what stays fixed. We then prepare test cases, calculate scores and traces, interpret the results, and check predictions on reserved conditions.

The flow is the order of that work. It is not a statement that each box mathematically determines the next.

**Source references:** `SRC-DESIGN`, `SRC-STATUS`, `SRC-PROJECT-INSTRUCTIONS`.


<a id="fw-01"></a>

## FW-01 — Stage 1 — Measurement contract and predictions

Write down what the explanation means, what the evaluator calculates, which settings we change, and which random choices actually occur.

### Overview

- **Who:** The researcher chooses and documents the rules.
- **What:** The explanation meaning, evaluator settings, compatibility checks, and testable predictions.
- **When:** During design, before interpreting results, and before freezing confirmation predictions.
- **Where:** The connection between the supplied explanation and the evaluator's calculation.
- **Why:** A changed score can reflect a changed comparison or different random draws.
- **How:** Separate C, theta, and xi; state the quantities on both sides of the comparison; record assumptions and expected responses.

### Details

#### What the Measurement Contract records

The Measurement Contract is the written specification linking the research question to a concrete comparison. It records the explanation's intended meaning, the evaluator's operational rules, and the assumptions needed to interpret the result. It is a project organizing document.

| Symbol | Meaning | Example in a subset-based evaluation |
|---|---|---|
| C | What the explanation is intended to represent and which property is examined. | Signed contributions to one fixed scalar output relative to baseline zero. |
| theta | The evaluator settings: how the calculation is configured. | Uniformly sample subsets of size k; use B draws; replace with b; compare using Pearson. |
| xi | The realized random choices in one evaluation. | The particular subsets drawn, in their recorded order. |

#### Score notation

```math
\widehat M_m(E;g,x,C,\theta,\xi)
```

Here m identifies the metric; E is the supplied explanation; g is the scalar model output being explained; and x is the input. C, theta, and xi have the meanings above. This is conceptual notation, not a library function signature.

#### Settings versus realized draws

For a fixed-size subset evaluator, one useful representation is:

```math
\theta:\quad S_r\sim\operatorname{Uniform}\{S\subseteq\{1,\ldots,d\}:|S|=k\},
\quad r=1,\ldots,B,
\qquad
\xi=(S_1,\ldots,S_B).
```

In the initial random-sampling protocol, the subset draws are independent across r. Theta specifies the sampling rule and budget; xi records the particular outcome of applying that rule. The same theta can produce another xi on a repeated run. The explanation E does not choose these subsets.

With compatible equal-size sampling rules, paired evaluations can reuse the same xi. When subset sizes differ, couple the underlying random feature orderings and use the appropriate prefixes, as specified in CTRL-PAIRING. A seed initializes a generator; the recorded draws show what was actually used. For an evaluator with no internal randomness, xi has no active role.

#### Compatibility check: are the intended comparison quantities aligned?

Before interpreting the score, write what E claims and what the evaluator computes. Check:

| Check | Concrete question |
|---|---|
| Output target | Are both sides about the same scalar output and, where relevant, the same fixed class? |
| Reference baseline | Was E defined relative to the baseline used to measure the replacement response, or are we deliberately testing a mismatch? |
| Representation | Are signed contributions, magnitudes, and units being used consistently with the intended property? |
| Comparison meaning | Are we testing numerical agreement, linear association, or ordering? Does the explanation convention justify that relationship? |

In RQ-02's example, E = 6 describes an output difference relative to input baseline 0. Replacing the input by baseline 1 produces a response of 4. The check identifies that these numbers refer to different starting points. The arithmetic can be correct on both sides.

A deliberate mismatch is allowed as an experimental condition. Label it and derive its expected effect; do not silently interpret it as matched-reference correctness. Matching baselines alone is insufficient when the allocation rule and deletion response represent different quantities, as EXP-03 examines.

#### Required records and prediction freeze

Record target selection, explanation reference, actual replacement values, signs and normalization, sampling law, subset size, draw budget, comparator, score direction, aggregation, and undefined-result handling.

Before confirmation, freeze the applicable contract, prediction, uncertainty method or tolerance, reserved conditions, and outcomes that would challenge the mechanism hypothesis.

### Evidence/Justification

**Research status:** proposed specification schema.

**Justification:** Bhatt et al.'s Faithfulness definition explicitly compares quantities relative to a baseline; Sundararajan and Najmi show why explanation operationalizations matter. These motivate explicit compatibility checks.

**Validation check:** every experiment must point to a contract and state what is fixed, manipulated, predicted, and not yet established. No result should be labeled a metric failure solely because its score changed.

**Notation clarification:** xi is defined here as the evaluator's realized random choices; theta defines their generating rule. The contribution example is the worked calculation in RQ-02, and the coupled sampling construction is specified in CTRL-PAIRING. These clarify the proposed design and do not report executed results.

### Easy explanation

##### C, theta, and xi do different jobs

**C** says what the explanation is intended to mean. **Theta** says how we configure the evaluator. **Xi**, written **ξ**, records the particular random choices made in one evaluation.

##### Xi is the actual list of draws

For a small illustration, theta says: “There are four features. Randomly select two features per draw, and make three independent draws.”

The settings stay the same, but two runs can produce:

| Draw | Run 1 selects | Run 2 selects |
|---|---|---|
| 1 | {1, 2} | {2, 4} |
| 2 | {1, 4} | {1, 3} |
| 3 | {2, 3} | {1, 2} |

**Xi for Run 1 is the complete list in the “Run 1 selects” column. Xi for Run 2 is the list in “Run 2 selects.”** Theta is the sampling rule; xi is what was actually drawn using it.

The explanation E did not choose these features. The evaluator did. Different xi values can lead to different estimated scores even with the same E and theta, although a score need not change on every run.

For paired evaluations using compatible, equal-size sampling rules, we can deliberately reuse the same list. This is decided before calculating the scores.

##### What is a compatibility check?

Before interpreting the score, ask: **“Do the numbers being compared mean what we intend them to mean?”**

In RQ-02, explanation value 6 refers to baseline 0, while evaluator response 4 refers to baseline 1. Both calculations are correct for their own starting points. The compatibility check identifies the mismatch.

Check the output being explained, reference baseline, signs and units, and whether the metric compares exact values or only association/order. If a mismatch is intentional, record it as part of the experiment and predict its effect. A high score by itself does not perform this check for us.

**Source references:** `SRC-DESIGN`, `REF-BHATT`, `REF-SHAPLEY`.


<a id="fw-02"></a>

## FW-02 — Stage 2 — Oracle worlds and reference bank

Construct known model functions and candidate explanations whose intended differences can be specified independently of the metric.

### Overview

- **Who:** The researcher constructs models and reference attributions.
- **What:** Controlled input-model-explanation cases.
- **When:** After the contract is defined and before scoring.
- **Where:** Synthetic additive and interaction model families.
- **Why:** Known references separate explanation modifications from evaluator behavior.
- **How:** Generate inputs, compute exact references under named conventions, and create controlled candidate variants.

### Details

#### Oracle worlds
An oracle is a model whose relevant behavior is known sufficiently precisely to derive reference quantities. It does not denote real-world causal truth.

Start with a sparse additive scalar model and independently generated inputs. Use additive nonlinear and interaction extensions to test the boundary of additivity rather than treating all nonlinear functions as one class.

#### Explanation bank
Include exact baseline-relative attributions and deliberately modified candidates. Relocation changes where contribution is assigned while preserving specified vector properties. Sign and positive-scale transformations probe distinct representation sensitivities.

#### Pair selection
Predefine pairs that answer a scientific contrast, such as exact versus relocated attribution or two relocation levels. The evaluator scores each candidate-condition combination; the analysis subsequently compares the pair.

#### Reference discipline
An exact reference is exact for the declared model, output, baseline, and attribution convention. When a model or convention changes, derive the corresponding reference again. Record reference error or another justified diagnostic independently of the tested score.

### Evidence/Justification

**Research status:** proposed fixture and explanation bank.

**Grounding:** analytical quantities come from the chosen model formulas. The controlled-reference strategy has precedent in fidelity-metric evaluation. Attribution conventions and their differences are discussed by Sundararajan and Najmi.

**Evidence boundary:** having a known reference does not guarantee that every metric attains its maximum on that reference. This is a question for the specified measurement relation.

### Easy explanation

We begin with models whose behavior we can calculate. This lets us prepare a reference explanation and controlled alternatives without asking the tested metric to decide which explanation is correct.

The reference is exact for a stated definition. We then make specified changes, such as moving contributions to unused features. We choose explanation pairs because their difference answers a research question; the evaluator does not choose those pairs for us.

**Source references:** `SRC-DESIGN`, `REF-SHAPLEY`, `REF-FIDELITY`.


<a id="fw-03"></a>

## FW-03 — Stage 3 — Metric evaluator with computation traces

Compute scores while preserving the intermediate selections and quantities that produced them.

### Overview

- **Who:** The metric evaluator and trace recorder.
- **What:** Scores plus intermediate numerical records.
- **When:** During each evaluation, before diagnosis.
- **Where:** The controlled model and attribution bank.
- **Why:** A final score alone cannot identify the mechanism responsible for a change.
- **How:** Apply the declared intervention, share compatible draws, record quantities and failures, then calculate the score.

### Details

#### Inputs and outputs
Inputs: fixed input x, scalar target g, candidate E, contract C, evaluator settings theta, and sampled subsets. Outputs: per-input scores; raw trace records; exceptions and nonfinite outcomes; and a link to the exact configuration.

Explanation intervention changes E outside the evaluator. Internal input perturbation changes selected coordinates of x to measure model response. These operations must be recorded separately.

#### FC trace quantities
For a selected subset S and replacement baseline b:
```math
U_S=\sum_{j\in S}E_j,\qquad V_S=g(x)-g(x_{S\leftarrow b}).
```
Store S, b, the original and perturbed outputs, U_S, V_S, and any no-change or zero-variance condition. Correlation is computed across subsets within an input; input-level results are aggregated afterward.

#### Randomness control
A shared draw plan is an input to computation. It is not post-evaluation extraction. Reuse compatible subsets across explanations and settings, while preserving each condition's sampling distribution. Keep new independent plans for replications.

#### Implementation boundary
Adapters must preserve the metric's definition. Instrumentation must not change a score formula to force agreement with a hypothesis.

### Evidence/Justification

**Definition source:** U and V follow the Faithfulness comparison in Bhatt et al., Sec. 3.2.

**Research status:** proposed architecture; no implementation of this new trace pipeline is asserted here.

**Required validation:** a deterministic toy case should reproduce hand-computed U and V; a degenerate case must be recorded as undefined rather than silently dropped; versioned adapter checks must confirm that instrumentation preserves the intended computation.

### Easy explanation

The evaluator returns both a score and a record of how it obtained that score. For each selected feature subset, it records the attribution sum and the corresponding change in the model output.

The evaluator selects features according to its sampling rule; the explanation supplies attribution values. Paired draws are prepared before these calculations. Keeping the intermediate records lets us locate the source of a changed result.

**Source references:** `SRC-DESIGN`, `REF-BHATT`.


<a id="fw-04"></a>

## FW-04 — Stage 4 — Diagnosis and held-out confirmation

Use traces, predictions, and repeated estimates to explain responses and check the explanation on new conditions.

### Overview

- **Who:** The researcher and analysis pipeline.
- **What:** A bounded explanation of metric behavior.
- **When:** After controlled evaluations and again in the confirmation phase.
- **Where:** Scores, pairwise margins, trace records, and held-out conditions.
- **Why:** Observed disagreement does not identify its cause.
- **How:** Compare observed and predicted responses, run mechanism controls, quantify uncertainty, and retain unresolved discrepancies.

### Details

#### Diagnosis inputs
Use raw scores, predefined pairwise comparisons, estimation error against an analytical reference when available, trace residuals, and failure rates. A prediction residual is observed metric behavior minus the behavior predicted by the mechanism account, not model prediction error against a label.

#### Interpretation
First account for the known explanation intervention. Then inspect expected changes in the evaluated quantity, finite-sampling effects, and remaining discrepancies. These causes may coexist; insufficient evidence is a valid outcome.

#### Strength of a mechanism account
Require a prediction stated before confirmation, consistency with the numerical trace, a control that removes the suspected cause where possible, and tests on new conditions. A post-hoc narrative alone is insufficient.

#### Confirmation
Use fresh sampling plans across repetitions and conditions withheld from hypothesis development. Pair draws within each relevant comparison. Report where predictions succeed, where they fail, and which assumptions delimit the conclusion.

#### Deliverable
A conditional behavior account connecting conditions, mechanism, response, uncertainty, and evidence. This is not a universal ranking of metrics.

### Evidence/Justification

**Research status:** proposed analysis architecture.

**Prior work:** MetaQuantus establishes evaluator meta-evaluation; Binder et al. provide concrete mechanism analyses of sanity checks. The present project must identify a narrower contribution rather than claiming to introduce mechanism reasoning itself.

**Future evidence:** an immutable prediction register, per-condition traces, repeated estimates, and held-out outcomes linked to actual runs.

### Easy explanation

Compare the result with the prediction, then inspect the calculations that produced it. Repeated evaluations reveal sampling variability; a control can check whether removing the suspected cause removes the effect.

Finally, test the account on reserved conditions. The useful conclusion names the condition, the measurement mechanism, and the response it explains. If a discrepancy remains, record it as unresolved rather than assigning a scientific verdict without evidence.

**Source references:** `SRC-DESIGN`, `REF-METAQUANTUS`, `REF-BINDER`, `REF-MORRIS`.


<a id="set-01"></a>

## SET-01 — Sparse additive oracle and scalar target

Use a 20-feature controlled model with five used features and fifteen unused features, explaining one fixed scalar output.

### Overview

- **Who:** The experiment fixture generator.
- **What:** Synthetic inputs and a known scalar function.
- **When:** For calibration and the first two core experiments.
- **Where:** A continuous tabular feature space.
- **Why:** Known feature effects permit analytical predictions and controlled relocation.
- **How:** Draw independent standard-normal inputs and evaluate a sparse linear target.

### Details

#### Proposed fixture
```math
x\sim\mathcal N(0,I_{20}),\qquad g(x)=w^\top x+c,\qquad c=0,
w=(2.5,-2,1.5,1,-0.5,0,\ldots,0).
```
Features 1-5 are used; features 6-20 are unused by g. Explain the same scalar g throughout each comparison. A classifier wrapper may be added later, but switching between logit and probability targets is a separately declared intervention.

#### Rationale
The used/unused distinction gives a clear placement contrast. Signed positive and negative coefficients permit sign-sensitive controls. The simple formula supports exact subset deletion differences, while twenty features allow multiple subset sizes.

#### Assumptions and limits
Independent normal inputs are a deliberate fixture choice, not a claim about real data. The fixed-size subset correlation identity relies on additivity and the sampling rule, not specifically on Gaussian inputs. Real-data validity and causal meaning are outside what this fixture alone establishes.

#### Extension
Compare an additive nonlinear function with an interaction model to isolate the role of additivity. Recompute the exact reference under every declared model and convention.

### Evidence/Justification

**Source:** proposed fixture from the research-design conversation; the sparse weights also appear in the historical project context.

**Analytical support:** linearity gives exact baseline-relative feature effects and subset output differences.

**Evidence status:** the new protocol has not been executed. Earlier pilot outputs do not validate the current settings or trace pipeline.

### Easy explanation

Use a deliberately simple model with twenty features, of which five affect its scalar output. Because we know the formula, we can calculate each feature's baseline-relative contribution and identify unused coordinates.

This gives us a clear place to start. It supports exact comparisons and controlled relocation, while the later model extensions test where those calculations stop applying. It does not by itself establish what happens in real datasets.

**Source references:** `SRC-DESIGN`, `SRC-STATUS`.


<a id="set-02"></a>

## SET-02 — Exact reference definitions and explanation variants

Specify baseline-relative contributions and a named interaction allocation rule before preparing candidate explanations.

### Overview

- **Who:** The researcher defines the reference; the fixture computes it.
- **What:** Exact attributions and controlled variants.
- **When:** Before any metric score is interpreted.
- **Where:** Each declared input-model-baseline combination.
- **Why:** Reference correctness must be established independently of the tested metric.
- **How:** Use analytical additive contributions, or explicit Baseline Shapley for the interaction game.

### Details

#### Additive reference
For the linear fixture:
```math
E_j^{*,b}(x)=w_j(x_j-b_j).
```
For an additive nonlinear function g(x)=sum_j h_j(x_j)+c, use E_j=h_j(x_j)-h_j(b_j). The coefficients w alone are a different representation from baseline-relative contributions.

#### Independent reference diagnostic
```math
e_{\rm ref}(E)=\frac{\|E-E^{*,b}\|_1}{\|E^{*,b}\|_1}.
```
Record a zero denominator explicitly. This error is meaningful only relative to this selected attribution convention; it is not universal explanation quality.

#### Candidate variants
Prepare relocation, sign reversal, positive scaling, and combinations required by a predefined contrast. State which vector properties are preserved and which are intentionally changed. A positive rescaling of E with a fixed target changes calibration; jointly rescaling target and explanation is a different experiment.

#### Interaction reference
Define the baseline coalition game explicitly and compute Baseline Shapley values. Exactness under this allocation rule need not make an attribution sum identical to the effect of deleting the same subset.

#### Pairs for ranking endpoints
Retain the exact reference as a calibration anchor. When studying sampling-induced rank reversals, specify a pair of two nonexact candidates before confirmation, because the exact additive reference already attains the largest possible defined Pearson score. Document each candidate and the development-stage population margin; do not select pairs after seeing confirmation reversals.

### Evidence/Justification

**Definition source:** additive reference values are derived directly from the chosen function and baseline. Baseline Shapley is defined in Sundararajan and Najmi (2020).

**Research status:** proposed reference-generation specification. No external algorithm's output is treated as ground truth merely because it scores well.

**Validation:** reference values must satisfy the declared model identity or allocation rule before a metric response is assessed.

### Easy explanation

A reference explanation says how contributions are assigned under a particular definition. For a linear model, the contribution depends on the coefficient and the difference from the baseline; the coefficient alone is a different quantity.

We retain that reference and create clearly described variants. The exact reference is a calibration anchor. To study sampling-driven ranking reversals, we also need two nonexact candidates, because a defined perfect correlation cannot be strictly exceeded.

**Source references:** `SRC-DESIGN`, `REF-SHAPLEY`.


<a id="set-03"></a>

## SET-03 — Evaluator settings and compatibility

Make baseline, target, sign handling, subset size, sampling budget, comparator, and aggregation explicit.

### Overview

- **Who:** The researcher sets conditions; adapters enforce them.
- **What:** The evaluator configuration theta.
- **When:** Before a run and whenever a configuration changes.
- **Where:** All metric calls and stored traces.
- **Why:** Defaults can change the evaluated quantity or hide incomparable outputs.
- **How:** Freeze a baseline protocol and declare one-factor or planned factorial changes.

### Details

#### Baseline protocol
Use a literal numeric zero baseline for the initial additive contribution test. Keep the scalar target fixed, preserve signed attributions for FC and FE, and avoid unannounced absolute-value or normalization preprocessing. Configuration choices for MC are documented separately because its magnitude-oriented comparison serves a different control.

#### Conditions
Subset size k is the number of selected coordinates per draw. Budget B is the number of subset draws used for one input's estimated score. Draw each subset without replacement internally; repeated draws may select the same subset again. Baseline b is fixed for the input and must not depend on the selected subset in tests using the fixed-baseline identity.

#### Compatibility check
Separate the explanation reference from the evaluator's replacement value. A theta change can preserve the attribution vector while changing the comparison's meaning. Preserve each condition's intended sampling law when pairing.

#### Implementation verification
Before execution, inspect the pinned library's actual perturbation and similarity functions, signatures, target handling, aggregation behavior, and nonfinite policy. A named baseline or similarity function must not be assumed to implement its everyday meaning.

### Evidence/Justification

**Research status:** proposed configuration policy; this node does not claim the repository currently implements it.

**Sources:** the project's Quantus audit priorities and the measurement definitions of Bhatt et al.

**Required evidence:** package version, exact parameters, inspected source revision, deterministic adapter checks, and the resolved configuration attached to every run.

### Easy explanation

Write the evaluator's settings explicitly. How many features are changed in one draw? How many draws form one score? What values replace those features? Are signs preserved? Which output and comparison function are used?

These choices tell us what calculation produced the number. A familiar setting name is insufficient: its actual behavior must be checked in the chosen implementation.

**Source references:** `SRC-DESIGN`, `SRC-PROJECT-INSTRUCTIONS`, `REF-BHATT`.


<a id="set-04"></a>

## SET-04 — Development and confirmation conditions

Separate conditions used to formulate predictions from conditions reserved for confirmation.

### Overview

- **Who:** The researcher declares the condition registry.
- **What:** Development grids, held-out settings, and repetition budgets.
- **When:** Before inspecting confirmation outcomes.
- **Where:** The synthetic experiments and their saved run configurations.
- **Why:** Repeating old cases and predicting new settings provide different evidence.
- **How:** Allocate development cases, freeze hypotheses, then evaluate independent inputs and reserved conditions.

### Details

#### Proposed development grid
For EXP-01, candidate subset sizes are k in {1,5,10,19} and budgets B in {25,100,400}, with d=20. Candidate held-out subset sizes are {2,7,13}. Repeated draw plans and multiple inputs are required to separate per-input behavior from estimator randomness.

#### Planning quantities
Earlier planning amounts were 128 development inputs, a fixed panel of 64 inputs with 20 evaluator repetitions, and 512 confirmation inputs. These are planning suggestions, not a power calculation or guaranteed sufficient precision. Revise them using the estimand and desired uncertainty before freezing the protocol.

#### Baseline experiment
Use b_0=0 and a declared nonzero pattern, such as b_1=(1,-1,1,-1,1,0,...,0), crossing explanation references with evaluator baselines. Reserve a different pattern for confirmation.

#### Interaction experiment
Reserve interaction strengths, interacting feature pairs, input draws, and applicable subset sizes. Specify the predicted trace or score relationship before evaluation.

#### Freshness
New seeds test replication at a condition. New settings test the scope of the mechanism prediction. Within each repetition, comparable evaluations may still share a paired draw plan.

### Evidence/Justification

**Research status:** proposed condition registry and provisional computational budgets.

**Justification:** explicit separation of development and confirmation supports falsifiable predictions; simulation planning should specify aims and precision rather than treating arbitrary sample counts as evidence of power.

**Required future update:** record the frozen grid, excluded development conditions, stopping or precision criteria, and exact run artifact links before results are labeled confirmatory.

### Easy explanation

Separate the conditions used to develop the explanation of metric behavior from those used to check it. The reserved conditions should test a prediction that has already been written down.

The proposed input counts and repetition budgets are planning values. We still need to choose them according to the precision required. New random seeds and new settings provide different checks, even when both are part of confirmation.

**Source references:** `SRC-DESIGN`, `REF-MORRIS`.


<a id="met-fc"></a>

## MET-FC — FaithfulnessCorrelation: subset attribution–response association

Primary metric: Pearson association between attribution sums and score changes under a declared subset-replacement contract.

### Overview

- **Who:** The evaluator receives a fixed predictor, target, input, and explanation.
- **What:** Correlation between subset attribution sums and replacement effects.
- **When:** A subset distribution, baseline, and target score are specified.
- **Where:** Within one input before aggregation across inputs.
- **Why:** Expose dependence on perturbation and comparison mechanisms.
- **How:** Pair independent subset replacements with Pearson correlation.

### Details

Fix scalar target `g`, input `x`, explanation `a`, baseline `b`, and uniform subsets of size `k`. Replace selected coordinates in the original input. The population quantity and its finite-sample estimator are distinct:

```math
U_S=\sum_{j\in S}a_j,\qquad
V_S=g(x)-g(x_{S\leftarrow b}),\qquad |S|=k.
```

```math
FC_{\mathrm{pop}}=\operatorname{Corr}_{S}(U_S,V_S),\qquad
\widehat{FC}_{B}=\operatorname{Corr}_{r=1}^{B}(U_{S_r},V_{S_r}).
```

The pinned Quantus v0.6.0 routine, freshly inspected at the commit cited below, samples coordinates without replacement within each subset and resamples subsets across runs. With the default replacement function, each perturbed input starts from the original input. `perturb_func` determines the intervention; `similarity_func` determines the comparison. Changing either can change the evaluated construct or create a metric variant.

The proposed primary contract retains signed attributions, disables attribution normalization, fixes a scalar output, and records per-input scores. Pearson requires nonzero variation in both paired vectors; undefined outcomes remain undefined. A positive affine relation can score one despite numerical disagreement. Thus this score measures association under the declared intervention, not universal explanation quality.

### Evidence/Justification

**Source definition:** U. Bhatt, A. Weller, and J. M. F. Moura, “Evaluating and aggregating feature-based model explanations,” in *Proc. 29th Int. Joint Conf. Artificial Intelligence (IJCAI)*, 2020, pp. 3016–3022, doi: [10.24963/ijcai.2020/417](https://doi.org/10.24963/ijcai.2020/417).

**Fresh versioned source inspection:** Freshly inspected through GitHub on November 4, 2026: Quantus tag `v0.6.0`, commit `85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a`. [FaithfulnessCorrelation](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/faithfulness/faithfulness_correlation.py) and the [replacement function](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/functions/perturb_func.py) support the sampling and replacement statements; [similarity functions](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/functions/similarity_func.py) specify the comparator. This verifies source behavior of the pinned release; it is not experiment execution, an audit of the project repository, or a claim about the latest release. The controlled configuration remains proposed.

### Easy explanation

For each sampled subset, add the supplied attribution values and measure how much replacing those features changes the model output. FC asks whether these two quantities move together across subsets.

It measures association. A perfect correlation can survive a scale difference, so it need not mean that the two quantities are numerically equal. The sample budget, subset rule, and replacement baseline therefore matter to the interpretation.

**Source references:** `REF-BHATT`, `SRC-DESIGN`, `SRC-QUANTUS-060-FC`, `SRC-QUANTUS-060-PERTURB`, `SRC-QUANTUS-060-SIMILARITY`, `SRC-QUANTUS-060-BASE`, `SRC-QUANTUS-060-TAG`.


<a id="met-fe"></a>

## MET-FE — FaithfulnessEstimate: deterministic singleton comparison

Related check: correlate each attribution with the corresponding independent single-feature replacement effect.

### Overview

- **Who:** The evaluator compares feature attributions and feature effects.
- **What:** Singleton attribution–response Pearson correlation.
- **When:** One feature is replaced per step with a deterministic baseline.
- **Where:** Across coordinates within each fixed input.
- **Why:** Check the FC mechanism without random subset sampling.
- **How:** Evaluate every singleton replacement from the original input.

### Details

In the proposed singleton configuration, `features_in_step=1`, signed attributions and a fixed scalar target give:

```math
v_j=g(x)-g(x_{j\leftarrow b_j}),\qquad
FE(a;x,b)=\operatorname{Corr}_{j=1}^{d}(a_j,v_j).
```

The pinned Quantus v0.6.0 implementation, freshly inspected at the commit cited below, sorts attributions before forming nonoverlapping groups. With the default replacement function, each group is replaced in the original input; deletion is not cumulative. Singleton sorting only reorders paired observations, leaving correlation unchanged. With deterministic inference and baseline replacement, this configuration has no subset-sampling uncertainty.

Under the additive conditions in HYP-01, singleton FE equals population FC for every admissible fixed subset size. Agreement therefore checks a shared analytical mechanism; it does not supply an independent validation of explanation quality. Larger feature groups introduce an attribution-dependent partition and should be treated as a separate configuration. Constant attributions or effects produce undefined correlation. The implementation's default comparator is Pearson, despite a conflicting Spearman mention in its documentation text.

### Evidence/Justification

**Literature provenance:** D. Alvarez-Melis and T. S. Jaakkola, “Towards robust interpretability with self-explaining neural networks,” in *Advances in Neural Information Processing Systems*, vol. 31, 2018. [Paper](https://papers.nips.cc/paper/2018/hash/3e9f0fc9b2f89e043bc6233994dfcf76-Abstract.html).

**Fresh operational verification:** Freshly inspected through GitHub on November 4, 2026: Quantus tag `v0.6.0`, commit `85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a`. [FaithfulnessEstimate](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/faithfulness/faithfulness_estimate.py) confirms noncumulative groups and the executable Pearson default, while its parameter docstring mentions Spearman. This verifies source behavior of the pinned release; it is not experiment execution, an audit of the project repository, or a claim about the latest release. The paper establishes provenance; equality with population FC is a project derivation under explicit assumptions, not an empirical result.

### Easy explanation

In the proposed configuration, examine every feature individually: compare its attribution with the output change when that feature is replaced. Correlate those paired values across features.

With deterministic replacements, there is no random subset selection to repeat. Under our additive assumptions, this calculation equals population FC. Their agreement is a useful consistency check, but it is not two independent proofs of explanation quality.

**Source references:** `REF-ALVAREZ`, `REF-BHATT`, `SRC-DESIGN`, `SRC-QUANTUS-060-FE`, `SRC-QUANTUS-060-PERTURB`, `SRC-QUANTUS-060-SIMILARITY`, `SRC-QUANTUS-060-TAG`.


<a id="met-mc"></a>

## MET-MC — MonotonicityCorrelation: magnitude–response rank association

Optional contrast metric: rank agreement between attribution magnitudes and mean squared output changes, with sign information intentionally discarded.

### Overview

- **Who:** The evaluator compares attribution magnitudes with response magnitudes.
- **What:** Spearman association with squared perturbation effects.
- **When:** Singleton perturbations and the target-output contract are fixed.
- **Where:** Across features of one input.
- **Why:** Distinguish magnitude agreement from signed contribution agreement.
- **How:** Square response differences and compare their feature ranks.

### Details

For `features_in_step=1`, `abs=True`, `normalise=False`, and `similarity_func=correlation_spearman`, use a fixed scalar target with `softmax=False`. In this configuration, the freshly inspected pinned Quantus v0.6.0 routine forms a mean squared change over `B` perturbation draws per feature and compares it with attribution magnitude:

```math
W_j=c_x\frac{1}{B}\sum_{r=1}^{B}
[g(x_{j\leftarrow b_{jr}})-g(x)]^2,\qquad
MC(a)=\rho_{\mathrm{Spearman}}(|a|,W).
```

On valid finite evaluations, the implementation's positive input-level factor `c_x` does not change within-input ranks. In the proposed deterministic matched-baseline additive setting, `W` is proportional to squared exact contributions `v`:

```math
MC(a)=\rho_{\mathrm{Spearman}}(|a|,v^2).
```

If attribution magnitudes vary, exact `v`, sign-reversed `-v`, and any positively scaled `v` all attain one. This follows from the chosen magnitude-and-rank construct. It is not sufficient evidence of metric failure. Constant magnitude vectors are undefined cases. Repeating deterministic replacements yields identical effects, so a larger perturbation count does not provide additional independent information.

Magnitude conversion is configurable preprocessing, not an unconditional operation of every MC variant. This controlled scalar-output configuration must be distinguished from the pinned `softmax=True` probability-output default and from the similarly named `Monotonicity` class.

### Evidence/Justification

**Literature provenance:** A.-P. Nguyen and M. Rodríguez Martínez, “On quantitative aspects of model interpretability,” arXiv:2007.07584, 2020. [Paper](https://arxiv.org/abs/2007.07584).

**Fresh versioned mechanism inspection:** Freshly inspected through GitHub on November 4, 2026: Quantus tag `v0.6.0`, commit `85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a`. [MonotonicityCorrelation](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/faithfulness/monotonicity_correlation.py) specifies the squared-response calculation and defaults; the [Metric base](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/base.py) applies configurable absolute-value preprocessing, and [similarity functions](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/functions/similarity_func.py) define Spearman. This verifies source behavior of the pinned release; it is not experiment execution, an audit of the project repository, or a claim about the latest release. The stated configuration is distinct from the defaults. Sign/scale predictions are project derivations; tests remain proposed.

### Easy explanation

This branch asks whether features with larger attribution magnitudes also produce larger squared output changes. It compares rankings and deliberately uses magnitudes under the stated configuration.

Consequently, reversing every attribution sign can leave this score unchanged. That behavior follows the property being measured. We must not interpret it as evidence that the explanation's signs are correct. The exact configuration, including preprocessing, is part of the metric definition.

**Source references:** `REF-NGUYEN`, `SRC-DESIGN`, `SRC-QUANTUS-060-MC`, `SRC-QUANTUS-060-BASE`, `SRC-QUANTUS-060-SIMILARITY`, `SRC-QUANTUS-060-PERTURB`, `SRC-QUANTUS-060-TAG`.


<a id="met-residual"></a>

## MET-RESIDUAL — Subset response residual: diagnostic agreement error

Diagnostic only: squared disagreement between a subset attribution sum and its baseline-replacement score change.

### Overview

- **Who:** The mechanism analyst inspects paired attribution and response values.
- **What:** Mean squared subset-response residual.
- **When:** Explanation units and the replacement contract are aligned.
- **Where:** Within one input and one specified subset distribution.
- **Why:** Detect disagreement hidden by correlation's affine invariance.
- **How:** Square and average paired numerical residuals.

### Details

With attribution sum `U` and replacement response `V` defined in MET-FC, use the auxiliary diagnostic:

```math
R(a;x,b,k)=\mathbb{E}_{S:|S|=k}
\left[\sum_{j\in S}a_j-
\{g(x)-g(x_{S\leftarrow b})\}\right]^2.
```

Its unit is squared target-output units; lower values mean closer numerical agreement under this contract. Also retain signed residuals and their mean. For exact matched-baseline additive contributions `v`, the residual is zero. Positive scaling is detected even when Pearson remains one:

```math
R(\alpha v;x,b,k)=(\alpha-1)^2\mathbb{E}_{S}[V_S^2].
```

This proposed diagnostic is related to Infidelity's perturbation–explanation mismatch principle. It is not an unchanged invocation of Quantus Infidelity. The latter's pinned v0.6.0 implementation, freshly inspected at the commit cited below, declares image applicability and uses patch perturbation machinery. Moreover, the original difference-perturbation expression pairs displacement with coefficient-like explanations; inserting already displacement-weighted contributions without conversion can double-count displacement.

This diagnostic tests the stated subset response identity, not universal explanation correctness. An exact Baseline Shapley explanation can have positive residual under interactions, and positive residual alone does not imply Pearson below one.

### Evidence/Justification

**Conceptual foundation:** C.-K. Yeh, C.-Y. Hsieh, A. S. Suggala, D. I. Inouye, and P. Ravikumar, “On the (in)fidelity and sensitivity of explanations,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019. [Paper](https://papers.neurips.cc/paper_files/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html).

**Project status:** The displayed contribution-sum diagnostic and its use here are proposed. Its analytical properties are conditional derivations; no empirical performance claim is made.

**Fresh implementation scope check:** Freshly inspected through GitHub on November 4, 2026: Quantus tag `v0.6.0`, commit `85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a`. [Infidelity](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/faithfulness/infidelity.py) supports the image/patch implementation statement. This verifies source behavior of the pinned release; it is not experiment execution, an audit of the project repository, or a claim about the latest release. This source does not implement the proposed subset-sum diagnostic above.

### Easy explanation

Ask whether the attribution sum predicts the actual output change numerically, not just whether the two quantities move together. Square their difference and average it over subsets.

This diagnostic can reveal a scale mismatch that Pearson correlation ignores. A nonzero value still needs interpretation: under interactions, an exact Shapley allocation can answer a different question from deleting a subset. This is a proposed diagnostic, not an unchanged Quantus Infidelity call.

**Source references:** `REF-YEH`, `SRC-DESIGN`, `SRC-QUANTUS-060-INFIDELITY`, `SRC-QUANTUS-060-TAG`.


<a id="exp-01"></a>

## EXP-01 — E1 — Separate subset size from finite-sampling effects

Test whether changing subset size alters a population judgment or only its finite-sample estimate under a controlled additive model.

### Overview

- **Who:** The experiment runner varies evaluator sampling settings.
- **What:** Cross subset size with perturbation sample budget.
- **When:** Predictor, input, explanation, and baseline stay fixed.
- **Where:** An additive synthetic model with enumerable feature subsets.
- **Why:** Identify sampling effects without changing explanation semantics.
- **How:** Compare exact population scores with repeated finite estimates.

### Details

Start with a deterministic additive scalar target, including a sparse linear case. Declare a baseline before subset selection and sample uniformly among all subsets of each fixed size. Keep `1 <= k < d`. Hold the explanation fixed while varying `k` and the within-input perturbation budget `B`. The number of evaluated inputs is a separate design dimension.

For small feature dimension, enumerate all subsets to obtain a reference population score. Separately specify repeated random-sampling estimates for increasing budgets, with independently recorded seeds and the same subset realizations across explanation variants when pairing is intended. Include exact contributions and a fixed corrupted explanation. Exact contributions can yield sample correlation one on every nondegenerate draw, so they alone cannot diagnose estimator noise. If ranking reversals are an endpoint, preregister two distinct nonexact explanations in addition to the exact anchor, for example separate relocations of informative features into different unused coordinates. Select and fix their population margin during development, then retain the pair for confirmation. An exact-versus-corrupted pair cannot show a strict preference above the exact score of one in this contract; it is a calibration control.

```math
g(x)=2x_1+x_2,\quad x=(3,4,5),\quad b=0,
\quad v=(6,4,0),\quad a_{\mathrm{relocated}}=(0,4,6).
```

The formal prediction is HYP-01: population FC equals coordinate correlation for every admissible `k`. Singleton FE supplies the related deterministic check. Finite Pearson estimates may exhibit budget- and subset-size-dependent bias, variance, or undefined cases. These possibilities must be measured separately instead of averaging undefined outcomes into a nominal score.

Planned outputs are population score versus subset size; estimator error versus sample budget, stratified by subset size; and the fraction of undefined estimates. Within-input replicates characterize estimation uncertainty. Across-input summaries use inputs as sampling units and retain the per-input pairing.

**Falsification criterion:** A reproducible exact-enumeration discrepancy from HYP-01, with its premises verified, would contradict the stated derivation or reveal a calculation error. A finite-sample discrepancy alone does not do so. Nonuniform subset selection, nonlinear target transformations, interaction terms, or subset-dependent baselines belong to different conditions and cannot be pooled into this test.

### Evidence/Justification

**Foundation:** U. Bhatt, A. Weller, and J. M. F. Moura, “Evaluating and aggregating feature-based model explanations,” in *Proc. 29th Int. Joint Conf. Artificial Intelligence (IJCAI)*, 2020, pp. 3016–3022, doi: [10.24963/ijcai.2020/417](https://doi.org/10.24963/ijcai.2020/417). The paper supplies the subset-correlation family, not the fixed-size invariance theorem claimed here.

**Justification and status:** The separation of population value, finite estimator, and undefined cases follows the project's HYP-01 derivation. The experiment is proposed and unexecuted. A small exact-enumeration setting is selected for identifiability; extension to high-dimensional or image settings would require separate evidence.

### Easy explanation

Change how many features are selected per draw and how many draws form one score, while keeping the explanation fixed. In the stated additive setting, the population correlation should remain unchanged across valid fixed subset sizes.

Finite estimates can still differ. Exact calculations provide the reference; repeated draws reveal estimation behavior. Two predefined nonexact explanations allow a ranking-reversal analysis, while the exact explanation remains a calibration control.

**Source references:** `REF-BHATT`, `SRC-DESIGN`.


<a id="exp-02"></a>

## EXP-02 — E2 — Cross explanation reference and evaluator baseline

Use a two-by-two reference design to identify baseline compatibility effects without assuming that mismatch must lower correlation.

### Overview

- **Who:** The experiment runner controls both reference definitions.
- **What:** Explanation baseline crossed with evaluator replacement baseline.
- **When:** Predictor, input, target, and subset distribution remain fixed.
- **Where:** A linear model with exact reference-relative contributions.
- **Why:** Separate changed measurement semantics from explanation deterioration.
- **How:** Evaluate all four matched and mismatched reference combinations.

### Details

Choose two explicit baseline vectors, `b0` and `b1`, and calculate an exact explanation under each. Rows designate the explanation reference; columns designate the evaluator's replacement baseline. Every cell uses the same input, scalar target, subset size, and subset law. Use exact enumeration first and paired finite samples as a secondary estimation check.

```math
g(x)=w^\top x+c,\qquad
E_j^{(q)}=w_j(x_j-b_{q,j}),\qquad q\in\{0,1\}.
```

| Explanation reference | Evaluator `b0` | Evaluator `b1` |
|---|---|---|
| `E^(0)` | Matched: FC = 1, R = 0 | Mismatched: FC = r01, R = R01 |
| `E^(1)` | Mismatched: FC = r01, R = R01 | Matched: FC = 1, R = 0 |

```math
r_{01}=\operatorname{Corr}_{j}(E_j^{(0)},E_j^{(1)}),\qquad
R_{01}=\mathbb{E}_{S}\left[\sum_{j\in S}
(E_j^{(0)}-E_j^{(1)})\right]^2.
```

Correlation predictions require nonconstant vectors; the symmetric off-diagonal values additionally use the same subset law and deterministic replacements. Positive affine compatibility can keep `r01` equal to one. Consequently, baseline mismatch is not a prediction of inevitable score decline. Analyze the effect-vector covariance predicted in HYP-02, including a mismatch case that preserves correlation and one that changes it.

Baseline vectors must be recorded explicitly. A population zero mean and a zero baseline coincide on centered data, creating a null manipulation. In the pinned Quantus v0.6.0 batch replacement path for these flattened tabular inputs, `mean` denotes each input's coordinate mean, while `black` denotes its minimum; neither should be silently substituted for training-set feature means or literal zero.

Planned outputs are the compatibility matrix, paired attribution–response scatterplots, and observed-versus-predicted correlations. Holding a row fixed isolates evaluator changes while preserving the explanation's original reference semantics. Comparing rows also changes what the explanation means. Neither comparison alone licenses the claim that explanation quality intrinsically worsened.

### Evidence/Justification

**Reference semantics:** M. Sundararajan and A. Najmi, “The many Shapley values for model explanation,” in *Proc. 37th Int. Conf. Machine Learning*, vol. 119, 2020, pp. 9269–9278. [Paper](https://proceedings.mlr.press/v119/sundararajan20b.html).

**Fresh baseline inspection:** Freshly inspected through GitHub on November 4, 2026: Quantus tag `v0.6.0`, commit `85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a`. The [batch replacement function](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/functions/perturb_func.py) requests batched baselines; [baseline utilities](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/helpers/utils.py) compute means and minima across each input's flattened coordinates. This verifies source behavior of the pinned release; it is not experiment execution, an audit of the project repository, or a claim about the latest release.

**Project derivation / proposed test:** The matrix values follow linear replacement algebra and Pearson symmetry, conditional on the stated contracts. They are predictions, not newly observed results. The design tests semantic compatibility and metric response; it does not establish a universally preferable baseline.

### Easy explanation

Make two exact explanations, each using a different reference baseline. Evaluate each explanation using both replacement baselines, giving a two-by-two comparison.

The matched cells establish compatibility. The crossed cells show what changes when the explanation and evaluator use different references. The prediction comes from the resulting contribution patterns. A mismatch can change the score, but a correlation-preserving mismatch is also possible and must be included.

**Source references:** `REF-SHAPLEY`, `REF-BHATT`, `SRC-DESIGN`, `SRC-QUANTUS-060-BASELINES`, `SRC-QUANTUS-060-PERTURB`, `SRC-QUANTUS-060-TAG`.


<a id="exp-03"></a>

## EXP-03 — E3 — Locate the boundary created by feature interactions

Compare additive nonlinear and interaction models to test when exact Baseline Shapley allocations cease to reproduce arbitrary deletion responses.

### Overview

- **Who:** The experiment runner changes the known model mechanism.
- **What:** Additive structure contrasted with a controlled interaction term.
- **When:** Exact explanations are recomputed under one declared convention.
- **Where:** Small synthetic models with explicit baseline cooperative games.
- **Why:** Distinguish interaction incompatibility from generic nonlinearity.
- **How:** Test a closed-form residual law and inspect FC separately.

### Details

Define a scalar-output family and a zero-baseline cooperative game. A coalition retains its coordinates and replaces all others with zero. Use exact Baseline Shapley allocation for that game; avoid treating feature support alone as a complete quantitative explanation oracle.

```math
g_\gamma(x)=w^\top x+\gamma x_1x_2,\qquad b=0.
```

```math
\phi_1=w_1x_1+\frac{\gamma x_1x_2}{2},\qquad
\phi_2=w_2x_2+\frac{\gamma x_1x_2}{2},\qquad
\phi_j=w_jx_j\quad(j\ge3).
```

Vary interaction strength, including zero, while holding the input and baseline fixed within each paired comparison. Recompute the exact explanation whenever the model changes. Include a separate additive nonlinear control of the form `c + sum_j h_j(x_j)`, with exact reference differences. Such coordinatewise nonlinearity preserves the subset-sum identity and therefore separates interaction effects from nonlinearity in general.

Under uniform fixed-size subset replacement, HYP-03 predicts:

```math
R(\phi;x,0,k)=
\frac{\gamma^2x_1^2x_2^2\,k(d-k)}{2d(d-1)}.
```

The diagnostic is positive only when the selected input activates the interaction and subsets sometimes remove exactly one interaction partner. Subsets containing neither or both partners supply internal zero-residual controls. Exact enumeration provides a mechanism reference; sampled estimates address estimation uncertainty separately.

Planned outputs plot residual against interaction strength, compare residuals by subset overlap with the interacting pair, and compare predicted with calculated residuals across subset sizes. Report FC alongside these diagnostics, but do not preregister universal FC decline: correlation can remain one under nonzero affine residual patterns. Optional corrupted explanations are evaluated against the same declared Baseline Shapley reference.

**Inference limit:** Model and explanation both change across interaction strengths. This experiment establishes a boundary between an allocation convention and a deletion-response identity. It does not identify a pure evaluator intervention with all explanatory content fixed, nor prove that an exact Shapley allocation is unsuitable for its own stated purpose.

### Evidence/Justification

**Explanation convention:** M. Sundararajan and A. Najmi, “The many Shapley values for model explanation,” in *Proc. 37th Int. Conf. Machine Learning*, vol. 119, 2020, pp. 9269–9278. [Paper](https://proceedings.mlr.press/v119/sundararajan20b.html).

**Residual rationale:** C.-K. Yeh *et al.*, “On the (in)fidelity and sensitivity of explanations,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019. [Paper](https://papers.neurips.cc/paper_files/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html).

**Project status:** The interaction residual law is an explicit project derivation from the specified game, not a theorem attributed to these papers. All comparisons are proposed; no experimental evidence is claimed.

### Easy explanation

Introduce a controlled interaction between two features and calculate the exact Shapley allocation for the declared baseline game. Then compare allocated contribution sums with deletion responses.

Removing one interaction partner can eliminate an effect that the allocation shared between both partners. The residual formula predicts that difference. An additive nonlinear control checks that the issue is interaction, rather than nonlinearity alone. The explanation is recomputed when the model changes.

**Source references:** `REF-SHAPLEY`, `REF-YEH`, `REF-BHATT`, `SRC-DESIGN`.


<a id="exp-04"></a>

## EXP-04 — E4 — Separate sign, magnitude, and numerical scale

Apply sign and scale transformations to a fixed contribution vector to reveal which information each comparison mechanism preserves or discards.

### Overview

- **Who:** The experiment runner transforms supplied explanation values.
- **What:** Exact, sign-reversed, and positively rescaled contributions.
- **When:** Predictor, input, target, baseline, and subsets are fixed.
- **Where:** An additive model with signed reference contributions.
- **Why:** Make metric-specific information loss directly testable.
- **How:** Compare Pearson, magnitude Spearman, and numerical residuals.

### Details

Fix a deterministic additive model and matched-baseline reference vector `v`. Supply `v`, `-v`, and `alpha v` to the evaluator, with positive scales below and above one. Fix the target's numerical units. Disable normalization and retain signs in FC, singleton FE, and the residual diagnostic. In the optional MonotonicityCorrelation branch, explicitly use `abs=True`, `normalise=False`, `features_in_step=1`, `similarity_func=correlation_spearman`, and `softmax=False`, with deterministic replacements. This compares attribution magnitudes with squared scalar-output responses.

```math
V_S=\sum_{j\in S}v_j,\qquad
C_k=\mathbb{E}_{S:|S|=k}[V_S^2],\qquad \alpha>0.
```

| Supplied explanation | Population FC / singleton FE | Magnitude MC | Residual R |
|---|---|---|---|
| Exact `v` | 1 | 1 | 0 |
| Sign reversal `-v` | -1 | 1 | 4 Ck |
| Positive scale `alpha v` | 1 | 1 | (alpha - 1)^2 Ck |

Pearson results require nonconstant paired values. MC additionally requires nonconstant magnitudes. All statements concern the explicit configurations above; they must not be transferred to probability targets, different perturbations, or normalization pipelines without rederivation.

This experiment specifies distinct explanation contracts. A sign reversal contradicts signed contribution interpretation while preserving absolute magnitudes. Positive rescaling changes numerical calibration but preserves positive association and relative ranking. Therefore scale invariance is not by itself evidence that a metric overlooks every relevant notion of quality. Conversely, a magnitude score must not be reported as evidence of signed correctness.

Planned outputs are the transformation-by-metric prediction table, paired response scatterplots, and residual against the applied scale. Report undefined cases separately. Exact reference explanations and deterministic replacements do not acquire Monte Carlo uncertainty simply by repeating identical evaluations. Any deviation from the predicted invariances requires checking preprocessing, target extraction, baseline behavior, ties, and numerical precision before drawing a scientific conclusion.

**Inference limit:** Changing the comparator or applying an absolute-value transform changes the evaluated property. Results explain disagreement between explicitly named metric configurations; they do not support ranking the configurations with a single universal validity score.

### Evidence/Justification

**Metric foundations:** U. Bhatt, A. Weller, and J. M. F. Moura, “Evaluating and aggregating feature-based model explanations,” in *Proc. IJCAI*, 2020, pp. 3016–3022, doi: [10.24963/ijcai.2020/417](https://doi.org/10.24963/ijcai.2020/417); A.-P. Nguyen and M. Rodríguez Martínez, “On quantitative aspects of model interpretability,” arXiv:2007.07584, 2020. [Nguyen paper](https://arxiv.org/abs/2007.07584).

**Project derivation / proposed test:** Table entries follow Pearson sign/scale behavior, rank preservation after magnitude squaring, and the displayed residual definition. They are analytical predictions with explicit degeneracy conditions, not observed test outcomes.

**Fresh configuration check:** Freshly inspected through GitHub on November 4, 2026: Quantus tag `v0.6.0`, commit `85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a`. [MonotonicityCorrelation](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/faithfulness/monotonicity_correlation.py) and the [Metric base](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/85bcf2c8e88f7b8f3334f552ee72f8a37e870e1a/quantus/metrics/base.py) support the explicit comparator, preprocessing, and target configuration. This verifies source behavior of the pinned release; it is not experiment execution, an audit of the project repository, or a claim about the latest release. This verifies the mapping to source parameters, not the predicted experimental outcomes.

### Easy explanation

Give the evaluator the original contribution vector, its sign reversal, and positive rescalings. Keep the model output and units fixed.

The predicted responses differ because the comparisons retain different information. Signed Pearson detects sign reversal; the magnitude-based branch does not. Positive rescaling preserves correlation but can increase numerical residual. These controls help us interpret what a metric's agreement actually establishes.

**Source references:** `REF-BHATT`, `REF-NGUYEN`, `REF-YEH`, `SRC-DESIGN`, `SRC-QUANTUS-060-MC`, `SRC-QUANTUS-060-BASE`, `SRC-QUANTUS-060-SIMILARITY`, `SRC-QUANTUS-060-TAG`.


<a id="exp-05"></a>

## EXP-05 — E5 — Relocate attribution while preserving its value distribution

Move contribution values to irrelevant coordinates while preserving sparsity and norms, and test the predicted dependence on relocated squared mass.

### Overview

- **Who:** The experiment runner relocates explanation coordinates.
- **What:** Informative attribution moved to distinct zero-effect decoys.
- **When:** The model, input, reference baseline, and value multiset stay fixed.
- **Where:** A sparse additive reference vector with sufficient decoy coordinates.
- **Why:** Isolate coordinate misassignment from changes in sparsity or scale.
- **How:** Compare relocation curves with a closed-form correlation prediction.

### Details

Let `v` be the exact matched-baseline contribution vector. Choose distinct zero-effect coordinates as decoys. Moving a contribution means swapping its value with a previously zero decoy and setting its old position to zero. There must be enough unused decoys for the declared corruption range. This operation preserves signs, the value multiset, attribution sum, nonzero count, and L1/L2 norms. The explanation alone is changed; model coordinates remain fixed.

For a permutation `P_r` that relocates a nested set of informative coordinates, HYP-01 supplies the population FC prediction:

```math
a^{(r)}=P_rv,\qquad
D_v=\sum_{j=1}^{d}(v_j-\bar v)^2>0.
```

```math
FC_{\mathrm{pop}}(a^{(r)},v)
=1-\frac{\|a^{(r)}-v\|_2^2}{2D_v}
=1-\frac{\sum_{j\in\mathcal{R}_r}v_j^2}{D_v}.
```

The final equality specifically assumes disjoint swaps into zero-effect decoys. The general permutation identity precedes it. Nested relocation sets imply a nonincreasing population score, but equal numbers of moved features need not yield equal scores. Squared relocated contribution mass, rather than count alone or L1 mass, predicts the change under this model.

Planned conditions compare prespecified relocation orders and several coefficient/input profiles, with a held-out profile or ordering reserved to assess whether the mechanism prediction transfers. Planned plots show FC against moved-feature fraction and against normalized moved squared mass; the latter has a known intercept and slope. Singleton FE checks the same coordinate correlation, while sampled FC quantifies estimation uncertainty around it. Preserve per-input results before aggregation.

A separate relabeling control applies the same coordinate permutation to predictor, input, baseline, and explanation. With correspondingly permuted masks the evaluation is identical; with newly sampled uniform masks its distribution is identical. This distinguishes explanation-only misassignment from a harmless change of feature names.

**Inference limit:** The result identifies this corruption's effect on an additive replacement contract. It does not make feature relocation a comprehensive model of explanation error, nor establish monotonic behavior for arbitrary nonlinear interactions, changing subset laws, or different comparators.

### Evidence/Justification

**Metric foundation:** U. Bhatt, A. Weller, and J. M. F. Moura, “Evaluating and aggregating feature-based model explanations,” in *Proc. 29th Int. Joint Conf. Artificial Intelligence (IJCAI)*, 2020, pp. 3016–3022, doi: [10.24963/ijcai.2020/417](https://doi.org/10.24963/ijcai.2020/417).

**Project derivation / proposed test:** The permutation identity follows equal means and equal squared norms, then HYP-01. The relocation experiment and held-out comparison are proposed. Any earlier fixed-order pilot is historical context only and is not treated here as current validation of this stronger design.

### Easy explanation

Move contribution values from relevant features into unused coordinates while preserving the collection of attribution values. This changes placement without also changing sparsity or the vector's norms.

The predicted correlation loss depends on how much squared contribution is moved, not simply how many features are moved. A separate control renames the model, input, and explanation coordinates together; that checks the difference between wrong assignment and harmless relabeling.

**Source references:** `REF-BHATT`, `SRC-DESIGN`.


<a id="hyp-01"></a>

## HYP-01 — P1 — Population FC is invariant to uniform fixed subset size in additive models

Conditional analytical prediction: changing subset size alone cannot change population Pearson FC under additive, matched-contract replacement.

### Overview

- **Who:** The analyst derives a prediction for the metric evaluator.
- **What:** Equality between subset-sum and coordinate Pearson correlations.
- **When:** Uniform fixed-size sampling and additive replacement effects hold.
- **Where:** Within a fixed input with nonzero attribution and response variance.
- **Why:** Provide a falsifiable reference for separating estimator effects.
- **How:** Cancel the shared finite-population covariance factor.

### Details

Fix the scalar model, input, explanation, and baseline. The baseline may depend on the full input, but it must remain independent of the subsequently chosen subset. Assume additive separability and let each feature's reference difference be `v`. Uniformly select exactly `k` distinct coordinates, with `1 <= k < d`.

```math
g(x)=c+\sum_{j=1}^{d}h_j(x_j),\qquad
v_j=h_j(x_j)-h_j(b_j),\qquad
U_S=\sum_{j\in S}a_j,\quad V_S=\sum_{j\in S}v_j.
```

For the coordinate inclusion indicators, sampling without replacement produces:

```math
\operatorname{Cov}_{S}(U_S,V_S)=
\frac{k(d-k)}{d(d-1)}
\sum_{j=1}^{d}(a_j-\bar a)(v_j-\bar v).
```

Applying the same identity to both variances cancels the positive common factor:

```math
FC_{\mathrm{pop}}(a;x,b,k)=
\operatorname{Corr}_{j=1}^{d}(a_j,v_j).
```

Require positive centered sums of squares for both vectors. The equality applies to arbitrary supplied explanations, including controlled corruptions, and to nonlinear coordinate functions `h_j`; linearity of the entire target is unnecessary. At `k=0` or `k=d`, the subset response is constant and Pearson is undefined. A constant explanation or effect vector is likewise outside the nondegenerate domain.

This is a population statement. Finite sampled Pearson correlation is a nonlinear estimator; its bias, variance, and chance of degeneracy may vary with subset size and budget. When `a=v`, every nondegenerate finite sample lies exactly on the identity line, so exact explanations alone may show no sampling variation. Averaging within-input correlations and pooling all attribution–response pairs are different estimands.

**Boundary / falsification:** Exact enumeration must agree under the premises. Interaction effects, nonuniform sampling, random or subset-dependent baselines, altered target scales, and nonlinear preprocessing require separate derivations. Observing a change under one of those interventions does not refute this conditional identity.

### Evidence/Justification

**Literature definition:** U. Bhatt, A. Weller, and J. M. F. Moura, “Evaluating and aggregating feature-based model explanations,” in *Proc. 29th Int. Joint Conf. Artificial Intelligence (IJCAI)*, 2020, pp. 3016–3022, doi: [10.24963/ijcai.2020/417](https://doi.org/10.24963/ijcai.2020/417).

**Project derivation:** The covariance cancellation is derived here from uniform sampling without replacement; it is not presented as a result reported by Bhatt et al. EXP-01 is a proposed verification and boundary study. No new empirical result or novelty claim is asserted.

### Easy explanation

For an additive model, deleting a subset changes the output by the sum of its feature contributions. Uniform fixed-size sampling changes the covariance and both variances by the same factor, which cancels in Pearson correlation.

Therefore the population FC value does not depend on subset size under these assumptions. Finite estimates can still behave differently. Constant vectors and other zero-variance cases do not yield a defined correlation.

**Source references:** `REF-BHATT`, `SRC-DESIGN`.


<a id="hyp-02"></a>

## HYP-02 — P2 — Baseline mismatch changes FC through effect-vector covariance

Conditional prediction: a baseline mismatch changes the evaluated contribution vector, and its covariance with the supplied vector determines FC.

### Overview

- **Who:** The analyst compares explanation and evaluator reference semantics.
- **What:** A closed-form population correlation under baseline mismatch.
- **When:** The target is linear and both baselines are fixed per input.
- **Where:** Uniform fixed-size subset replacement with nonzero variance.
- **Why:** Avoid assuming that every mismatch must reduce the score.
- **How:** Express evaluator effects as the supplied contributions plus a shift vector.

### Details

Let the explanation use reference `bA` and the evaluator replace coordinates using `bB`. Both baselines are deterministic and fixed independently of selected subsets. For a linear scalar target, define:

```math
g(x)=w^\top x+c,\qquad
a_j=w_j(x_j-b_{A,j}),\qquad
q_j=w_j(b_{A,j}-b_{B,j}),\qquad
v_j=a_j+q_j.
```

Under HYP-01's uniform fixed-size sampling conditions, population FC is the coordinate correlation between `a` and `a+q`. Using coordinate variance and covariance under the same normalization gives:

```math
FC_{\mathrm{pop}}=
\frac{\operatorname{Var}_j(a)+\operatorname{Cov}_j(a,q)}
{\sqrt{\operatorname{Var}_j(a)
[\operatorname{Var}_j(a)+2\operatorname{Cov}_j(a,q)+
\operatorname{Var}_j(q)]}}.
```

Both denominator variances must be positive. A raw baseline distance does not determine this value: feature weights, the input-relative attribution pattern, and covariance matter. If evaluator effects are a positive affine transformation of the explanation vector, correlation remains one. A common coordinate shift in the effect vector is a special case; a common shift of the raw baseline does not generally produce that special case when weights differ.

```math
v=\alpha a+\beta\mathbf{1},\quad\alpha>0
\quad\Longrightarrow\quad FC_{\mathrm{pop}}=1.
```

Negative affine slope reverses correlation, and zero slope makes evaluator variance vanish. For the two-by-two design in EXP-02, the matched cells have correlation one and zero residual; off-diagonal correlations are symmetric under the same per-input evaluation contract. They need not be smaller than the diagonal values.

**Predicted test:** Construct baseline pairs with distinct covariance structures, then compare exact population values with the formula. Include an explicit correlation-preserving mismatch and a nondegenerate mismatch that changes correlation. If chosen zero and mean baselines coincide, the intervention has not changed the reference and cannot test this prediction.

**Inference limit:** Baseline replacement changes the measurement question. An explanation can remain exact for its original reference while receiving another score under a different evaluator reference. Such a result establishes conditional compatibility, not reference-independent deterioration or a universally optimal baseline.

### Evidence/Justification

**Semantic foundation:** M. Sundararajan and A. Najmi, “The many Shapley values for model explanation,” in *Proc. 37th Int. Conf. Machine Learning*, vol. 119, 2020, pp. 9269–9278. [Paper](https://proceedings.mlr.press/v119/sundararajan20b.html).

**Project derivation:** The displayed prediction follows linear replacement algebra and HYP-01. Its covariance formula and planned test are authored for this design, not attributed as empirical findings or a quoted theorem from the reference.

### Easy explanation

Changing the evaluator's baseline changes the feature-effect vector against which the explanation is compared. The predicted FC score is determined by the correlation between those two vectors.

Baseline distance alone cannot tell us the score. Some mismatches preserve a positive affine relationship and therefore retain perfect correlation. The experiment tests the explicit covariance prediction, including cases with and without a changed score.

**Source references:** `REF-SHAPLEY`, `REF-BHATT`, `SRC-DESIGN`.


<a id="hyp-03"></a>

## HYP-03 — P3 — A pairwise interaction induces a predictable Baseline Shapley residual

Conditional prediction: exact Baseline Shapley allocations and subset-deletion responses differ when deletion removes exactly one member of an interacting pair.

### Overview

- **Who:** The analyst compares an allocation convention with deletion responses.
- **What:** A closed-form subset residual for one pairwise interaction.
- **When:** Exact Baseline Shapley uses the stated zero-baseline game.
- **Where:** Uniform subsets of a fixed size in a known scalar model.
- **Why:** Identify construct disagreement despite convention-exact explanations.
- **How:** Count subsets deleting exactly one interaction partner.

### Details

Fix input `x`, use zero baseline, and define the coalition game by retaining coalition coordinates and zeroing the rest. The scalar target contains a single pairwise product interaction:

```math
g_\gamma(x)=w^\top x+\gamma x_1x_2.
```

Linearity and symmetry of Shapley allocation assign half of the interaction value to each partner:

```math
\phi_1=w_1x_1+\frac{\gamma x_1x_2}{2},\qquad
\phi_2=w_2x_2+\frac{\gamma x_1x_2}{2},\qquad
\phi_j=w_jx_j\quad(j\ge3).
```

Define the signed diagnostic residual as deletion response minus the sum of allocated contributions. Its square is the diagnostic in MET-RESIDUAL regardless of this sign convention.

```math
\delta_S=g_\gamma(x)-g_\gamma(x_{S\leftarrow0})
-\sum_{j\in S}\phi_j
=\begin{cases}
\gamma x_1x_2/2,& |S\cap\{1,2\}|=1,\\
0,& |S\cap\{1,2\}|\in\{0,2\}.
\end{cases}
```

Uniform sampling of `k` distinct coordinates gives:

```math
\Pr(|S\cap\{1,2\}|=1)=\frac{2k(d-k)}{d(d-1)},\qquad
R(\phi;x,0,k)=\mathbb{E}_{S}[\delta_S^2]
=\frac{\gamma^2x_1^2x_2^2k(d-k)}{2d(d-1)}.
```

The residual vanishes when the interaction strength or either active coordinate is zero. It also vanishes for subsets deleting neither or both partners. For `1 <= k < d`, a nonzero active interaction gives positive expected squared residual. Additive nonlinear coordinate functions alone do not create this residual; their reference contributions still sum to every replacement response.

This predicts numerical disagreement between exact allocation and a deletion identity. It does not imply that Pearson FC is always below one: correlation discards affine scale and shift, and can remain perfect despite nonzero residual. The residual formula requires no correlation nondegeneracy assumption, whereas any accompanying FC calculation does.

**Falsification / boundary:** Exact calculations must reproduce the three overlap cases and their weighted average. Different reference games, additional interactions, probability transformations, or nonuniform subsets require new calculations. The term “exact explanation” always remains qualified by the declared Baseline Shapley game.

### Evidence/Justification

**Convention reference:** M. Sundararajan and A. Najmi, “The many Shapley values for model explanation,” in *Proc. 37th Int. Conf. Machine Learning*, vol. 119, 2020, pp. 9269–9278. [Paper](https://proceedings.mlr.press/v119/sundararajan20b.html).

**Project derivation:** The half-interaction allocation follows the specified game; the deletion residual and combinatorial averaging are derived here. They are not claimed as a theorem reported by that paper or as executed experimental findings. EXP-03 is the proposed verification and boundary investigation.

### Easy explanation

The Shapley explanation splits the specified two-feature interaction equally between its partners. Deleting exactly one partner removes the whole interaction, so the deletion response and the allocated contribution sum differ by half of that interaction.

Counting how often this case occurs gives the residual formula. This predicts numerical disagreement under a particular game and deletion rule. It does not imply that every corresponding Pearson correlation must fall below one.

**Source references:** `REF-SHAPLEY`, `REF-YEH`, `SRC-DESIGN`.


<a id="ana-outcomes"></a>

## ANA-OUTCOMES — Scores, Pairwise Judgments, and Ties

Define score effects and explanation-ranking effects separately, with explicit score direction, aggregation, tie rules, and uncertainty.

### Overview

- **Who:** Research analyst and metric reviewer.
- **What:** Primary score contrasts and pairwise judgment changes.
- **When:** After valid metric outputs are recorded, before causal diagnosis.
- **Where:** Within each metric, oracle world, and declared explanation pair.
- **Why:** A numerical score shift need not change which explanation is preferred.
- **How:** Orient scores, estimate paired margins, and apply declared decision rules.

### Details

For metric m, set its direction s_m to +1 when larger scores are preferred and −1 otherwise. Preserve raw scores; use q = s_m M only for oriented comparisons. Use equal-weight input means as the primary aggregation, retaining instance-level contrasts separately. Let q̄ denote the resulting estimate, with evaluator repetitions summarized within each input.

```math
\Delta_e = \bar q_{e,\theta_1}-\bar q_{e,\theta_0},\qquad
D_\theta = \bar q_{A,\theta}-\bar q_{B,\theta}.
```

Δ measures the condition effect on one explanation. D measures the preference for A over B under one condition. A reversal requires opposite signs for the two pairwise margins beyond a prespecified practical tie tolerance. Report the margins and their uncertainty alongside any reversal rate; include ties, unresolved comparisons, and invalid evaluations in the outcome accounting.

An exact numerical tie, a practical tie within tolerance, and a comparison unresolved by uncertainty are distinct labels. A point estimate near zero does not establish population equivalence. Rank of aggregated scores and average instance rank answer different questions and must not be interchanged. Raw score magnitudes are compared only within the same metric variant. Explanation quality is interpreted through the declared reference relation, not inferred from the sign of Δ alone.

For each prespecified contrast, compare the same inputs and matched repetitions that are jointly valid across all arms of that contrast. Record this common eligibility set, joint-validity probability, and excluded cases. This is a conditional finite-score estimand, not an unconditional population claim. If validity depends materially on the condition, investigate the selection mechanism and report the broader ranking as unresolved; use analytical or exact-enumeration values where available to assess the intended population contrast. Separate per-arm valid-only means may be descriptive but are not the primary condition comparison.

### Evidence/Justification

**Proposed analysis plan; no new experimental results.** These endpoints operationalize the current research question in SRC-DESIGN. Their interpretation follows the simulation-study distinction between estimands and performance measures: T. P. Morris, I. R. White, and M. J. Crowther, “Using simulation studies to evaluate statistical methods,” *Statistics in Medicine*, vol. 38, no. 11, pp. 2074–2102, 2019, [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086).

### Easy explanation

Keep two comparisons visible: the change in one explanation's score and the score gap between two explanations. A preference reversal concerns the second comparison.

Compare the same eligible inputs and matched repetitions across the relevant arms. If different conditions fail on different cases, separate valid-only averages can be misleading. Report failures and the conditioning rule alongside score differences, practical ties, and uncertainty.

**Source references:** `SRC-DESIGN`, `REF-MORRIS`.


<a id="ana-uncertainty"></a>

## ANA-UNCERTAINTY — Sampling Uncertainty and Replication Hierarchy

Separate input-level uncertainty, evaluator Monte Carlo variation, finite-estimator bias, and undefined outcomes before interpreting apparent metric disagreement.

### Overview

- **Who:** Statistical analyst.
- **What:** Uncertainty in metric scores and paired contrasts.
- **When:** During design and after repeated evaluations.
- **Where:** Within inputs and across independently sampled inputs.
- **Why:** Internal perturbations are not independent experimental input units.
- **How:** Repeat matched draw plans and preserve the sampling hierarchy in analysis.

### Details

Keep the model, input, explanation, and evaluator settings fixed when estimating conditional Monte Carlo variability. Use independent draw plans between repetitions, while pairing comparable arms within each repetition. If inference concerns an input population, independently sampled inputs provide its replication units; many subsets from one input do not increase that input sample size. Resampling must retain the explanation/condition pairing and distinguish input sampling from evaluator variability.

Where these moments exist, define estimator bias and variance relative to the population metric M∞:

```math
\operatorname{Bias}_\xi = \mathbb E_\xi[\widehat M_B]-M_\infty,
\qquad
\operatorname{Var}_\xi = \operatorname{Var}_\xi(\widehat M_B).
```

Use analytic values or exact enumeration when available. Without such a reference, repetitions estimate variability rather than the realized estimation error or bias. Increasing budget B does not guarantee that every score moves toward an ideal value. In particular, finite-sample correlation can exhibit bias, variable precision, and zero-variance failures.

Retain attempted, valid, nonfinite, and exception counts with reasons. An undefined correlation is neither zero faithfulness nor a low score; valid-only summaries require their conditioning set to be reported. For invariance claims, use analytic derivation or prespecified equivalence margins with an appropriate interval-based procedure. Failure to reject a difference is not evidence of equivalence; wide intervals remain inconclusive.

Apply ANA-OUTCOMES' common eligibility rule before score contrasts. Conditioning on jointly valid estimates can itself select cases, so retain validity probabilities and examine their dependence on the conditions. More repetitions do not remove this selection issue. Undefined comparisons remain undefined.

### Evidence/Justification

**Proposed statistical protocol; no repetitions executed.** Morris et al. [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086) support explicit estimands and Monte Carlo uncertainty. D. Lakens, “Equivalence tests: A practical primer for t tests, correlations, and meta-analyses,” *Social Psychological and Personality Science*, vol. 8, no. 4, pp. 355–362, 2017, [doi:10.1177/1948550617697177](https://doi.org/10.1177/1948550617697177), supports the distinction between nonsignificance and equivalence.

### Easy explanation

A random evaluator can return different estimates even when the explanation and settings stay fixed. Repeat its draw plan independently to measure that variability. Use multiple independent inputs when the claim concerns an input population.

Many subsets from one input do not become many independent inputs. Also retain undefined results: removing them changes which cases the summary describes. Evidence of little change requires a suitable precision or equivalence analysis.

**Source references:** `SRC-DESIGN`, `REF-MORRIS`, `REF-LAKENS`.


<a id="ana-diagnosis"></a>

## ANA-DIAGNOSIS — Mechanism Diagnosis from Computation Traces

Test whether declared semantic changes and estimation effects explain the observed response, using intermediate calculations and controls that remove the proposed cause.

### Overview

- **Who:** Mechanism analyst and implementation auditor.
- **What:** Explanations of metric behavior under controlled changes.
- **When:** After contract checks and uncertainty assessment.
- **Where:** At the individual metric calculation and condition contrast.
- **Why:** A response curve alone cannot identify its generating mechanism.
- **How:** Compare frozen predictions with traces and discriminating controls.

### Details

Specify whether each mechanism prediction targets a population metric, an expected finite-budget estimator, or a distribution of estimates. Record the actual subsets, attribution sums U, output changes V, replacements, preprocessing, and validity states. Traces locate discrepancies within the calculation.

```text
Contract and trace audit → predicted response comparison
→ cause-removing control → remaining discrepancy assessment
```

Controls must target a named mechanism while retaining other relevant conditions. Examples include restoring explanation/evaluator reference compatibility, replacing sampling with exact subset enumeration, or testing a predicted Pearson scaling invariance. Analyze fixed explanations and recomputed matched references as separate arms. A successful control strengthens a mechanism account only if competing causes were also considered.

```math
r_{\mathrm{pred}}=\widehat M-M_{\mathrm{pred}}.
```

This prediction residual concerns the metric score; it is not the predictive model's error against an outcome label. It may include sampling error, finite-estimator bias omitted from the prediction, incorrect assumptions, or implementation discrepancies. Evaluate it against the stated prediction target and uncertainty envelope.

Diagnostic labels may coexist: **predicted semantic effect**, **estimation effect**, and **unresolved discrepancy**. They describe supported explanations of the response, not mutually exclusive metric-quality classes. An unresolved discrepancy is a prompt for further discrimination, not proof that the metric is defective. Within audited assumptions, systematic residuals exceeding the prespecified tolerance challenge the mechanism prediction.

### Evidence/Justification

**Proposed diagnosis procedure; no empirical causes established.** The residual is a study definition. Meta-evaluation is established prior work: A. Hedström et al., “The Meta-Evaluation Problem in Explainable AI: Identifying Reliable Estimators with MetaQuantus,” *Transactions on Machine Learning Research*, 2023, [arXiv:2302.07265](https://arxiv.org/abs/2302.07265). Trace-based discrimination is the present proposed extension, whose novelty remains unverified.

### Easy explanation

Use the intermediate calculations to test a specific reason for the observed response. If baseline mismatch is the proposed cause, check the predicted response and a condition that restores the match.

The remaining prediction residual tells us what the account has not explained. It may contain sampling error, omitted bias, incorrect assumptions, or a calculation problem. Several causes can coexist; an unresolved discrepancy is a research finding to investigate.

**Source references:** `SRC-DESIGN`, `REF-METAQUANTUS`, `SRC-STATUS`.


<a id="ana-confirm"></a>

## ANA-CONFIRM — Held-Out Conditions and Frozen Predictions

Test mechanism predictions on reserved conditions; use fresh random draws to assess repeatability.

### Overview

- **Who:** Research owner and confirmation analyst.
- **What:** Prospective tests of mechanism predictions.
- **When:** After predictions and decision rules are frozen.
- **Where:** Reserved evaluator conditions or oracle configurations.
- **Why:** Explaining an existing curve does not establish predictive validity.
- **How:** Separate development data, confirmation conditions, and random repetitions.

### Details

Before inspecting confirmation outcomes, fix the hypothesis, applicable assumptions, prediction target, held-out condition set, uncertainty method, practical tolerances, and failure handling. Archive this specification separately from the resulting evidence. Reserve a compact set of condition values that distinguish the mechanism account from plausible alternatives.

Fresh seeds under an already examined condition assess stochastic repeatability. They do not constitute a test on a new condition. Held-out conditions may include unexamined subset sizes, baseline values within a declared compatibility family, or controlled interaction strengths. Fresh inputs test transfer across inputs; deliberate assumption violations test a stated boundary and must not be reported as failures of a theorem outside its scope.

For example, if a derivation predicts population FC invariance to subset size in a uniform-sampling additive world, test reserved subset sizes and input profiles. Specify finite-budget behavior separately, because population invariance does not imply identical finite-sample distributions. Exact enumeration, where feasible, isolates the population prediction from sampling variation.

```text
Develop mechanism account → freeze predictions and holdouts
→ evaluate reserved conditions → compare with predicted response
```

Use fresh independent draw streams for confirmation, retaining valid pairing across its compared arms. Evaluate prediction residuals, pairwise judgment predictions, and nonfinite outcomes against the frozen criteria. A failed prediction requires a recorded revision; the revised account needs fresh confirmation conditions. Report support only for the tested conditions and declared scope, including inconclusive results.

### Evidence/Justification

**Proposed confirmation design; no holdouts evaluated.** SRC-DESIGN defines this prospective test. SRC-STATUS identifies held-out mechanism prediction as an open research requirement. Morris et al., “Using simulation studies to evaluate statistical methods,” 2019, [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086), provides the general basis for defining aims, data-generating mechanisms, estimands, methods, and performance measures in advance.

### Easy explanation

Before looking at the reserved results, write down what the mechanism predicts, where it applies, and what would challenge it. Then test the reserved conditions without revising the prediction to fit them.

Fresh seeds check repeatability at a setting. Reserved settings test the account's range. If a failed prediction leads to a revision, that revised account needs fresh confirmation conditions.

**Source references:** `SRC-DESIGN`, `SRC-STATUS`, `REF-MORRIS`.


<a id="ana-alignment"></a>

## ANA-ALIGNMENT — DMS and Response Alignment as Secondary Summaries

Use ordinal response alignment only after justifying the intervention's quality direction; retain scores, uncertainty, and mechanism tests as primary evidence.

### Overview

- **Who:** Research analyst and presentation reviewer.
- **What:** Secondary summaries of metric response across intervention levels.
- **When:** After defining the intervention and expected direction.
- **Where:** Within one metric variant and controlled condition family.
- **Why:** Monotone response is useful but cannot establish general reliability.
- **How:** Calculate oriented Spearman alignment and state its limits.

### Details

Let ℓ order intervention strength, s_m indicate the metric's preferred score direction, and s_q indicate whether the declared reference quality should increase (+1) or decrease (−1) as ℓ increases. With the prespecified mean raw score M̄ at each level:

```math
A_m=s_m s_q\,\rho_S(\ell,\bar M_m(\ell)).
```

For an intended degradation level λ and scores Q already oriented higher-is-better, the historical DMS convention is:

```math
\mathrm{DMS}_m=-\rho_S(\lambda,\bar Q_m(\lambda)).
```

A positive alignment indicates agreement with the stipulated ordinal direction. It does not measure effect magnitude, prediction accuracy, sampling uncertainty, or explanation correctness. Its interpretation requires an independently justified quality relation; increasing relocation count alone need not imply equal increments of corrupted attribution mass.

A perfect monotone ordering across a few mean values can occur under several mechanisms, including evaluator-reference mismatch. It cannot distinguish them or establish robustness beyond the tested settings. Constant responses make Spearman correlation undefined; when invariance is the prediction, assess score contrasts and equivalence instead of forcing an alignment score. Record ties and propagate paired input/repetition uncertainty if alignment uncertainty is reported.

Response Alignment is an outer analysis across intervention levels. Any Spearman operation inside MonotonicityCorrelation belongs to that metric's own calculation and is a different quantity. Use alignment as a compact descriptive complement to raw curves, paired judgments, and prospective mechanism predictions.

### Evidence/Justification

**Method definition and historical context; no new result.** SRC-STATUS reports earlier six-level, fixed-order pilot means with Response Alignment +1 for FC and MonotonicityCorrelation. These are documented researcher-provided observations, not independently rerun here. Their limited scope motivates the present mechanism analysis; no publication-level reliability claim follows from that pilot.

### Easy explanation

DMS or Response Alignment summarizes whether a score curve moves in an expected direction as an intervention becomes stronger. First justify why that intervention should improve or degrade the chosen explanation property.

A perfectly ordered curve does not reveal its cause or effect size. Several mechanisms can produce it. If the prediction is no change, inspect differences and uncertainty instead of forcing a monotonic-alignment score.

**Source references:** `SRC-DESIGN`, `SRC-STATUS`.


<a id="ctrl-pairing"></a>

## CTRL-PAIRING — Matched Draws with Preserved Sampling Laws

Couple comparable metric evaluations through explicit draw plans while preserving each condition’s intended sampling distribution and estimator dependence.

### Overview

- **Who:** Evaluation designer and statistical reviewer.
- **What:** A shared-randomness control for condition and explanation comparisons.
- **When:** Draw plans are generated before or during evaluation.
- **Where:** Inside the metric's stochastic sampling process.
- **Why:** Unrelated random choices can obscure comparative effects.
- **How:** Match appropriate random inputs and verify each arm's sampling law.

### Details

Pairing is an input to evaluation, not a selection of results afterward. With identical subset size and sampling rules, reuse the same sampled feature subsets across comparable explanations or evaluator conditions. Preserve the complete intended draw law in each arm, including within-repetition independence; matching only a marginal subset frequency is insufficient.

For different subset sizes k, use a common random permutation for each draw and select its first k features. Under uniform permutations, each arm retains uniform size-k subsets, although the subsets are not identical. For different budgets B, use prefixes of a shared independent draw stream. These nested score estimates are dependent and must be analyzed accordingly.

If selection is explanation-dependent by definition, couple primitive randomness while applying each arm's own selector; imposing identical masks could change the metric. When a valid coupling is unavailable, use an explicitly unpaired comparison with its corresponding uncertainty analysis. A common seed alone does not guarantee matching if execution paths consume randomness differently.

```math
\operatorname{Var}(\widehat M_A-\widehat M_B)
=\operatorname{Var}(\widehat M_A)+\operatorname{Var}(\widehat M_B)
-2\operatorname{Cov}(\widehat M_A,\widehat M_B).
```

Positive covariance reduces comparison variance; pairing does not guarantee this in every setting. Use independent plans between repetitions and fresh streams for confirmation, while preserving within-repetition pairing. Record draw identities, coupling rules, and any law violations in the provenance manifest.

### Evidence/Justification

**Proposed control with mathematical justification; no variance reduction observed.** The variance identity is algebraic. Uniform-permutation prefixes yield uniform subsets because every size-k subset has the same number of compatible permutations. The protocol follows the general simulation-design principles of Morris et al., 2019, [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086); its concrete coupling is specified in SRC-DESIGN.

### Easy explanation

Decide the random evaluation draws before calculating scores. When two settings use the same subset size, they can evaluate corresponding subsets. When sizes differ, one random feature ordering can generate the appropriate prefix for each setting.

This makes comparisons more directly matched while preserving each setting's sampling rule. It does not guarantee lower variance. Use new independent draw plans between repetitions and record exactly how the pairing worked.

**Source references:** `SRC-DESIGN`, `REF-MORRIS`.


<a id="ctrl-provenance"></a>

## CTRL-PROVENANCE — Configuration, Trace, and Evidence Provenance

Make every score and research claim traceable to its measurement contract, inputs, evaluator configuration, computation records, and evidence status.

### Overview

- **Who:** Experiment maintainer and independent reviewer.
- **What:** Reproducibility records and explicit evidence labels.
- **When:** At specification, evaluation, analysis, and reporting.
- **Where:** The experiment manifest, trace records, and linked sources.
- **Why:** A named metric alone does not identify its operational calculation.
- **How:** Record resolved settings, versions, failures, and claim provenance.

### Details

Assign stable identifiers to oracle worlds, inputs, explanations, conditions, repetitions, and draws. Record model coefficients or provenance, scalar output target and class selection, explanation meaning, explanation reference, intervention parameters, and all score aggregation rules. Preserve original and transformed explanations with signedness, absolute-value conversion, and normalization choices explicitly identified.

For each metric variant, record its implementation version and revision, perturbation and similarity functions, replacement values actually used, subset law and size, sampling budget, score direction, coupling rule, and random-stream provenance. Names such as a baseline preset must be resolved to their operative values or rules. A similarity override or altered preprocessing receives a distinct variant identifier when it changes the measurement definition.

Store intermediate values for reconstruction: subset membership, U and V where applicable, perturbed target outputs, intermediate validity checks, raw instance scores, and aggregation membership. Log exceptions, zero-variance correlations, missing outputs, and nonfinite results with their reasons. Do not silently replace failed evaluations or exclude them from outcome counts.

Use explicit claim labels: **proposed** for an unexecuted method; **derived** for a stated mathematical consequence; **implemented** for inspected code; **observed** for identified recorded outputs; and **unverified** for an unresolved assertion. A derivation is not an experimental observation, and inspected code is not evidence of successful execution. Archive frozen predictions separately from subsequent interpretations and revisions.

### Evidence/Justification

**Proposed provenance control; no new execution or implementation audit.** SRC-STATUS distinguishes implementation, historical researcher-provided Colab observations, and unverified causes of missing outputs. It records that original pilot CSV files were not supplied in that review session. This node specifies the stronger records required for future analysis; it does not imply that those records already exist.

### Easy explanation

A future reader should be able to identify what produced every score: the input, explanation, target, settings, implementation revision, random draws, and intermediate quantities.

Record what failed as well as what succeeded. Label whether a statement is proposed, derived, implemented, observed, or unverified. A source inspection tells us what code does; an identified execution record is needed to claim that an experiment produced a result.

**Source references:** `SRC-DESIGN`, `SRC-STATUS`.


<a id="ctrl-scope"></a>

## CTRL-SCOPE — Construct Compatibility and Limits of Transfer

Bound conclusions by explanation meaning and evaluator assumptions; distinguish transfer evidence from universal validity or unverified novelty.

### Overview

- **Who:** Research owner and methodological reviewer.
- **What:** Interpretation boundaries for interventions and conclusions.
- **When:** Before prediction, comparison, diagnosis, and external transfer.
- **Where:** Every oracle world and metric variant.
- **Why:** Holding an explanation vector fixed does not guarantee unchanged measurement meaning.
- **How:** Audit compatibility and attach explicit assumptions to each claim.

### Details

The measurement contract specifies both the explanation construct and its evaluation. Distinguish the explanation reference from the evaluator's replacement baseline. Changing only the latter can alter the relation being tested even when the numerical explanation remains fixed. Before interpretation, classify interventions as construct-preserving, intentionally semantic, or unresolved in compatibility; justify the classification.

Use fixed-explanation and recomputed matched-reference arms separately when investigating baseline compatibility. In interaction worlds, state the contribution-allocation rule and derive the corresponding reference; a unique universal attribution must not be presumed. Positive scaling preserves defined Pearson correlations but can change absolute contribution error. Thus an unchanged metric score can reflect its measurement invariance, while whether explanation quality changed depends on the chosen definition.

An oracle reference is exact relative to a specified model, scalar target, reference, and attribution definition. It does not establish human usefulness or real-world causal truth. Additive tabular results delimit a mechanism under stated assumptions; they do not automatically cover images, probabilities instead of logits, other perturbation distributions, or every metric within a category.

Real-data extensions may test whether a diagnosed behavior recurs, but without a validated reference they cannot directly certify explanation correctness. Compactness and Robustness/Continuity extensions require their own contracts. State novelty as an open comparison with prior meta-evaluation and sanity-check studies, rather than claiming the general problem is new.

### Evidence/Justification

**Proposed scope control; no transfer or novelty established.** Relevant prior work includes R. Tomsett et al., “Sanity Checks for Saliency Metrics,” *AAAI*, 2020, [doi:10.1609/aaai.v34i04.6064](https://doi.org/10.1609/aaai.v34i04.6064), and A. Hedström et al., “The Meta-Evaluation Problem in Explainable AI: Identifying Reliable Estimators with MetaQuantus,” *TMLR*, 2023, [arXiv:2302.07265](https://arxiv.org/abs/2302.07265). SRC-STATUS preserves the project's unresolved scope and novelty questions.

### Easy explanation

Every conclusion has a boundary. An exact reference is exact for a declared explanation question, and a successful metric comparison establishes that particular relation.

Changing the baseline, output, representation, or intervention can change the question. The controlled additive experiments provide a starting account of specific mechanisms. Broader data types and other metric families need their own contracts and evidence.

**Source references:** `SRC-DESIGN`, `SRC-STATUS`, `REF-TOMSETT`, `REF-METAQUANTUS`.


<a id="ref-bhatt"></a>

## REF-BHATT — Evaluating and Aggregating Feature-based Model Explanations

Definitions for faithfulness, sensitivity, and attribution complexity under explicit evaluation choices.

### Overview

- **Who:** Bhatt, Weller, and Moura.
- **What:** explanation evaluation and aggregation.
- **When:** 2020.
- **Where:** IJCAI.
- **Why:** formal definitions for our metric contracts.
- **How:** mathematical criteria and empirical comparisons.

### Details

Faithfulness compares attribution aggregates with prediction changes under subset replacement; complexity uses entropy of normalized absolute attributions. Baselines and subset sizes therefore belong in the measurement specification. Here, these definitions motivate controlled evaluator interventions; they do not establish universal explanation quality.

### Evidence/Justification

U. Bhatt, A. Weller, and J. M. F. Moura, “Evaluating and aggregating feature-based model explanations,” in Proc. 29th Int. Joint Conf. Artif. Intell. (IJCAI), 2020, pp. 3016–3022, doi: 10.24963/ijcai.2020/417.

Primary source: https://ijcai.org/proceedings/2020/417

Reviewed: published text, definitions, and experimental settings. Supports metric specification, not our proposed outcomes.

### Easy explanation

Use this paper to anchor the metric definitions and the choices that enter them. Our subset-size prediction is a separate project derivation with explicit assumptions. Citing the definition paper does not mean it already demonstrated our proposed experiments.

**Source references:** `REF-BHATT`.


<a id="ref-alvarez"></a>

## REF-ALVAREZ — Towards Robust Interpretability with Self-Explaining Neural Networks

Local stability and feature-removal evaluation depend on the explanatory representation.

### Overview

- **Who:** Alvarez-Melis and Jaakkola.
- **What:** self-explaining neural networks.
- **When:** 2018.
- **Where:** NeurIPS.
- **Why:** stability and faithfulness foundations.
- **How:** constrained architecture, regularization, and evaluation.

### Details

The paper formalizes locally stable explanatory coefficients and evaluates relevance against feature-removal responses. Its model-specific explanations and intervention procedures limit direct generalization. Here, it motivates declaring representation, neighborhood, and removal rules before interpreting robustness or faithfulness scores.

### Evidence/Justification

D. Alvarez-Melis and T. S. Jaakkola, “Towards robust interpretability with self-explaining neural networks,” in Advances in Neural Information Processing Systems, vol. 31, 2018, pp. 7775–7784.

Primary source: https://proceedings.neurips.cc/paper/2018/hash/3e9f0fc9b2f89e043bc6233994dfcf76-Abstract.html

Reviewed: published paper, especially stability definitions and evaluation sections. Supports design rationale, not our results.

### Easy explanation

This paper connects explanation structure, local stability, and feature-removal evaluation. It helps us ask what representation and intervention a score refers to. Its setting should be checked before transferring a claim to another explanation type.

**Source references:** `REF-ALVAREZ`.


<a id="ref-nguyen"></a>

## REF-NGUYEN — On quantitative aspects of model interpretability

Monotonicity measures rank agreement with a specified feature-perturbation response.

### Overview

- **Who:** Nguyen and Rodríguez Martínez.
- **What:** quantitative interpretability metrics.
- **When:** 2020.
- **Where:** arXiv preprint.
- **Why:** Monotonicity Correlation foundations.
- **How:** definitions and controlled examples.

### Details

Metric 2.3 compares absolute-attribution ranks with expected loss under feature perturbations using Spearman correlation. The perturbation distribution changes what response is measured. Here, this grounds a rank-based measurement contract; correspondence with a particular Quantus implementation requires separate code verification.

### Evidence/Justification

A.-P. Nguyen and M. Rodríguez Martínez, “On quantitative aspects of model interpretability,” arXiv:2007.07584v1, 2020, doi: 10.48550/arXiv.2007.07584.

Primary source: https://arxiv.org/abs/2007.07584v1

Reviewed: version 1, including Section 2.2.2. Preprint status retained; source definitions do not prove our hypotheses.

### Easy explanation

This reference explains why a monotonicity score compares attribution importance with a declared perturbation response. The response distribution matters. We separately inspect the pinned Quantus implementation because a library routine and a paper formulation must not be assumed identical.

**Source references:** `REF-NGUYEN`.


<a id="ref-yeh"></a>

## REF-YEH — On the (In)fidelity and Sensitivity of Explanations

An explanation’s optimality can depend mathematically on the evaluator’s perturbation distribution.

### Overview

- **Who:** Yeh and colleagues.
- **What:** infidelity and sensitivity.
- **When:** 2019.
- **Where:** NeurIPS.
- **Why:** direct theoretical prior art.
- **How:** perturbation-dependent loss and analytical optimization.

### Details

Infidelity is expected squared mismatch between attribution-predicted and actual model responses. Its optimizer depends on the perturbation law; minimizing sensitivity alone can yield a constant explanation. These are existing mechanism results. Here, they delimit novelty and motivate explicit response, loss, and intervention contracts.

### Evidence/Justification

C.-K. Yeh, C.-Y. Hsieh, A. S. Suggala, D. I. Inouye, and P. Ravikumar, “On the (in)fidelity and sensitivity of explanations,” in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 10967–10978.

Primary source: https://proceedings.neurips.cc/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html

Reviewed: published paper, Definition 2.1 and Proposition 2.1. Their assumptions do not automatically hold in our experiments.

### Easy explanation

This is direct prior work on how an evaluation objective depends on its perturbation rule. It gives our study a foundation and limits broad novelty claims. We still need to derive and test the specific relationships proposed in this Atlas.

**Source references:** `REF-YEH`.


<a id="ref-shapley"></a>

## REF-SHAPLEY — The Many Shapley Values for Model Explanation

Different attribution games can answer different questions about the same model.

### Overview

- **Who:** Sundararajan and Najmi.
- **What:** operationalizations of Shapley attribution.
- **When:** 2020.
- **Where:** ICML.
- **Why:** estimand compatibility.
- **How:** axiomatic analysis of alternative explanation games.

### Details

Different choices of context, baseline, and feature treatment produce different Shapley explanations. Here, this motivates matching the explanation’s target question to the evaluator’s intervention. Disagreement between differently defined questions cannot, by itself, establish poorer explanation quality.

### Evidence/Justification

M. Sundararajan and A. Najmi, “The many Shapley values for model explanation,” in Proc. 37th Int. Conf. Mach. Learn., PMLR, vol. 119, 2020, pp. 9269–9278.

Primary source: https://proceedings.mlr.press/v119/sundararajan20b.html

Reviewed: published abstract and bibliographic record. Supports the compatibility rationale; no claimed verification of our model-specific results.

### Easy explanation

Different Shapley constructions can answer different explanation questions. Use this reference when explaining why we declare a baseline game and why two reference choices need not yield the same attribution. Exactness must always name the definition being used.

**Source references:** `REF-SHAPLEY`.


<a id="ref-metaquantus"></a>

## REF-METAQUANTUS — The Meta-Evaluation Problem in Explainable AI: Identifying Reliable Estimators with MetaQuantus

Meta-evaluation already tests evaluator reliability and investigates some failure mechanisms.

### Overview

- **Who:** Hedström and colleagues.
- **What:** MetaQuantus.
- **When:** 2023.
- **Where:** TMLR.
- **Why:** closest reliability-framework precedent.
- **How:** noise resilience, adversary reactivity, and consistency tests.

### Details

The study varies Faithfulness Correlation baselines and subset sizes, and explains localization behavior through mask size and chance overlap. Thus, neither parameter sweeps nor all mechanism analysis are new here. Our proposed contribution must specify a narrower controlled mechanism claim and independent calibration.

### Evidence/Justification

A. Hedström, P. Bommer, K. K. Wickstrøm, W. Samek, S. Lapuschkin, and M. M.-C. Höhne, “The meta-evaluation problem in explainable AI: Identifying reliable estimators with MetaQuantus,” Transactions on Machine Learning Research, 2023.

Primary source: https://openreview.net/forum?id=j3FK00HyfU

Reviewed: publication record and arXiv:2302.07265v2, including Section 6.1.2 and Appendix A.6.1. Reliability evidence is not universal validity.

### Easy explanation

MetaQuantus already studies evaluator reliability and some reasons for evaluator behavior. We must compare our work with those results. The proposed contribution needs a specific controlled prediction and added evidence; a parameter sweep alone is insufficient as a novelty claim.

**Source references:** `REF-METAQUANTUS`.


<a id="ref-tomsett"></a>

## REF-TOMSETT — Sanity Checks for Saliency Metrics

Evaluator consistency needs investigation before saliency rankings can support conclusions.

### Overview

- **Who:** Tomsett and colleagues.
- **What:** saliency-metric sanity checks.
- **When:** 2020.
- **Where:** AAAI.
- **Why:** reliability precedent.
- **How:** psychometric checks and comparative evaluation.

### Details

The study reports inconsistent saliency-method judgments across evaluation choices. Here, it motivates explicit protocol definitions and ranking-stability analysis. Agreement is a reliability property; it does not independently establish that a metric measures the intended explanation construct.

### Evidence/Justification

R. Tomsett, D. Harborne, S. Chakraborty, P. Gurram, and A. Preece, “Sanity checks for saliency metrics,” in Proc. AAAI Conf. Artif. Intell., vol. 34, no. 4, 2020, pp. 6021–6029, doi: 10.1609/aaai.v34i04.6064.

Primary source: https://ojs.aaai.org/index.php/AAAI/article/view/6064

Reviewed: published abstract and bibliographic record. This evidence supports the motivation, not our proposed causal explanations.

### Easy explanation

This paper supports the need to examine the reliability of evaluators before relying on their rankings. Consistent judgments are useful evidence, but consistency alone does not prove that the score measures the explanation property we intended.

**Source references:** `REF-TOMSETT`.


<a id="ref-binder"></a>

## REF-BINDER — Shortcomings of Top-Down Randomization-Based Sanity Checks for Evaluations of Deep Neural Network Explanations

Randomization judgments have established mathematical and representational failure mechanisms.

### Overview

- **Who:** Binder and colleagues.
- **What:** top-down randomization sanity checks.
- **When:** 2023.
- **Where:** CVPR.
- **Why:** direct mechanism precedent.
- **How:** analytical arguments and empirical comparisons.

### Details

The paper examines decorrelation effects and structure surviving network randomization, showing why similarity changes can mislead. Here, it requires us to distinguish metric response from explanation degradation. Our novelty must extend beyond demonstrating that randomization or similarity measures have confounds.

### Evidence/Justification

A. Binder, L. Weber, S. Lapuschkin, G. Montavon, K.-R. Müller, and W. Samek, “Shortcomings of top-down randomization-based sanity checks for evaluations of deep neural network explanations,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 16143–16152, doi: 10.1109/CVPR52729.2023.01549.

Primary source: https://openaccess.thecvf.com/content/CVPR2023/papers/Binder_Shortcomings_of_Top-Down_Randomization-Based_Sanity_Checks_for_Evaluations_of_Deep_CVPR_2023_paper.pdf

Reviewed: published full paper. Its conclusions concern specified sanity checks, not every robustness or faithfulness metric.

### Easy explanation

This is an important precedent for explaining misleading evaluation behavior through the evaluation mechanism. It prevents us from claiming that mechanism diagnosis itself is entirely new. Our account must specify what additional question and evidence it contributes.

**Source references:** `REF-BINDER`.


<a id="ref-latec"></a>

## REF-LATEC — Navigating the Maze of Explainable AI: A Systematic Approach to Evaluating Methods and Metrics

Broad benchmarking motivates evaluating conclusions across multiple metric configurations.

### Overview

- **Who:** Klein and colleagues.
- **What:** LATEC benchmark.
- **When:** 2024.
- **Where:** NeurIPS, Datasets and Benchmarks.
- **Why:** broad empirical context.
- **How:** cross-method, metric, architecture, and modality comparisons.

### Details

The benchmark compares 17 explanation methods and 20 metrics across 7,560 combinations, documenting evaluation-dependent rankings. Here, it motivates configuration coverage and replication beyond one setting. Benchmark aggregation does not establish a universal explanation-quality ordering or identify every causal mechanism.

### Evidence/Justification

L. Klein, C. T. Lüth, U. Schlegel, T. J. Bungert, M. El-Assady, and P. F. Jäger, “Navigating the maze of explainable AI: A systematic approach to evaluating methods and metrics,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 67106–67146.

Primary source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/7beb1ddb04401ba21869f23a77f3e4e1-Abstract-Datasets_and_Benchmarks_Track.html

Reviewed: published abstract and bibliographic record. Reported benchmark findings do not constitute observations from our proposed experiments.

### Easy explanation

Use this benchmark as broad context for why evaluation choices deserve attention. It motivates checking more than one configuration. Its results should be distinguished from the tightly controlled predictions and future experiments described in this Atlas.

**Source references:** `REF-LATEC`.


<a id="ref-fidelity"></a>

## REF-FIDELITY — A comprehensive study on fidelity metrics for XAI

Published linear and nonlinear calibration results constrain claims of novelty.

### Overview

- **Who:** Miró-Nicolau, Jaume-i-Capó, and Moyà-Alcover.
- **What:** fidelity-metric calibration.
- **When:** 2025 publication.
- **Where:** Information Processing & Management.
- **Why:** oracle precedent.
- **How:** synthetic benchmarks with interpretable models.

### Details

The published abstract reports expected perfect Faithfulness Estimate and Faithfulness Correlation results for linear models, with deviations for nonlinear models. The 2024 preprint reviewed here emphasizes decision trees. Preserve this version distinction; perfect linear calibration is already prior art, and does not imply universal validity.

### Evidence/Justification

M. Miró-Nicolau, A. Jaume-i-Capó, and G. Moyà-Alcover, “A comprehensive study on fidelity metrics for XAI,” Information Processing & Management, vol. 62, no. 1, Art. no. 103900, Jan. 2025, doi: 10.1016/j.ipm.2024.103900.

Primary source: https://www.sciencedirect.com/science/article/pii/S0306457324002590

Reviewed: published abstract and full arXiv:2401.10640v1; journal full methods unverified. Supports calibration rationale, not our proposed results.

### Easy explanation

The published article already includes linear-model calibration findings, so we cannot present perfect linear calibration as a new contribution. Keep the journal abstract and older preprint separate when discussing what was actually verified. The journal's full methods still require review.

**Source references:** `REF-FIDELITY`.


<a id="ref-morris"></a>

## REF-MORRIS — Using simulation studies to evaluate statistical methods

ADEMP provides a protocol structure and explicit simulation-precision accounting.

### Overview

- **Who:** Morris, White, and Crowther.
- **What:** simulation-study methodology.
- **When:** 2019.
- **Where:** Statistics in Medicine.
- **Why:** experiment protocol structure.
- **How:** ADEMP and Monte Carlo uncertainty guidance.

### Details

ADEMP specifies aims, data-generating mechanisms, estimands, methods, and performance measures before simulation. Here, it organizes controlled metric experiments and separates finite-replication uncertainty from substantive effects. It is methodological guidance, not an XAI validity theorem or a substitute for realistic data assumptions.

### Evidence/Justification

T. P. Morris, I. R. White, and M. J. Crowther, “Using simulation studies to evaluate statistical methods,” Statistics in Medicine, vol. 38, no. 11, 2019, pp. 2074–2102, doi: 10.1002/sim.8086.

Primary source: https://onlinelibrary.wiley.com/doi/10.1002/sim.8086

Reviewed: published methods and reporting guidance. Supports protocol design and precision assessment; no experiment outcomes are implied.

### Easy explanation

This paper helps turn an experiment idea into a clear simulation protocol: state the aim, generated data, target quantity, compared methods, and performance measures. It also motivates checking how much uncertainty remains from finite repetitions.

**Source references:** `REF-MORRIS`.


<a id="ref-lakens"></a>

## REF-LAKENS — Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses

Practical invariance needs justified equivalence bounds and appropriate uncertainty analysis.

### Overview

- **Who:** Lakens.
- **What:** practical equivalence testing.
- **When:** 2017.
- **Where:** Social Psychological and Personality Science.
- **Why:** invariance inference.
- **How:** two one-sided tests and meaningful effect bounds.

### Details

A nonsignificant difference does not demonstrate equivalence. Here, invariance claims require prespecified, scientifically justified bounds and a compatible sampling analysis. Failure to establish equivalence is also inconclusive; it cannot automatically be interpreted as evidence that the metric meaningfully changed.

### Evidence/Justification

D. Lakens, “Equivalence tests: A practical primer for t tests, correlations, and meta-analyses,” Social Psychological and Personality Science, vol. 8, no. 4, 2017, pp. 355–362, doi: 10.1177/1948550617697177.

Primary source: https://journals.sagepub.com/doi/10.1177/1948550617697177

Reviewed: published abstract and bibliographic record. Supports the inference principle; our bounds and dependence model remain design decisions.

### Easy explanation

Use this reference when claiming that a change is small enough to count as practically absent. An inconclusive difference test does not establish that. The analysis needs justified equivalence bounds and an uncertainty method appropriate for the comparison.

**Source references:** `REF-LAKENS`.
