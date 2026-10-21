## F1 — Faithfulness relocation

Suppose the original oracle attribution is:

```math
E^*
=
[a_1,a_2,a_3,a_4,a_5,0,0,0,0,0,\ldots]
```

Here, $a_1,\ldots,a_5$ correspond to the five features actually used by the oracle predictive model.

Features $x_6,\ldots,x_{10}$ are not used by the predictive model, so we use them as **decoy features**.

Instead of reducing attribution magnitude, we progressively **relocate attribution values from informative features to decoy features**.

For example:

```text
0% corruption
[a1, a2, a3, a4, a5,  0,  0,  0,  0,  0, ...]

20% corruption
[ 0, a2, a3, a4, a5, a1,  0,  0,  0,  0, ...]

40% corruption
[ 0,  0, a3, a4, a5, a1, a2,  0,  0,  0, ...]

60% corruption
[ 0,  0,  0, a4, a5, a1, a2, a3,  0,  0, ...]

80% corruption
[ 0,  0,  0,  0, a5, a1, a2, a3, a4,  0, ...]

100% corruption
[ 0,  0,  0,  0,  0, a1, a2, a3, a4, a5, ...]
```

Let $r$ be the number of informative attributions relocated to decoy features.

Since there are five informative features:

```math
r = 0,\;1,\;2,\;3,\;4,\;5
```

We can also represent the corruption level as:

```math
\lambda=\frac{r}{5}
```

so that:

```math
\lambda
=
0,\;0.2,\;0.4,\;0.6,\;0.8,\;1.0
```

The important point is that the attribution values themselves are not reduced or newly created.

For every corruption level:

```math
\|E_\lambda\|_1
=
\|E^*\|_1
```

Also, the number of non-zero attribution values remains fixed:

```math
\|E_\lambda\|_0
=
\|E^*\|_0
=
5
```

Therefore, throughout this intervention:

- total attribution magnitude stays the same,
- the number of attributed features stays the same,
- attribution values and their signs stay the same,
- but fewer attribution values remain on features actually used by the predictive model.

The main variable being changed is therefore:

> **How much of the explanation attribution is placed on model-relevant features rather than irrelevant decoy features?**

This makes the intervention cleaner for testing Faithfulness because changes in metric score are less likely to be caused simply by changes in attribution magnitude or explanation sparsity.

For an ideal Faithfulness metric, as corruption $\lambda$ increases, its evaluation of the explanation should become worse.

However, different metrics may react differently even though they are all intended to evaluate Faithfulness.

That's where we take a look at the response curve per metric.

Define the quality-oriented score of metric $m$ as:

```math
Q_m(\lambda)
```

Then define the **Degradation Monotonicity Score** as:

```math
DMS_m
=
-\rho_S
\left(
\lambda,Q_m(\lambda)
\right)
```

where $\rho_S$ is the Spearman rank correlation.

If the metric behaves ideally, meaning that its score consistently decreases as corruption increases, then:

```math
DMS_m \approx 1
```

should hold.

If:

```math
DMS_m \approx 0
```

the metric does not show a consistent monotonic response to the controlled degradation.

If:

```math
DMS_m < 0
```

the metric is tending to move in the opposite direction from what we expected.

Metrics we're testing:

- Faithfulness Correlation
- Monotonicity Correlation
- Sensitivity-N

*All metrics are provided by the Quantus framework.

## C1 — Support-size compactness

```math
E_i^{(k)}=
\begin{cases}
1/k,&i\le k\\
0,&i>k
\end{cases}
```

Increasing $k$ reduces compactness under support-count semantics.

## C2 — Concentration compactness

```math
a_i(\beta)=
\frac{e^{-\beta(i-1)}}{\sum_j e^{-\beta(j-1)}}
```

All 20 values remain non-zero while larger `beta` concentrates mass.

Metrics:

- Complexity (entropy)
- Sparseness (Gini)
- Effective Complexity (threshold count)

## R1 — Controlled instability

Use:

```math
E_\lambda(x)=E^*(x)+\lambda R(x)
```

where $R(x)$ is deterministic, high-frequency, and restricted to informative features. The predictive model is unchanged.

Metrics:

- Avg Sensitivity
- Max Sensitivity
- Local Lipschitz Estimate

## Response alignment

Let:

- $d_m=+1$ if higher metric score is better, $-1$ if lower is better.
- $d_q=+1$ if quality should improve as the intervention level rises, $-1$ if quality should degrade.

```math
Alignment = \rho_S(level, score)\,d_m\,d_q
```

$+1$ means the observed metric ordering matches the expected ordinal direction.

This is a meta-analysis statistic, of explanation evaluation metrics.

## Interpretation discipline

A failed oracle response supports:

> Metric X did not behave monotonically under this specific intervention and configuration.

It does not by itself support:

> Metric X is universally invalid.
