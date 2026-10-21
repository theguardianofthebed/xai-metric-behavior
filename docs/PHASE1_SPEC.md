# Phase 1 v0.1 — Experimental Specification

## G0

**Under what controlled conditions do quantitative XAI evaluation metrics behave consistently with the explanation property they claim to measure?**

## Oracle task

```math
x_j \sim \mathcal N(0,1)
```

```math
z(x)=2.5x_1-2x_2+1.5x_3+x_4-0.5x_5
```

The classifier outputs logits $[-z, z]$. Features 6–20 are irrelevant to the oracle model.

For the predicted target class:

```math
E^*(x)=s_y[w_1x_1,\ldots,w_5x_5,0,\ldots,0]
```

with $s_y=1$ for class 1 and $-1$ for class 0.

## F1 — Faithfulness relocation

Move the first five attribution values to features 6–10, creating a decoy explanation $D(E^*)$

```math
E_\lambda=(1-\lambda)E^*+\lambda D(E^*)
\]
```

Because the supports do not overlap:

```math
\|E_\lambda\|_1=\|E^*\|_1
```

The intervention therefore changes *where* attribution is placed rather than total attribution magnitude.

Metrics:

- Faithfulness Correlation
- Monotonicity Correlation
- Sensitivity-N

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
