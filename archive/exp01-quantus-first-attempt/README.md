# First Quantus smoke attempt — retained numerical diagnostic

This incomplete run used the official Quantus engine at source commit [`02cd3d2`](https://github.com/theguardianofthebed/xai-metric-behavior/commit/02cd3d2e93a9142b8ce3cbd8d1d36640dd8ab031). Its recorded files are preserved unchanged. It is not the current completed smoke.

The auxiliary parity guard stopped after 332 FC attempts at input 2, repetition 1, candidate A, k=19, B=5. Quantus returned **0.31622776601683805** and `NearConstantInputWarning`. The recorded attribution sums differed by only **8.88e-16** around 4.82889; the scaled NumPy replay returned **0.2581988897471609**. Correlation was numerically ill-conditioned. No returned value was replaced.

The revised policy preserves official scores and warnings, distinguishes finite-but-near-constant results from undefined Pearson, and excludes numerical warnings from conditional estimator/ranking summaries with explicit eligibility counts. Strict score parity applies to well-conditioned calls; all structural trace checks still apply. The new smoke is reported separately in [`results/verification/`](../../results/verification/).
