# Research collaboration rules

## Authority and scope

Read `docs/FRAMEWORK.md`, `docs/EXP-01.md` and the current accessible XAI Framework Atlas before changing research semantics. `docs/atlas_snapshot_2026-11-04.json` is a revision-4 source snapshot; it does not silently override later Atlas edits. `docs/ATLAS_REFERENCE.md` preserves the source wording. `archive/phase1/` and `archive/exp01-numpy/` are historical and are not the active experiment. The current EXP-01 report follows the compact seven-heading Atlas layout; preserve full historical wording in its source/archive instead of copying it into the active report.

Preserve supplied definitions and distinguish **proposed**, **derived**, **implemented**, **observed** and **unverified**. Do not claim a universal metric or unverified novelty. Inspect current code before describing implementation. Preserve source language unless translation is requested; explain research to the owner in Korean with English terms and GitHub-compatible math.

## Scientific controls

- Freeze scalar target, explanation/reference baseline, sign handling, normalization, sampling law, comparator, aggregation and failures.
- EXP-01 tests population subset-size invariance against finite estimation effects. Do not use DMS/Response Alignment to force a direction for an invariance prediction.
- Keep exact anchor and two preregistered nonexact candidates separate. Never select a pair from confirmation reversals.
- Retain all undefined/exception outcomes and reasons. Compare jointly eligible paired cases; report denominator and selection limits.
- Keep input replication separate from evaluator repetitions and subset draws. Preserve coupling when resampling or comparing arms.
- Do not consume reserved confirmation conditions during routine tests. Freeze predictions, margins, code/configuration and holdouts first. A revised prediction requires fresh confirmation.
- Never change upstream scores to obtain a preferred outcome; document and verify any compatibility workaround.

## Implementation and reporting

Use CPU-first analytical fixtures for EXP-01; no model training or external dataset is necessary. Atlas experiments default to pinned official Quantus metrics; justify and label any exception or custom variant. **Quantus 0.6.0 public FC/FE calls are the EXP-01 measurement engine.** NumPy calculations serve as analytical references and trace-replay checks, never a silent scoring fallback. Record the actual official-call subsets, attribution sums, output changes and returned scores. Wrappers must delegate to the official functions without changing their calculation; document dtype, shape, RNG and preprocessing controls.

Keep the notebook a readable front end to the same CLI package, running in an isolated environment with required Quantus and CPU PyTorch; do not upgrade the running notebook kernel's dependencies. Add tests for scientific invariants, failures, pairing and provenance, not repetitions of implementation syntax. Parity checks guard implementation; they do not establish HYP-01, explanation quality or scientific confirmation. Keep old backend results labeled historical; new measurement-engine claims require a fresh run. EXP-01-R is a proposed Atlas plan only until separately authorized for implementation.

In substantive research explanations, end with a proportionate **Research Learning Summary** for each key topic: 필요성 → 직관적인 예시/이해 → 공식적인 정의 → 근거/그게 되는 이유.
