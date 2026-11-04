# Research collaboration rules

## Authority and scope

Read `docs/FRAMEWORK.md`, `docs/EXP-01.md` and the current accessible XAI Framework Atlas before changing research semantics. `docs/atlas_snapshot_2026-11-04.json` is a revision-4 source snapshot; it does not silently override later Atlas edits. `docs/ATLAS_REFERENCE.md` preserves the source wording. `archive/phase1/` is historical and is not the active experiment.

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

Use CPU-first analytical fixtures for EXP-01; no model training or external dataset is necessary. Keep the notebook a readable front end to the same package used by the CLI. Add tests for scientific invariants, failures, pairing and provenance, not repetitions of implementation syntax. Record verification separately from scientific confirmation.

In substantive research explanations, end with a proportionate **Research Learning Summary** for each key topic: 필요성 → 직관적인 예시/이해 → 공식적인 정의 → 근거/그게 되는 이유.
