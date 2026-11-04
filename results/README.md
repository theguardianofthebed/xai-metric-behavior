# Experiment outputs

Each run writes to a **new directory**. Existing outputs are never overwritten. Keep the complete run directory together: configuration, code/dependency provenance, inputs, raw scores, draw traces, failures and summaries.

`verification/` contains explicitly labeled implementation verification records committed with this PR. They are not held-out confirmation results. Normal smoke/development/confirmation run directories are ignored by Git because full traces can be large.

To share a completed study, archive its complete directory or publish it as a versioned research artifact; a plot alone cannot reconstruct the computation. Keep confirmation's frozen prediction file with the outputs.
