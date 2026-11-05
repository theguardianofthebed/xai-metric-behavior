# XAI Metric Behavior Lab — Phase 1 v0.1

A browser-first research scaffold for studying **how quantitative XAI evaluation metrics behave under controlled explanation changes**.

## Core question

> When the quality or structure of an explanation is changed in a known and controlled way, do quantitative XAI metrics respond in the expected way?

Phase 1 studies:

- **Faithfulness / Co-12 Correctness**
- **Compactness**
- **Robustness / stability close to Co-12 Continuity**

## Phase structure

- **P1-A Oracle:** synthetic tabular data, known linear classifier, exact attribution.
- **P1-B Learned XAI:** the same task learned by an MLP, then explained with Captum.
- **P1-C Real-data smoke test:** intentionally left for the next iteration.

## Environment

- Python 3.12
- GitHub = source of truth
- Google Colab = experiment runner
- Streamlit Community Cloud = result viewer
- GitHub Codespaces = optional browser IDE

## Quick start

```bash
python -m pip install -r requirements-experiment.txt
python -m pip install -e .
pytest -q
python scripts/run_oracle_phase1.py --config configs/phase1_quick.yaml
```

P1-B:

```bash
python scripts/run_learned_xai.py --config configs/phase1_quick.yaml
```

Public dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Reproducibility rule

Do not silently change a completed experiment configuration after viewing results. Create a new YAML file instead.

Every run stores:

- `raw_scores.csv`
- `summary.csv`
- `behavior_profiles.csv`
- `resolved_config.yaml`
- `run_metadata.json`

No fabricated research results are included.
