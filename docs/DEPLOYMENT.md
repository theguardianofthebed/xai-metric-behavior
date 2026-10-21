# Deployment

## GitHub
Upload this repository as `xai-metric-behavior`.

## Google Colab
Open `notebooks/01_phase1_colab.ipynb`, set `YOUR_USERNAME`, and run all cells.

## Streamlit Community Cloud
Deploy `app/streamlit_app.py`.

Recommended Python version: **3.12**.

The app has a separate lightweight dependency file so public viewing does not install PyTorch, Captum, or Quantus.

## Codespaces
`.devcontainer/devcontainer.json` creates the Python 3.12 browser development environment automatically.
