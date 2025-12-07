# Visit with Us — Wellness MLOps Pipeline

This repository contains an end-to-end MLOps scaffold for predicting customer purchases of the Wellness Tourism Package.

## Contents
- `notebook.ipynb` — single notebook implementing data registration, preprocessing, training, MLflow tracking, model push to Hugging Face, and deployment artifacts.
- `train_script.py` — script for CI/CD to run training and push model to Hugging Face.
- `app.py` — Streamlit app to serve predictions using `best_model.joblib`.
- `best_model.joblib` — model artifact (add after training).
- `data/` — `train.csv` and `test.csv` (created by the notebook).
- `.github/workflows/pipeline.yml` — GitHub Actions workflow to run training and push model.

## Quick start
1. Set environment variables:
   - `HF_USER` = your Hugging Face username
   - `HF_TOKEN` = your Hugging Face token
2. Run the notebook (`notebook.ipynb`) in Colab or Jupyter to register data, create train/test, train model, and push artifacts.
3. Push repository to GitHub and add GitHub Secrets: `HF_USER`, `HF_TOKEN`.
4. Create a Hugging Face Space (Streamlit) and push `app.py`, `requirements.txt`, and `best_model.joblib`.

## Notes
- The notebook contains step-by-step cells; run sequentially.
- The GitHub Actions workflow expects `train_script.py` to exist and `data/` to be present or to fetch data from the Hugging Face dataset.
