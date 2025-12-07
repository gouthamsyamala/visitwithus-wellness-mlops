"""
train_script.py

Script to run training and push the best model to Hugging Face model hub.
Intended to be executed by GitHub Actions or locally.

Environment variables required:
- HF_USER
- HF_TOKEN
"""

import os
import joblib
import shutil
import pandas as pd
import numpy as np
from huggingface_hub import HfApi, Repository
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score

# Paths
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_FILE = "best_model.joblib"
HF_MODEL_DIR = "hf_model_repo"

def load_data():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError("Train/test CSVs not found in data/. Run the notebook to create them.")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def build_pipeline(train):
    target = "ProdTaken"
    features = [c for c in train.columns if c != target]
    numeric_features = train[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in features if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    return pipeline, features

def train_and_select(train, test):
    pipeline, features = build_pipeline(train)
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_split": [2, 5]
    }
    gs = GridSearchCV(pipeline, param_grid, cv=4, scoring="roc_auc", n_jobs=-1, verbose=1)
    gs.fit(train[features], train["ProdTaken"])
    best = gs.best_estimator_
    y_proba = best.predict_proba(test[features])[:, 1]
    y_pred = best.predict(test[features])
    auc = roc_auc_score(test["ProdTaken"], y_proba)
    acc = accuracy_score(test["ProdTaken"], y_pred)
    print("Best params:", gs.best_params_)
    print("Test AUC:", auc, "Test Accuracy:", acc)
    joblib.dump(best, MODEL_FILE)
    return MODEL_FILE, gs.best_params_, {"test_auc": auc, "test_accuracy": acc}

def push_model_to_hf(hf_user, hf_token):
    model_repo_id = f"{hf_user}/visitwithus-wellness-model"
    api = HfApi()
    api.create_repo(repo_id=model_repo_id, repo_type="model", token=hf_token, exist_ok=True)
    if os.path.exists(HF_MODEL_DIR):
        shutil.rmtree(HF_MODEL_DIR)
    repo = Repository(local_dir=HF_MODEL_DIR, clone_from=model_repo_id, token=hf_token)
    shutil.copy(MODEL_FILE, os.path.join(HF_MODEL_DIR, MODEL_FILE))
    with open(os.path.join(HF_MODEL_DIR, "README.md"), "w") as f:
        f.write("# Visit with Us — Wellness Purchase Prediction Model\n\nRandomForest model (joblib).")
    repo.push_to_hub(commit_message="Add best_model.joblib")
    print(f"Pushed model to https://huggingface.co/{model_repo_id}")

def main():
    hf_user = os.environ.get("HF_USER")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_user or not hf_token:
        raise EnvironmentError("HF_USER and HF_TOKEN must be set as environment variables.")
    train, test = load_data()
    model_file, best_params, metrics = train_and_select(train, test)
    push_model_to_hf(hf_user, hf_token)
    print("Training complete. Model saved to", model_file)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
