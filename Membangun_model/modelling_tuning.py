import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from joblib import dump
import os

# Load dataset hasil preprocessing
data = pd.read_csv("preprocessing/preprocessed_dataset.csv")

# Pisahkan fitur dan target
X = data.drop("Personality", axis=1)
y = data["Personality"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Eksperimen_Kriteria2")



# === modelling_tuning.py (Dengan Tuning) ===
with mlflow.start_run(run_name="RandomForest_Tuned"):
    mlflow.autolog(disable=True)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    dump(best_model, "best_model_tuned.pkl")
    mlflow.sklearn.log_model(best_model, "model_tuned")
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })