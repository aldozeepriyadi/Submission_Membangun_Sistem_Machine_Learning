import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from joblib import dump
import os
import dotenv

# Load data hasil preprocessing
data = pd.read_csv("Membangun_model/preprocessing/preprocessed_dataset.csv")

# Pisahkan fitur dan target
X = data.drop("Personality", axis=1)
y = data["Personality"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load environment variable dari .env
dotenv.load_dotenv()

# Set MLflow untuk DagsHub (bukan localhost)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
# === Mulai MLflow run ===
with mlflow.start_run(run_name="RandomForest_Default"):
    mlflow.autolog()
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Hitung metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    
    mlflow.sklearn.log_model(model, "model_default")
    
    # Logging manual
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })
