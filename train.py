import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Load & preprocess ──────────────────────────────────────────────────────────
df = pd.read_csv("data/titanic.csv")

# Drop low-value columns
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Fill missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Encode categoricals
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── MLflow experiment ──────────────────────────────────────────────────────────
mlflow.set_experiment("titanic-survival-prediction")

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec  = recall_score(y_test, preds)
        f1   = f1_score(y_test, preds)

        # Log params
        mlflow.log_param("model", name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[{name}] Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")

print("\nDone! Run `mlflow ui` to view results.")