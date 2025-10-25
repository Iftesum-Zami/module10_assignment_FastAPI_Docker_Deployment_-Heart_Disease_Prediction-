# model/model_run.py
# Logistic Regression classifier

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).parent
DATA_PATH = HERE / "heart.csv"
MODEL_PATH = HERE / "heart_model.joblib"

def main():
    # --- Load ---
    df = pd.read_csv(DATA_PATH)

    # --- Split features/target ---
    target_col = "target"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)  # 0 = absence, 1 = presence

    # Identify feature types
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # --- Preprocessing ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # --- Model ---
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=None,          
        class_weight=None,
        solver="lbfgs"
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", clf),
    ])

    # --- Train/Validate ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    # --- Metrics ---
    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    # proba may not always be available, but LogisticRegression does expose it
    y_proba = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation ROC-AUC : {auc:.4f}")

    # --- Persist ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model pipeline saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()

print("---------------- Model trained and saved successfully -------------.")