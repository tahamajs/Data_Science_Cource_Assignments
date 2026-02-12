#!/usr/bin/env python3
"""
Train baseline models on GlobalTechTalent_50k, evaluate, and generate a SHAP force plot
for a high-citation candidate predicted as non-migrant.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import os
import tempfile
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib

# Keep matplotlib cache writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt


LEAKAGE_FEATURES = ["Visa_Approval_Date"]
TARGET = "Migration_Status"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "GlobalTechTalent_50k.csv"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures"


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop leakage columns if present
    drop_cols = [c for c in LEAKAGE_FEATURES if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def build_preprocessor(df: pd.DataFrame):
    categorical = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if TARGET in categorical:
        categorical.remove(TARGET)
    numeric = [c for c in df.columns if c not in categorical + [TARGET]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    return preprocessor, categorical, numeric


def fit_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    preprocessor, categorical, numeric = build_preprocessor(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Logistic Regression with Elastic Net
    log_reg = LogisticRegression(
        # Newer sklearn versions infer elastic-net behavior from l1_ratio directly.
        l1_ratio=0.5,
        solver="saga",
        max_iter=2000,
    )

    log_pipeline = Pipeline(
        steps=[("preprocess", preprocessor), ("model", log_reg)]
    )

    log_pipeline.fit(X_train, y_train)
    y_pred = log_pipeline.predict(X_test)
    y_proba = log_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "classification_report": classification_report(y_test, y_pred, digits=3),
    }

    # Random Forest for tree-based SHAP explanations (no leakage columns)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    rf_pipeline = Pipeline(
        steps=[("preprocess", preprocessor), ("model", rf)]
    )

    rf_pipeline.fit(X_train, y_train)

    return {
        "log_pipeline": log_pipeline,
        "rf_pipeline": rf_pipeline,
        "metrics": metrics,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "X_test": X_test,
        "y_test": y_test,
    }


def pick_candidate(df: pd.DataFrame, y_pred: np.ndarray) -> int:
    # Choose a high-citation candidate predicted as non-migrant if available
    mask = (df["Research_Citations"] >= 2000) & (y_pred == 0)
    candidates = df[mask]
    if len(candidates) == 0:
        return df.sort_values("Research_Citations", ascending=False).index[0]
    return candidates.sort_values("Research_Citations", ascending=False).index[0]


def compute_shap_plot(rf_pipeline: Pipeline, X_test: pd.DataFrame, feature_names: list, output_path: Path):
    preprocessor = rf_pipeline.named_steps["preprocess"]
    model = rf_pipeline.named_steps["model"]

    # Limit to a manageable subset for SHAP speed
    subset = X_test.sample(n=min(500, len(X_test)), random_state=0).reset_index(drop=True)
    subset_enc = preprocessor.transform(subset)
    if hasattr(subset_enc, "toarray"):
        subset_enc = subset_enc.toarray()
    subset_enc = np.asarray(subset_enc, dtype=float)

    explainer = shap.TreeExplainer(model)

    preds = model.predict(subset_enc)
    candidate_idx = pick_candidate(subset, preds)

    # Compute SHAP values only for the selected candidate to keep runtime small
    shap_values = explainer.shap_values(subset_enc[candidate_idx : candidate_idx + 1])

    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1]).reshape(-1)
        base_value = explainer.expected_value[1]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            sv = arr[0, :, -1]
            base_value = np.asarray(explainer.expected_value).reshape(-1)[-1]
        elif arr.ndim == 2:
            sv = arr[0]
            base_value = np.asarray(explainer.expected_value).reshape(-1)[0]
        else:
            sv = arr.reshape(-1)
            base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    contributions = pd.Series(sv, index=feature_names)
    top = contributions.abs().sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    top.sort_values().plot(
        kind="barh",
        color=["#4c72b0" if v < 0 else "#dd8452" for v in top],
    )
    plt.axvline(0, color="k", linewidth=0.8)
    plt.title("Top SHAP Contributions (candidate with high citations)")
    plt.xlabel("Contribution to log-odds of migration")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    return {
        "candidate_index": int(candidate_idx),
        "force_plot_path": output_path,
        "base_value": float(base_value),
        "model_output_log_odds": float(base_value + contributions.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train models and generate SHAP explanations.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()

    df = load_data(args.data)
    artifacts = fit_models(df)

    # SHAP force plot
    shap_info = compute_shap_plot(
        artifacts["rf_pipeline"],
        artifacts["X_test"],
        artifacts["feature_names"],
        args.figdir / "shap_force_plot.png",
    )

    print("Elastic Net Logistic Regression Metrics:")
    print(artifacts["metrics"]["classification_report"])
    print(f"Accuracy: {artifacts['metrics']['accuracy']:.3f}")
    print(f"ROC-AUC: {artifacts['metrics']['roc_auc']:.3f}")
    print(f"SHAP force plot saved to: {shap_info['force_plot_path']}")
    print(f"Candidate index used: {shap_info['candidate_index']}")


if __name__ == "__main__":
    main()
