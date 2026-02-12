#!/usr/bin/env python3
"""
Train baseline models on GlobalTechTalent_50k, evaluate, and generate a SHAP force plot
for a high-citation candidate predicted as non-migrant.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt


LEAKAGE_FEATURES = ["Visa_Approval_Date"]
TARGET = "Migration_Status"


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop leakage columns if present
    drop_cols = [c for c in LEAKAGE_FEATURES if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def build_preprocessor(df: pd.DataFrame):
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
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
        penalty="elasticnet",
        l1_ratio=0.5,
        solver="saga",
        max_iter=2000,
        n_jobs=-1,
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

    X_test_enc = preprocessor.transform(X_test)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_enc)

    # predictions to find candidate
    preds = model.predict(preprocessor.transform(X_test))
    candidate_idx = pick_candidate(X_test.reset_index(drop=True), preds)

    # TreeExplainer returns list for classification
    if isinstance(shap_values, list):
        sv = shap_values[1][candidate_idx]
        base_value = explainer.expected_value[1]
    else:
        sv = shap_values[candidate_idx]
        base_value = explainer.expected_value

    # Build force plot and save as PNG via matplotlib
    shap.initjs()
    force = shap.force_plot(base_value, sv, feature_names=feature_names, matplotlib=True, show=False)
    plt.title("SHAP Force Plot: High-Citation Candidate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()

    return {
        "candidate_index": int(candidate_idx),
        "force_plot_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train models and generate SHAP explanations.")
    parser.add_argument("--data", type=Path, default=Path("code/data/GlobalTechTalent_50k.csv"))
    parser.add_argument("--figdir", type=Path, default=Path("code/figures"))
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
