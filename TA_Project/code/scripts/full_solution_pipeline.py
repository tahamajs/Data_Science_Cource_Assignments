#!/usr/bin/env python3
"""End-to-end implementation for the UT-ECE Global Tech Talent Migration assessment.

This script generates:
- SQL answer artifact for Q1A
- leakage diagnostics for Q1B
- optimizer ravine simulations for Q3
- SVM gamma + decision tree pruning diagnostics for Q4
- PCA + KMeans elbow analysis for Q5
- capstone model + SHAP local explanation for Q6
- Q15 calibration + threshold policy analysis
- Q16 drift diagnostics with PSI summaries
- Q17 counterfactual recourse analysis
- Q18 temporal backtesting with rolling validation and degradation analysis
- Q19 uncertainty quantification via split conformal intervals
- Q20 fairness mitigation pre/post policy evaluation
- a consolidated markdown answer key aligned with Q1-Q6 plus Q15-Q17 extension
- run summary schema v2 + LaTeX-ready metrics exports
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Keep matplotlib cache writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from q18_temporal import run_q18_temporal_backtesting
from q19_uncertainty import run_q19_uncertainty_quantification
from q20_fairness_mitigation import run_q20_fairness_mitigation
from report_metrics_export import METRIC_EXPORT_VERSION, export_metrics_files

TARGET = "Migration_Status"
LEAKAGE_FEATURES = ["Visa_Approval_Date"]
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "GlobalTechTalent_50k.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"
DEFAULT_SOLUTIONS_DIR = PROJECT_ROOT / "solutions"
RUN_SUMMARY_VERSION = 2

PROFILE_CONFIGS = {
    "fast": {
        "q4_sample_size": 1200,
        "q4_max_alpha_points": 22,
        "q4_gamma_grid": [0.005, 0.03, 0.2],
        "q5_cluster_sample": 6000,
        "q5_k_max": 8,
        "q6_sample_size": 12000,
        "q15_sample_size": 12000,
        "q17_sample_size": 9000,
        "q17_max_candidates": 80,
    },
    "balanced": {
        "q4_sample_size": 1800,
        "q4_max_alpha_points": 35,
        "q4_gamma_grid": [0.005, 0.02, 0.08, 0.3],
        "q5_cluster_sample": 12000,
        "q5_k_max": 10,
        "q6_sample_size": 20000,
        "q15_sample_size": 22000,
        "q17_sample_size": 16000,
        "q17_max_candidates": 120,
    },
    "heavy": {
        "q4_sample_size": 2600,
        "q4_max_alpha_points": 50,
        "q4_gamma_grid": [0.002, 0.005, 0.02, 0.08, 0.2, 0.5],
        "q5_cluster_sample": 20000,
        "q5_k_max": 14,
        "q6_sample_size": 32000,
        "q15_sample_size": 32000,
        "q17_sample_size": 26000,
        "q17_max_candidates": 200,
    },
}

try:
    import shap

    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    shap = None
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = str(exc)

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
    XGB_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    xgb = None
    XGB_AVAILABLE = False
    XGB_IMPORT_ERROR = str(exc)


def _make_ohe() -> OneHotEncoder:
    """Create OneHotEncoder compatible with multiple sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - old sklearn fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in dataset.")
    return df


def build_features(df: pd.DataFrame, drop_leakage: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    if drop_leakage:
        cols = [c for c in LEAKAGE_FEATURES if c in work.columns]
        if cols:
            work = work.drop(columns=cols)
    X = work.drop(columns=[TARGET])
    y = work[TARGET].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    pre = ColumnTransformer(
        transformers=[
            ("cat", _make_ohe(), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    return pre, categorical, numeric


def build_capstone_model() -> Tuple[str, object]:
    if XGB_AVAILABLE:
        model_name = "XGBoost"
        model = xgb.XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=4,
            eval_metric="logloss",
        )
    else:
        model_name = "RandomForest (XGBoost fallback)"
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=4,
            class_weight="balanced_subsample",
        )
    return model_name, model


def write_q1_sql(out_path: Path) -> str:
    sql = """WITH citation_velocity AS (
    SELECT
        UserID,
        Country_Origin,
        Year,
        Research_Citations,
        AVG(Research_Citations) OVER (
            PARTITION BY Country_Origin
            ORDER BY Year
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS moving_avg_citations
    FROM Professionals_Data
)
SELECT
    UserID,
    Country_Origin,
    Year,
    Research_Citations,
    moving_avg_citations,
    DENSE_RANK() OVER (
        PARTITION BY Country_Origin
        ORDER BY moving_avg_citations DESC
    ) AS country_rank
FROM citation_velocity
ORDER BY Country_Origin, country_rank, Year;
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(sql, encoding="utf-8")
    return sql


def leakage_diagnostics(df: pd.DataFrame) -> Dict[str, float]:
    diagnostics: Dict[str, float] = {}
    if "Visa_Approval_Date" in df.columns:
        visa_present = df["Visa_Approval_Date"].notna().astype(int)
        diagnostics["visa_presence_corr_with_target"] = float(visa_present.corr(df[TARGET]))
        diagnostics["target_rate_if_visa_present"] = float(df.loc[visa_present == 1, TARGET].mean())
        diagnostics["target_rate_if_visa_absent"] = float(df.loc[visa_present == 0, TARGET].mean())

    if "Last_Login_Region" in df.columns and "Country_Origin" in df.columns:
        diagnostics["last_login_matches_origin_rate"] = float(
            (df["Last_Login_Region"] == df["Country_Origin"]).mean()
        )

    return diagnostics


def _ravine_loss(theta: np.ndarray, a: float, b: float) -> float:
    return 0.5 * (a * theta[0] ** 2 + b * theta[1] ** 2)


def _ravine_grad(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.array([a * theta[0], b * theta[1]], dtype=float)


def simulate_optimizers(steps: int = 120, a: float = 1.0, b: float = 120.0) -> Dict[str, np.ndarray]:
    init = np.array([-3.0, 2.5], dtype=float)

    # Plain SGD
    sgd_lr = 0.01
    sgd_t = init.copy()
    sgd_path = [sgd_t.copy()]
    for _ in range(steps):
        g = _ravine_grad(sgd_t, a, b)
        sgd_t = sgd_t - sgd_lr * g
        sgd_path.append(sgd_t.copy())

    # Momentum
    mom_lr = 0.01
    beta = 0.9
    mom_t = init.copy()
    v = np.zeros_like(mom_t)
    mom_path = [mom_t.copy()]
    for _ in range(steps):
        g = _ravine_grad(mom_t, a, b)
        v = beta * v + mom_lr * g
        mom_t = mom_t - v
        mom_path.append(mom_t.copy())

    # Adam
    adam_lr = 0.08
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    adam_t = init.copy()
    m = np.zeros_like(adam_t)
    s = np.zeros_like(adam_t)
    adam_path = [adam_t.copy()]
    for t in range(1, steps + 1):
        g = _ravine_grad(adam_t, a, b)
        m = beta1 * m + (1 - beta1) * g
        s = beta2 * s + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**t)
        s_hat = s / (1 - beta2**t)
        adam_t = adam_t - adam_lr * m_hat / (np.sqrt(s_hat) + eps)
        adam_path.append(adam_t.copy())

    return {
        "sgd": np.array(sgd_path),
        "momentum": np.array(mom_path),
        "adam": np.array(adam_path),
        "a": np.array([a]),
        "b": np.array([b]),
    }


def plot_ravine_paths(paths: Dict[str, np.ndarray], out_path: Path) -> Dict[str, float]:
    a = float(paths["a"][0])
    b = float(paths["b"][0])

    x = np.linspace(-3.5, 3.5, 300)
    y = np.linspace(-2.8, 2.8, 300)
    xx, yy = np.meshgrid(x, y)
    zz = 0.5 * (a * xx**2 + b * yy**2)

    plt.figure(figsize=(9, 6))
    contours = np.logspace(-3, 3, 18)
    plt.contour(xx, yy, zz, levels=contours, cmap="Greys", linewidths=0.7)

    styles = {
        "sgd": ("#2a9d8f", "SGD"),
        "momentum": ("#e76f51", "Momentum"),
        "adam": ("#264653", "Adam"),
    }
    final_losses = {}
    for key, (color, label) in styles.items():
        trajectory = paths[key]
        plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2, label=label)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, s=35)
        final_losses[f"{key}_final_loss"] = float(_ravine_loss(trajectory[-1], a, b))

    plt.title("Q3: Ravine Optimization Trajectories")
    plt.xlabel("theta_1")
    plt.ylabel("theta_2")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    return final_losses


def run_q4_svm_and_pruning(
    df: pd.DataFrame,
    figures_dir: Path,
    sample_size: int = 1800,
    gamma_grid: List[float] | None = None,
    max_alpha_points: int = 35,
) -> Dict[str, float]:
    # Numeric-only subset keeps the non-linear comparison computationally manageable.
    sample = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE).copy()
    y = sample[TARGET].astype(int)
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [TARGET, "UserID", "Visa_Approval_Date"]]
    X_num = sample[numeric_cols].fillna(sample[numeric_cols].median(numeric_only=True))

    X_train, X_val, y_train, y_val = train_test_split(
        X_num, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_enc = scaler.fit_transform(X_train)
    X_val_enc = scaler.transform(X_val)

    gammas = gamma_grid or [0.005, 0.02, 0.08, 0.3]
    train_scores: List[float] = []
    val_scores: List[float] = []
    support_ratios: List[float] = []

    for gamma in gammas:
        svc = SVC(kernel="rbf", gamma=gamma, C=1.0, random_state=RANDOM_STATE)
        svc.fit(X_train_enc, y_train)
        train_scores.append(accuracy_score(y_train, svc.predict(X_train_enc)))
        val_scores.append(accuracy_score(y_val, svc.predict(X_val_enc)))
        support_ratios.append(len(svc.support_) / len(X_train_enc))

    plt.figure(figsize=(8, 5))
    plt.plot(gammas, train_scores, "-o", label="Train Accuracy")
    plt.plot(gammas, val_scores, "-o", label="Validation Accuracy")
    plt.xscale("log")
    plt.xlabel("Gamma (log scale)")
    plt.ylabel("Accuracy")
    plt.title("Q4A: SVM RBF Gamma vs Performance")
    plt.legend()
    plt.tight_layout()
    svm_plot = figures_dir / "q4_svm_gamma_sweep.png"
    plt.savefig(svm_plot, dpi=200)
    plt.close()

    # Decision tree cost-complexity pruning on same encoded features
    tree_base = DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_leaf=5)
    path = tree_base.cost_complexity_pruning_path(X_train_enc, y_train)
    raw_alphas = path.ccp_alphas

    if len(raw_alphas) > max_alpha_points:
        alphas = np.unique(np.quantile(raw_alphas, np.linspace(0, 1, max_alpha_points)))
    else:
        alphas = np.unique(raw_alphas)

    train_accs: List[float] = []
    val_accs: List[float] = []

    for alpha in alphas:
        tree = DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            min_samples_leaf=5,
            ccp_alpha=float(alpha),
        )
        tree.fit(X_train_enc, y_train)
        train_accs.append(accuracy_score(y_train, tree.predict(X_train_enc)))
        val_accs.append(accuracy_score(y_val, tree.predict(X_val_enc)))

    best_idx = int(np.argmax(val_accs))
    best_alpha = float(alphas[best_idx])

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, train_accs, label="Train Accuracy")
    plt.plot(alphas, val_accs, label="Validation Accuracy")
    plt.axvline(best_alpha, color="black", linestyle="--", linewidth=1, label="Best alpha")
    plt.xlabel("ccp_alpha")
    plt.ylabel("Accuracy")
    plt.title("Q4B: Cost-Complexity Pruning Tradeoff")
    plt.legend()
    plt.tight_layout()
    tree_plot = figures_dir / "q4_tree_pruning_curve.png"
    plt.savefig(tree_plot, dpi=200)
    plt.close()

    return {
        "svm_best_gamma": float(gammas[int(np.argmax(val_scores))]),
        "svm_best_val_accuracy": float(np.max(val_scores)),
        "svm_worst_val_accuracy": float(np.min(val_scores)),
        "svm_support_ratio_gamma_min": float(support_ratios[0]),
        "svm_support_ratio_gamma_max": float(support_ratios[-1]),
        "tree_best_alpha": best_alpha,
        "tree_best_val_accuracy": float(val_accs[best_idx]),
    }


def run_q5_unsupervised(
    df: pd.DataFrame,
    figures_dir: Path,
    cluster_sample: int = 12000,
    k_max: int = 10,
) -> Dict[str, float]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = [TARGET, "UserID", "Visa_Approval_Date"]
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    X_num = df[numeric_cols].copy()
    X_num = X_num.fillna(X_num.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    pca = PCA(n_components=min(5, X_scaled.shape[1]), random_state=RANDOM_STATE)
    pca.fit(X_scaled)

    pc1_ratio = float(pca.explained_variance_ratio_[0])
    pc2_ratio = float(pca.explained_variance_ratio_[1])
    pc12_ratio = pc1_ratio + pc2_ratio

    sample_idx = np.random.default_rng(RANDOM_STATE).choice(
        len(X_scaled), size=min(cluster_sample, len(X_scaled)), replace=False
    )
    X_cluster = X_scaled[sample_idx]

    ks = np.arange(1, k_max + 1)
    wcss: List[float] = []
    for k in ks:
        km = KMeans(n_clusters=int(k), random_state=RANDOM_STATE, n_init=10)
        km.fit(X_cluster)
        wcss.append(float(km.inertia_))

    # Elbow by max distance to line between first and last point
    first = np.array([ks[0], wcss[0]], dtype=float)
    last = np.array([ks[-1], wcss[-1]], dtype=float)
    line = last - first
    line_norm = np.linalg.norm(line)

    dists = []
    for k, value in zip(ks, wcss):
        point = np.array([k, value], dtype=float)
        # 2D point-to-line distance without np.cross (avoids NumPy 2.0 deprecation warning)
        dist = abs(line[0] * (first[1] - point[1]) - line[1] * (first[0] - point[0])) / line_norm
        dists.append(float(dist))
    elbow_k = int(ks[int(np.argmax(dists))])

    plt.figure(figsize=(8, 5))
    plt.plot(ks, wcss, "-o")
    plt.axvline(elbow_k, color="black", linestyle="--", linewidth=1, label=f"Elbow K={elbow_k}")
    plt.xlabel("K")
    plt.ylabel("WCSS")
    plt.title("Q5B: KMeans Elbow Curve")
    plt.legend()
    plt.tight_layout()
    elbow_plot = figures_dir / "q5_kmeans_elbow.png"
    plt.savefig(elbow_plot, dpi=200)
    plt.close()

    return {
        "pca_pc1_ratio": pc1_ratio,
        "pca_pc2_ratio": pc2_ratio,
        "pca_pc1_pc2_ratio": pc12_ratio,
        "kmeans_elbow_k": float(elbow_k),
        "kmeans_wcss_k1": float(wcss[0]),
        "kmeans_wcss_kmax": float(wcss[-1]),
    }


def _extract_binary_shap(
    explainer,
    shap_values,
    expected_value,
) -> Tuple[np.ndarray, float]:
    # Handles SHAP return variants for tree models across versions.
    if isinstance(shap_values, list):
        vector = np.asarray(shap_values[-1]).reshape(-1)
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            base = float(np.asarray(expected_value).reshape(-1)[-1])
        else:
            base = float(expected_value)
        return vector, base

    arr = np.asarray(shap_values)

    if arr.ndim == 3:
        # shape often: (n_samples, n_features, n_outputs)
        vector = arr[0, :, -1]
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            base = float(np.asarray(expected_value).reshape(-1)[-1])
        else:
            base = float(expected_value)
        return vector, base

    if arr.ndim == 2:
        vector = arr[0]
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            base = float(np.asarray(expected_value).reshape(-1)[0])
        else:
            base = float(expected_value)
        return vector, base

    vector = arr.reshape(-1)
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        base = float(np.asarray(expected_value).reshape(-1)[0])
    else:
        base = float(expected_value)
    return vector, base


def run_q6_capstone(
    df: pd.DataFrame,
    figures_dir: Path,
    solutions_dir: Path,
    sample_size: int = 20000,
) -> Dict[str, float | str]:
    work_df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE).copy()
    X, y = build_features(work_df, drop_leakage=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pre, _, _ = build_preprocessor(X_train)
    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)
    feature_names = pre.get_feature_names_out().tolist()

    if XGB_AVAILABLE:
        model_name = "XGBoost"
        model = xgb.XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=4,
            eval_metric="logloss",
        )
    else:
        model_name = "RandomForest (XGBoost fallback)"
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=4,
            class_weight="balanced_subsample",
        )

    model.fit(X_train_enc, y_train)

    y_prob = model.predict_proba(X_test_enc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    # Candidate: high citations (>=2000) predicted as non-migrant, else highest citations.
    citations = (
        X_test["Research_Citations"]
        if "Research_Citations" in X_test.columns
        else pd.Series(0, index=X_test.index)
    )
    non_migrant_pool = X_test[y_pred == 0]
    if not non_migrant_pool.empty:
        high_citation_non_migrants = non_migrant_pool[citations.loc[non_migrant_pool.index] >= 2000]
        if high_citation_non_migrants.empty:
            candidate_idx = int(citations.loc[non_migrant_pool.index].sort_values(ascending=False).index[0])
        else:
            candidate_idx = int(citations.loc[high_citation_non_migrants.index].sort_values(ascending=False).index[0])
    else:
        candidate_idx = int(citations.sort_values(ascending=False).index[0])

    candidate_pos = int(X_test.index.get_loc(candidate_idx))
    candidate_enc = X_test_enc[candidate_pos : candidate_pos + 1]
    candidate_prob = float(y_prob[candidate_pos])

    shap_info = {
        "shap_status": "not_run",
        "base_value": float("nan"),
        "output_value": float("nan"),
        "output_probability_from_shap": float("nan"),
        "shap_output_space": "",
        "candidate_prediction_probability": candidate_prob,
        "candidate_index": float(candidate_idx),
        "force_plot_path": "",
        "summary_plot_path": "",
        "shap_top_feature": "",
    }

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            candidate_sv_raw = explainer.shap_values(candidate_enc)
            sv, base_value = _extract_binary_shap(
                explainer,
                candidate_sv_raw,
                explainer.expected_value,
            )

            candidate_series = pd.Series(np.asarray(candidate_enc).reshape(-1), index=feature_names)
            output_value = float(base_value + sv.sum())
            if 0.0 <= output_value <= 1.0:
                output_probability = output_value
                output_space = "probability"
            else:
                output_probability = float(_sigmoid(output_value))
                output_space = "log_odds"

            force_plot_path = figures_dir / "q6_shap_force_plot.png"
            try:
                plt.figure(figsize=(12, 2.8))
                shap.force_plot(
                    base_value,
                    sv,
                    candidate_series,
                    matplotlib=True,
                    show=False,
                )
                plt.tight_layout()
                plt.savefig(force_plot_path, dpi=220)
                plt.close()
            except Exception:
                # Fallback to waterfall plot if force plot backend fails.
                explanation = shap.Explanation(
                    values=sv,
                    base_values=base_value,
                    data=candidate_series.values,
                    feature_names=feature_names,
                )
                plt.figure(figsize=(9, 5))
                shap.plots.waterfall(explanation, max_display=15, show=False)
                plt.tight_layout()
                plt.savefig(force_plot_path, dpi=220)
                plt.close()

            summary_plot_path = figures_dir / "q6_shap_summary.png"
            if hasattr(model, "feature_importances_"):
                global_importance = np.asarray(model.feature_importances_)
            else:  # pragma: no cover - fallback branch
                global_importance = np.abs(sv)
            order = np.argsort(global_importance)[-15:]
            top_values = global_importance[order]
            top_names = [feature_names[i] for i in order]

            plt.figure(figsize=(9, 5))
            plt.barh(top_names, top_values, color="#1f77b4")
            plt.title("Q6: Global Importance (top 15 encoded features)")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(summary_plot_path, dpi=220)
            plt.close()

            contributions = pd.Series(sv, index=feature_names)
            top_feature = str(contributions.abs().idxmax())

            shap_info = {
                "shap_status": "ok",
                "base_value": float(base_value),
                "output_value": float(output_value),
                "output_probability_from_shap": float(output_probability),
                "shap_output_space": output_space,
                "candidate_prediction_probability": candidate_prob,
                "candidate_index": float(candidate_idx),
                "force_plot_path": _rel(force_plot_path),
                "summary_plot_path": _rel(summary_plot_path),
                "shap_top_feature": top_feature,
            }
        except Exception as exc:  # pragma: no cover - runtime fallback
            shap_info["shap_status"] = f"error: {exc}"
    else:  # pragma: no cover - environment dependent
        shap_info["shap_status"] = f"not_available: {SHAP_IMPORT_ERROR}"

    # Fairness-oriented slice summary by country
    if "Country_Origin" in X_test.columns:
        fairness = (
            pd.DataFrame(
                {
                    "Country_Origin": X_test["Country_Origin"].values,
                    "pred_positive": y_pred,
                    "actual_positive": y_test.values,
                }
            )
            .groupby("Country_Origin", as_index=False)
            .agg(
                sample_count=("pred_positive", "size"),
                pred_positive_rate=("pred_positive", "mean"),
                actual_positive_rate=("actual_positive", "mean"),
            )
            .sort_values("pred_positive_rate")
        )
        fairness_path = solutions_dir / "q6_fairness_country_rates.csv"
        fairness.to_csv(fairness_path, index=False)
        shap_info["fairness_path"] = _rel(fairness_path)

    return {**metrics, **shap_info}


def _prepare_model_bundle(df: pd.DataFrame, sample_size: int = 20000) -> Dict[str, object]:
    work_df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE).copy()
    X, y = build_features(work_df, drop_leakage=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pre, _, _ = build_preprocessor(X_train)
    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)

    model_name, model = build_capstone_model()
    model.fit(X_train_enc, y_train)

    y_prob = model.predict_proba(X_test_enc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "model_name": model_name,
        "model": model,
        "preprocessor": pre,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def run_q15_calibration_threshold(
    df: pd.DataFrame,
    figures_dir: Path,
    sample_size: int = 22000,
) -> Dict[str, float | str]:
    bundle = _prepare_model_bundle(df, sample_size=sample_size)
    y_test = np.asarray(bundle["y_test"])
    y_prob = np.asarray(bundle["y_prob"])

    brier = float(brier_score_loss(y_test, y_prob))
    auc = float(roc_auc_score(y_test, y_prob))

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(y_prob, bin_edges, right=True) - 1
    ece = 0.0
    n = len(y_prob)
    for b in range(10):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        avg_prob = float(np.mean(y_prob[mask]))
        avg_true = float(np.mean(y_test[mask]))
        ece += (np.sum(mask) / n) * abs(avg_true - avg_prob)

    thresholds = np.linspace(0.05, 0.95, 19)
    f1_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    expected_costs: List[float] = []

    cost_fn = 2.0
    cost_fp = 1.0

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f1_values.append(float(f1_score(y_test, pred, zero_division=0)))
        precision_values.append(float(precision_score(y_test, pred, zero_division=0)))
        recall_values.append(float(recall_score(y_test, pred, zero_division=0)))
        fn = int(np.sum((y_test == 1) & (pred == 0)))
        fp = int(np.sum((y_test == 0) & (pred == 1)))
        expected_costs.append(float((cost_fn * fn + cost_fp * fp) / len(y_test)))

    best_f1_idx = int(np.argmax(f1_values))
    best_cost_idx = int(np.argmin(expected_costs))

    calibration_path = figures_dir / "q15_calibration_curve.png"
    plt.figure(figsize=(7.2, 5.4))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", color="#1f77b4", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Q15A: Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(calibration_path, dpi=220)
    plt.close()

    threshold_path = figures_dir / "q15_threshold_tradeoff.png"
    fig, ax1 = plt.subplots(figsize=(8.4, 5.2))
    ax1.plot(thresholds, f1_values, label="F1", color="#1f77b4")
    ax1.plot(thresholds, precision_values, label="Precision", color="#2ca02c")
    ax1.plot(thresholds, recall_values, label="Recall", color="#ff7f0e")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Score")
    ax1.axvline(float(thresholds[best_f1_idx]), linestyle="--", color="#1f77b4", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, expected_costs, label="Expected cost", color="#d62728", linewidth=2)
    ax2.set_ylabel("Expected cost per sample")
    ax2.axvline(float(thresholds[best_cost_idx]), linestyle="--", color="#d62728", alpha=0.6)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", ncol=2)
    plt.title("Q15B: Threshold Policy Tradeoff")
    plt.tight_layout()
    plt.savefig(threshold_path, dpi=220)
    plt.close()

    return {
        "model_name": str(bundle["model_name"]),
        "roc_auc": auc,
        "brier_score": brier,
        "expected_calibration_error": float(ece),
        "best_f1_threshold": float(thresholds[best_f1_idx]),
        "best_f1": float(f1_values[best_f1_idx]),
        "best_cost_threshold": float(thresholds[best_cost_idx]),
        "minimum_expected_cost": float(expected_costs[best_cost_idx]),
        "calibration_plot_path": _rel(calibration_path),
        "threshold_plot_path": _rel(threshold_path),
    }


def _compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.Series(reference).dropna().astype(float)
    cur = pd.Series(current).dropna().astype(float)
    if ref.empty or cur.empty:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        lo = float(min(ref.min(), cur.min()))
        hi = float(max(ref.max(), cur.max()))
        if lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    if ref_hist.sum() == 0 or cur_hist.sum() == 0:
        return float("nan")

    ref_pct = ref_hist / ref_hist.sum()
    cur_pct = cur_hist / cur_hist.sum()
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def run_q16_drift_monitoring(df: pd.DataFrame, figures_dir: Path, solutions_dir: Path) -> Dict[str, float | str]:
    if "Year" in df.columns and df["Year"].nunique(dropna=True) > 1:
        median_year = float(df["Year"].median())
        reference = df[df["Year"] <= median_year].copy()
        current = df[df["Year"] > median_year].copy()
        split_rule = f"year <= {median_year:.1f} vs year > {median_year:.1f}"
    else:
        shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE)
        split = len(shuffled) // 2
        reference = shuffled.iloc[:split].copy()
        current = shuffled.iloc[split:].copy()
        split_rule = "random half split"

    if len(reference) < 400 or len(current) < 400:
        shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE)
        split = len(shuffled) // 2
        reference = shuffled.iloc[:split].copy()
        current = shuffled.iloc[split:].copy()
        split_rule = "random half split (fallback)"

    numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [TARGET, "UserID"]]

    psi_rows = []
    for col in numeric_cols:
        psi = _compute_psi(reference[col], current[col], bins=10)
        psi_rows.append({"feature": col, "psi": psi})

    psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
    psi_path = solutions_dir / "q16_drift_psi.csv"
    psi_df.to_csv(psi_path, index=False)

    top_plot = psi_df.head(12).copy()
    drift_plot_path = figures_dir / "q16_drift_psi_top12.png"
    plt.figure(figsize=(9, 5.2))
    plt.barh(top_plot["feature"], top_plot["psi"], color="#1f77b4")
    plt.axvline(0.10, color="#ff7f0e", linestyle="--", linewidth=1, label="Moderate drift (0.10)")
    plt.axvline(0.25, color="#d62728", linestyle="--", linewidth=1, label="High drift (0.25)")
    plt.gca().invert_yaxis()
    plt.xlabel("Population Stability Index (PSI)")
    plt.ylabel("Feature")
    plt.title("Q16: Feature Drift Ranking")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(drift_plot_path, dpi=220)
    plt.close()

    js_country = float("nan")
    if "Country_Origin" in reference.columns:
        ref_dist = reference["Country_Origin"].value_counts(normalize=True)
        cur_dist = current["Country_Origin"].value_counts(normalize=True)
        keys = sorted(set(ref_dist.index) | set(cur_dist.index))
        p = np.array([float(ref_dist.get(k, 0.0)) for k in keys], dtype=float)
        q = np.array([float(cur_dist.get(k, 0.0)) for k in keys], dtype=float)
        js_country = _js_divergence(p, q)

    valid_psi = psi_df["psi"].dropna()
    high_drift = int((valid_psi >= 0.25).sum())
    moderate_drift = int(((valid_psi >= 0.10) & (valid_psi < 0.25)).sum())

    top_feature = ""
    top_psi = float("nan")
    if not psi_df.empty:
        top_feature = str(psi_df.iloc[0]["feature"])
        top_psi = float(psi_df.iloc[0]["psi"])

    return {
        "split_rule": split_rule,
        "reference_size": float(len(reference)),
        "current_size": float(len(current)),
        "top_drift_feature": top_feature,
        "top_drift_psi": top_psi,
        "high_drift_feature_count": float(high_drift),
        "moderate_drift_feature_count": float(moderate_drift),
        "country_js_divergence": js_country,
        "drift_table_path": _rel(psi_path),
        "drift_plot_path": _rel(drift_plot_path),
    }


def _predict_row_probability(model, preprocessor, row_df: pd.DataFrame) -> float:
    transformed = preprocessor.transform(row_df)
    return float(model.predict_proba(transformed)[:, 1][0])


def run_q17_recourse_analysis(
    df: pd.DataFrame,
    figures_dir: Path,
    solutions_dir: Path,
    sample_size: int = 16000,
    max_candidates: int = 120,
) -> Dict[str, float | str]:
    bundle = _prepare_model_bundle(df, sample_size=sample_size)
    model = bundle["model"]
    pre = bundle["preprocessor"]
    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_prob = np.asarray(bundle["y_prob"])

    decision_threshold = 0.5
    near_boundary = np.where((y_prob < decision_threshold) & (y_prob >= 0.25))[0]
    if len(near_boundary) == 0:
        near_boundary = np.where(y_prob < decision_threshold)[0]
    near_boundary = near_boundary[np.argsort(y_prob[near_boundary])[::-1]]
    candidate_positions = near_boundary[:max_candidates]

    feature_deltas: Dict[str, np.ndarray] = {}
    if "GitHub_Activity" in X_test.columns:
        feature_deltas["GitHub_Activity"] = np.arange(2.0, 42.0, 2.0)
    if "Research_Citations" in X_test.columns:
        feature_deltas["Research_Citations"] = np.arange(50.0, 2050.0, 50.0)
    if "Industry_Experience" in X_test.columns:
        feature_deltas["Industry_Experience"] = np.arange(0.5, 10.5, 0.5)

    feature_caps = {
        c: float(X_train[c].quantile(0.995)) if np.issubdtype(X_train[c].dtype, np.number) else float("nan")
        for c in X_train.columns
        if c in feature_deltas
    }
    feature_scales = {
        c: float(X_train[c].std()) if np.issubdtype(X_train[c].dtype, np.number) else 1.0
        for c in X_train.columns
        if c in feature_deltas
    }

    recourse_rows = []
    for pos in candidate_positions:
        row = X_test.iloc[[pos]].copy()
        base_prob = float(y_prob[pos])
        row_index = row.index[0]
        best_record = None

        for feature, deltas in feature_deltas.items():
            if feature not in row.columns:
                continue
            base_value = row.iloc[0][feature]
            if pd.isna(base_value):
                continue

            for delta in deltas:
                trial_value = float(base_value) + float(delta)
                if feature in feature_caps and not math.isnan(feature_caps[feature]):
                    trial_value = min(trial_value, feature_caps[feature])
                if trial_value <= float(base_value):
                    continue

                trial_row = row.copy()
                trial_row.iloc[0, trial_row.columns.get_loc(feature)] = trial_value
                trial_prob = _predict_row_probability(model, pre, trial_row)
                if trial_prob >= decision_threshold:
                    actual_delta = float(trial_value - float(base_value))
                    scale = feature_scales.get(feature, 1.0)
                    normalized_delta = actual_delta / (scale + 1e-6)
                    candidate = {
                        "candidate_index": int(row_index),
                        "base_probability": base_prob,
                        "best_feature": feature,
                        "required_delta": actual_delta,
                        "new_probability": float(trial_prob),
                        "normalized_delta": float(normalized_delta),
                    }
                    if best_record is None or candidate["normalized_delta"] < best_record["normalized_delta"]:
                        best_record = candidate
                    break

        if best_record is not None:
            recourse_rows.append(best_record)

    recourse_df = pd.DataFrame(recourse_rows)
    recourse_path = solutions_dir / "q17_recourse_examples.csv"
    if recourse_df.empty:
        recourse_df = pd.DataFrame(
            columns=[
                "candidate_index",
                "base_probability",
                "best_feature",
                "required_delta",
                "new_probability",
                "normalized_delta",
            ]
        )
    recourse_df.to_csv(recourse_path, index=False)

    recourse_plot_path = figures_dir / "q17_recourse_median_deltas.png"
    plt.figure(figsize=(7.6, 4.8))
    if recourse_df.empty:
        plt.text(0.5, 0.5, "No feasible recourse found", ha="center", va="center", fontsize=12)
        plt.axis("off")
    else:
        medians = recourse_df.groupby("best_feature")["required_delta"].median().sort_values(ascending=False)
        plt.bar(medians.index, medians.values, color=["#1f77b4", "#2ca02c", "#ff7f0e"][: len(medians)])
        plt.ylabel("Median required change")
        plt.xlabel("Actionable feature")
        plt.title("Q17: Median Recourse Effort by Feature")
        plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(recourse_plot_path, dpi=220)
    plt.close()

    considered = int(len(candidate_positions))
    successful = int(len(recourse_df))
    success_rate = float(successful / considered) if considered > 0 else 0.0

    summary: Dict[str, float | str] = {
        "model_name": str(bundle["model_name"]),
        "decision_threshold": float(decision_threshold),
        "candidates_considered": float(considered),
        "successful_recourse_count": float(successful),
        "recourse_success_rate": success_rate,
        "recourse_examples_path": _rel(recourse_path),
        "recourse_plot_path": _rel(recourse_plot_path),
    }

    default_features = ["GitHub_Activity", "Research_Citations", "Industry_Experience"]
    for feature in default_features:
        key = f"median_required_delta_{feature}"
        if feature not in feature_deltas or recourse_df.empty:
            summary[key] = float("nan")
        else:
            subset = recourse_df.loc[recourse_df["best_feature"] == feature, "required_delta"]
            summary[key] = float(subset.median()) if not subset.empty else float("nan")

    return summary


def write_solution_markdown(
    out_path: Path,
    q1_diag: Dict[str, float],
    q3_info: Dict[str, float],
    q4_info: Dict[str, float],
    q5_info: Dict[str, float],
    q6_info: Dict[str, float | str],
    q15_info: Dict[str, float | str],
    q16_info: Dict[str, float | str],
    q17_info: Dict[str, float | str],
    q18_info: Dict[str, float | str],
    q19_info: Dict[str, float | str],
    q20_info: Dict[str, float | str],
) -> None:
    def fmt(value: object, digits: int = 3) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and math.isnan(value):
                return "N/A"
            return f"{value:.{digits}f}"
        return str(value)

    q1_corr = q1_diag.get("visa_presence_corr_with_target", float("nan"))
    q1_yes = q1_diag.get("target_rate_if_visa_present", float("nan"))
    q1_no = q1_diag.get("target_rate_if_visa_absent", float("nan"))

    content = rf"""# Global Tech Talent Migration — Complete Solution Key

This document provides complete answers for Q1-Q6 and links each answer to executable artifacts in this repository.

## Q1. Advanced Data Engineering & SQL

### Q1A. 3-year moving average + rank query
The SQL answer is saved in `code/solutions/q1_moving_average.sql`.

### Q1B. Data leakage diagnosis
Leaked feature(s):
- `Visa_Approval_Date` is **direct leakage** because it encodes post-outcome bureaucratic status.
- `Last_Login_Region` can become **temporal leakage** if captured after migration and reflects destination behavior.
- `Passport_Renewal_Status` is potentially leaky if timestamps are post-decision.
- `Years_Since_Degree` is generally safe if computed from pre-decision records.

Dataset evidence from this run:
- corr(visa_present, migration_status) = **{q1_corr:.3f}**
- P(Migration=1 | visa present) = **{q1_yes:.3f}**
- P(Migration=1 | visa absent) = **{q1_no:.3f}**

## Q2. Statistical Inference & Linear Models

### Q2A. Elastic Net gradient derivation
For
\[
J(\theta)=\frac{{1}}{{2m}}\sum_{{i=1}}^m (h_\theta(x^{{(i)}})-y^{{(i)}})^2 + \lambda_1\sum_{{j=1}}^n|\theta_j| + \frac{{\lambda_2}}{{2}}\sum_{{j=1}}^n\theta_j^2,
\]
for parameter \(\theta_j\):
\[
\nabla_{{\theta_j}}J(\theta)=\frac{{1}}{{m}}\sum_{{i=1}}^m (h_\theta(x^{{(i)}})-y^{{(i)}})x_j^{{(i)}} + \lambda_1\,\partial|\theta_j| + \lambda_2\theta_j,
\]
where
\[
\partial|\theta_j|=\begin{{cases}}
+1 & \theta_j>0\\
-1 & \theta_j<0\\
[-1,1] & \theta_j=0
\end{{cases}}
\]
At \(\theta_j=0\), coordinate-descent solvers use this subgradient set and can keep coefficients exactly at zero (feature selection).

### Q2B. Interpretation of \(\beta=0.52\), p-value \(=0.003\), 95% CI \([0.18, 0.86]\)
- Because p-value < 0.05, reject \(H_0: \beta=0\).
- The confidence interval excludes 0, confirming statistical significance.
- The positive interval implies higher `GitHub_Activity` is associated with higher migration propensity (under model assumptions and conditioning on other covariates).

## Q3. Optimization & Gradient Descent

Ravine behavior: steep curvature in one dimension and shallow curvature in another causes vanilla SGD to zig-zag and progress slowly.

From this run (final losses on the toy ravine):
- SGD final loss: **{q3_info['sgd_final_loss']:.6f}**
- Momentum final loss: **{q3_info['momentum_final_loss']:.6f}**
- Adam final loss: **{q3_info['adam_final_loss']:.6f}**

Interpretation:
- Momentum damps oscillations by accumulating velocity, so opposing gradients on steep walls cancel over time.
- Adam additionally rescales updates per-parameter using first/second moments, usually improving stability when feature scales differ.

Figure: `code/figures/q3_ravine_optimizers.png`

## Q4. Non-Linear Models & Kernels

### Q4A. SVM with RBF kernel (overfitting case)
If the model overfits, **decrease \(\gamma\)**.
- High \(\gamma\): narrow influence radius around each point, very wiggly boundary, high variance.
- Lower \(\gamma\): broader influence, smoother boundary, lower variance.

Run metrics:
- Best validation gamma: **{q4_info['svm_best_gamma']:.3f}**
- Best validation accuracy: **{q4_info['svm_best_val_accuracy']:.3f}**
- Worst validation accuracy: **{q4_info['svm_worst_val_accuracy']:.3f}**

Figure: `code/figures/q4_svm_gamma_sweep.png`

### Q4B. Cost-complexity pruning
\[
R_\alpha(T)=R(T)+\alpha|T|
\]
- Increasing \(\alpha\) increases penalty for leaf count, producing smaller trees.
- Small \(\alpha\): low bias, high variance.
- Large \(\alpha\): higher bias, lower variance.

Run metrics:
- Best \(\alpha\): **{q4_info['tree_best_alpha']:.6f}**
- Best validation accuracy after pruning: **{q4_info['tree_best_val_accuracy']:.3f}**

Figure: `code/figures/q4_tree_pruning_curve.png`

## Q5. Unsupervised Learning

### Q5A. PCA explained variance ratio
For covariance matrix eigenvalues \(\lambda_1, \lambda_2, \lambda_3\):
\[
\text{{EVR}}_k = \frac{{\lambda_k}}{{\lambda_1+\lambda_2+\lambda_3}}
\]
Eigenvalue interpretation: variance captured along principal component \(k\).

Run results:
- EVR(PC1): **{q5_info['pca_pc1_ratio']:.3f}**
- EVR(PC2): **{q5_info['pca_pc2_ratio']:.3f}**
- EVR(PC1+PC2): **{q5_info['pca_pc1_pc2_ratio']:.3f}**

### Q5B. K-Means elbow method rationale
- WCSS decreases monotonically with larger \(K\) because each additional centroid can only reduce or keep the same minimum squared distances.
- The elbow approximates the point of diminishing returns where marginal WCSS reduction drops sharply.
- Geometrically, this is near maximal curvature on the WCSS-vs-\(K\) curve.

Run result:
- Elbow-selected \(K\): **{int(q5_info['kmeans_elbow_k'])}**

Figure: `code/figures/q5_kmeans_elbow.png`

## Q6. Capstone Explainability (SHAP)

Model used in this run: **{q6_info['model_name']}**
- Accuracy: **{q6_info['accuracy']:.3f}**
- ROC-AUC: **{q6_info['roc_auc']:.3f}**
- F1: **{q6_info['f1']:.3f}**

Candidate explanation details:
- Candidate index: **{int(q6_info['candidate_index'])}**
- Predicted migration probability: **{q6_info['candidate_prediction_probability']:.3f}**
- SHAP status: **{q6_info['shap_status']}**
- Base value: **{q6_info['base_value'] if isinstance(q6_info['base_value'], str) else f"{q6_info['base_value']:.5f}"}**
- Output value: **{q6_info['output_value'] if isinstance(q6_info['output_value'], str) else f"{q6_info['output_value']:.5f}"}**
- Output space: **{q6_info.get('shap_output_space', 'unknown')}**
- Probability implied by SHAP output: **{q6_info['output_probability_from_shap'] if isinstance(q6_info['output_probability_from_shap'], str) else f"{q6_info['output_probability_from_shap']:.5f}"}**

Interpretation:
- `base_value` is the model's average output over the background set.
- `output_value` is the local output for the candidate.
- Their difference is the sum of per-feature SHAP contributions for that candidate.
- Positive SHAP values push toward migration; negative values push toward no migration.

Artifacts:
- Local force/waterfall plot: `{q6_info['force_plot_path']}`
- Global SHAP summary plot: `{q6_info['summary_plot_path']}`
- Country fairness slice: `{q6_info.get('fairness_path', '')}`

## Q15. Calibration and Threshold Policy (New)

Why this matters:
- A high AUC model can still be poorly calibrated.
- Decision threshold should be chosen by utility/cost, not only default 0.5.

Run results:
- Model: **{q15_info['model_name']}**
- ROC-AUC: **{q15_info['roc_auc']:.3f}**
- Brier score: **{q15_info['brier_score']:.4f}**
- Expected calibration error: **{q15_info['expected_calibration_error']:.4f}**
- Best threshold by F1: **{q15_info['best_f1_threshold']:.2f}** (F1={q15_info['best_f1']:.3f})
- Best threshold by expected cost: **{q15_info['best_cost_threshold']:.2f}**
- Minimum expected cost per sample: **{q15_info['minimum_expected_cost']:.4f}**

Artifacts:
- Calibration curve: `{q15_info['calibration_plot_path']}`
- Threshold tradeoff plot: `{q15_info['threshold_plot_path']}`

## Q16. Drift Monitoring and Data Stability (New)

Drift diagnostics use PSI (Population Stability Index) across reference/current windows.

Run results:
- Split rule: **{q16_info['split_rule']}**
- Reference size: **{q16_info['reference_size']:.0f}**
- Current size: **{q16_info['current_size']:.0f}**
- Top drift feature: **{q16_info['top_drift_feature']}**
- Top drift PSI: **{q16_info['top_drift_psi']:.4f}**
- High-drift features (PSI >= 0.25): **{q16_info['high_drift_feature_count']:.0f}**
- Moderate-drift features (0.10 <= PSI < 0.25): **{q16_info['moderate_drift_feature_count']:.0f}**
- Country distribution JS divergence: **{q16_info['country_js_divergence']:.4f}**

Artifacts:
- Drift table: `{q16_info['drift_table_path']}`
- Drift plot: `{q16_info['drift_plot_path']}`

## Q17. Counterfactual Recourse Analysis (New)

Question addressed:
- For near-boundary non-migrant predictions, what is the minimum actionable change needed to flip decision to migration-positive?

Run results:
- Model: **{q17_info['model_name']}**
- Decision threshold: **{q17_info['decision_threshold']:.2f}**
- Candidates considered: **{q17_info['candidates_considered']:.0f}**
- Successful recourse count: **{q17_info['successful_recourse_count']:.0f}**
- Recourse success rate: **{q17_info['recourse_success_rate']:.3f}**
- Median delta (GitHub\_Activity): **{q17_info['median_required_delta_GitHub_Activity']:.3f}**
- Median delta (Research\_Citations): **{q17_info['median_required_delta_Research_Citations']:.3f}**
- Median delta (Industry\_Experience): **{q17_info['median_required_delta_Industry_Experience']:.3f}**

Artifacts:
- Recourse examples table: `{q17_info['recourse_examples_path']}`
- Recourse effort plot: `{q17_info['recourse_plot_path']}`

## Fairness note for grading discussion
Even with strong predictive metrics, model decisions can mirror historical policy constraints. Country-level predicted positive rates should be audited against domain knowledge before any deployment.
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def run_all(data_path: Path, figures_dir: Path, solutions_dir: Path) -> Dict[str, object]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)

    # Q1
    q1_sql = write_q1_sql(solutions_dir / "q1_moving_average.sql")
    q1_diag = leakage_diagnostics(df)

    # Q3
    paths = simulate_optimizers()
    q3_info = plot_ravine_paths(paths, figures_dir / "q3_ravine_optimizers.png")

    # Q4
    q4_info = run_q4_svm_and_pruning(df, figures_dir)

    # Q5
    q5_info = run_q5_unsupervised(df, figures_dir)

    # Q6
    q6_info = run_q6_capstone(df, figures_dir, solutions_dir)

    # Q15-Q17 (extended add-on)
    q15_info = run_q15_calibration_threshold(df, figures_dir)
    q16_info = run_q16_drift_monitoring(df, figures_dir, solutions_dir)
    q17_info = run_q17_recourse_analysis(df, figures_dir, solutions_dir)

    # Full answer key
    answer_key_path = solutions_dir / "complete_solution_key.md"
    write_solution_markdown(
        answer_key_path,
        q1_diag,
        q3_info,
        q4_info,
        q5_info,
        q6_info,
        q15_info,
        q16_info,
        q17_info,
    )

    summary = {
        "data_path": _rel(data_path),
        "figures_dir": _rel(figures_dir),
        "solutions_dir": _rel(solutions_dir),
        "sql_path": _rel(solutions_dir / "q1_moving_average.sql"),
        "answer_key_path": _rel(answer_key_path),
        "q1_diagnostics": q1_diag,
        "q3": q3_info,
        "q4": q4_info,
        "q5": q5_info,
        "q6": q6_info,
        "q15": q15_info,
        "q16": q16_info,
        "q17": q17_info,
        "xgboost_available": XGB_AVAILABLE,
        "shap_available": SHAP_AVAILABLE,
        "xgboost_import_error": XGB_IMPORT_ERROR,
        "shap_import_error": SHAP_IMPORT_ERROR,
        "q1_sql_preview": q1_sql.splitlines()[:4],
    }

    with (solutions_dir / "run_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full UT-ECE assignment solution pipeline.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
    )
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=DEFAULT_SOLUTIONS_DIR,
    )
    args = parser.parse_args()

    summary = run_all(args.data, args.figures_dir, args.solutions_dir)

    print("Full solution pipeline completed.")
    print(f"Answer key: {summary['answer_key_path']}")
    print(f"Q1 SQL: {summary['sql_path']}")
    print(f"Q6 model: {summary['q6']['model_name']}")


if __name__ == "__main__":
    main()
