#!/usr/bin/env python3
"""End-to-end implementation for the UT-ECE Global Tech Talent Migration assessment.

This script generates:
- SQL answer artifact for Q1A
- leakage diagnostics for Q1B
- optimizer ravine simulations for Q3
- SVM gamma + decision tree pruning diagnostics for Q4
- PCA + KMeans elbow analysis for Q5
- capstone model + SHAP local explanation for Q6
- a consolidated markdown answer key aligned with Q1-Q6
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
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET = "Migration_Status"
LEAKAGE_FEATURES = ["Visa_Approval_Date"]
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "GlobalTechTalent_50k.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"
DEFAULT_SOLUTIONS_DIR = PROJECT_ROOT / "solutions"

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


def run_q4_svm_and_pruning(df: pd.DataFrame, figures_dir: Path) -> Dict[str, float]:
    # Numeric-only subset keeps the non-linear comparison computationally manageable.
    sample = df.sample(n=min(1800, len(df)), random_state=RANDOM_STATE).copy()
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

    gammas = [0.005, 0.02, 0.08, 0.3]
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

    if len(raw_alphas) > 35:
        alphas = np.unique(np.quantile(raw_alphas, np.linspace(0, 1, 35)))
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


def run_q5_unsupervised(df: pd.DataFrame, figures_dir: Path) -> Dict[str, float]:
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
        len(X_scaled), size=min(12000, len(X_scaled)), replace=False
    )
    X_cluster = X_scaled[sample_idx]

    ks = np.arange(1, 11)
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
        "kmeans_wcss_k10": float(wcss[-1]),
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


def run_q6_capstone(df: pd.DataFrame, figures_dir: Path, solutions_dir: Path) -> Dict[str, float | str]:
    work_df = df.sample(n=min(20000, len(df)), random_state=RANDOM_STATE).copy()
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


def write_solution_markdown(
    out_path: Path,
    q1_diag: Dict[str, float],
    q3_info: Dict[str, float],
    q4_info: Dict[str, float],
    q5_info: Dict[str, float],
    q6_info: Dict[str, float | str],
) -> None:
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

    # Full answer key
    answer_key_path = solutions_dir / "complete_solution_key.md"
    write_solution_markdown(answer_key_path, q1_diag, q3_info, q4_info, q5_info, q6_info)

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
