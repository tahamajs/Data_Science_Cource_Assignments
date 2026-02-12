#!/usr/bin/env python3
"""Q19: Uncertainty quantification with split-conformal probability intervals."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Migration_Status"
LEAKAGE_FEATURES = ("Visa_Approval_Date",)
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILE_TO_CONFIG = {
    "fast": {"sample_size": 12000, "alphas": [0.20, 0.10, 0.05], "n_estimators": 140},
    "balanced": {"sample_size": 20000, "alphas": [0.20, 0.10, 0.05, 0.02], "n_estimators": 220},
    "heavy": {"sample_size": 32000, "alphas": [0.20, 0.10, 0.05, 0.02, 0.01], "n_estimators": 320},
}

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    xgb = None
    XGB_AVAILABLE = False


def _rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        transformers=[
            ("cat", _make_ohe(), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )


def _build_model(profile: str, random_state: int):
    n_estimators = PROFILE_TO_CONFIG[profile]["n_estimators"]
    if XGB_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=4,
            eval_metric="logloss",
        )
    return RandomForestClassifier(
        n_estimators=max(120, n_estimators // 2),
        max_depth=10,
        random_state=random_state,
        n_jobs=4,
        class_weight="balanced_subsample",
    )


def _quantile_higher(values: np.ndarray, q: float) -> float:
    q = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:  # pragma: no cover
        return float(np.quantile(values, q, interpolation="higher"))


def run_q19_uncertainty_quantification(
    df: pd.DataFrame,
    figures_dir: Path,
    solutions_dir: Path,
    profile: str = "balanced",
    random_state: int = RANDOM_STATE,
) -> Dict[str, float | str]:
    if profile not in PROFILE_TO_CONFIG:
        raise ValueError(f"Unknown profile '{profile}'.")

    cfg = PROFILE_TO_CONFIG[profile]
    sample_size = int(cfg["sample_size"])
    if len(df) > sample_size:
        work_df = df.sample(n=sample_size, random_state=random_state).copy()
    else:
        work_df = df.copy()

    drop_cols = [TARGET, *[c for c in LEAKAGE_FEATURES if c in work_df.columns]]
    X = work_df.drop(columns=[c for c in drop_cols if c in work_df.columns])
    y = work_df[TARGET].astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_full,
    )

    pre = _build_preprocessor(X_train)
    X_train_enc = pre.fit_transform(X_train)
    X_calib_enc = pre.transform(X_calib)
    X_test_enc = pre.transform(X_test)

    model = _build_model(profile=profile, random_state=random_state)
    model.fit(X_train_enc, y_train)

    p_calib = model.predict_proba(X_calib_enc)[:, 1]
    p_test = model.predict_proba(X_test_enc)[:, 1]

    calib_scores = np.abs(y_calib.to_numpy() - p_calib)
    n_calib = len(calib_scores)

    rows = []
    alphas = [float(a) for a in cfg["alphas"]]
    for alpha in alphas:
        rank_q = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        q_hat = _quantile_higher(calib_scores, min(1.0, rank_q))

        lower = np.clip(p_test - q_hat, 0.0, 1.0)
        upper = np.clip(p_test + q_hat, 0.0, 1.0)
        y_arr = y_test.to_numpy().astype(float)
        covered = ((y_arr >= lower) & (y_arr <= upper)).astype(int)

        nominal = 1.0 - alpha
        empirical = float(covered.mean())
        avg_width = float(np.mean(upper - lower))

        rows.append(
            {
                "alpha": alpha,
                "nominal_coverage": nominal,
                "empirical_coverage": empirical,
                "coverage_gap": empirical - nominal,
                "under_coverage_gap": max(0.0, nominal - empirical),
                "avg_interval_width": avg_width,
                "q_hat": float(q_hat),
            }
        )

    coverage_df = pd.DataFrame(rows).sort_values("alpha", ascending=False)
    coverage_path = solutions_dir / "q19_uncertainty_coverage.csv"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(coverage_path, index=False)

    coverage_plot_path = figures_dir / "q19_coverage_vs_alpha.png"
    coverage_plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8.6, 5.3))
    ax1.plot(
        coverage_df["nominal_coverage"],
        coverage_df["nominal_coverage"],
        linestyle="--",
        color="gray",
        label="Ideal coverage",
    )
    ax1.plot(
        coverage_df["nominal_coverage"],
        coverage_df["empirical_coverage"],
        marker="o",
        color="#1f77b4",
        label="Empirical coverage",
    )
    ax1.set_xlabel("Nominal coverage (1-alpha)")
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(0.0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(
        coverage_df["nominal_coverage"],
        coverage_df["avg_interval_width"],
        marker="s",
        color="#d62728",
        label="Avg interval width",
    )
    ax2.set_ylabel("Average interval width")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    plt.title("Q19: Coverage vs Confidence Level")
    plt.tight_layout()
    plt.savefig(coverage_plot_path, dpi=220)
    plt.close()

    y_pred = (p_test >= 0.5).astype(int)

    by_alpha = {f"coverage_at_{int((1 - a) * 100)}": float(coverage_df.loc[coverage_df["alpha"] == a, "empirical_coverage"].iloc[0]) for a in alphas}

    result: Dict[str, float | str] = {
        "status": "ok",
        "profile": profile,
        "method": "split_conformal_probability_interval",
        "train_size": float(len(X_train)),
        "calibration_size": float(len(X_calib)),
        "test_size": float(len(X_test)),
        "base_accuracy": float(accuracy_score(y_test, y_pred)),
        "base_auc": float(roc_auc_score(y_test, p_test)),
        "base_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "max_under_coverage_gap": float(coverage_df["under_coverage_gap"].max()),
        "mean_interval_width": float(coverage_df["avg_interval_width"].mean()),
        "coverage_table_path": _rel(coverage_path),
        "coverage_plot_path": _rel(coverage_plot_path),
    }
    result.update(by_alpha)
    return result
