#!/usr/bin/env python3
"""Q20: Fairness mitigation experiment with pre/post comparison and policy constraints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Tuple

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
SENSITIVE_FEATURE = "Country_Origin"
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILE_TO_CONFIG = {
    "fast": {"sample_size": 12000, "n_estimators": 140},
    "balanced": {"sample_size": 20000, "n_estimators": 220},
    "heavy": {"sample_size": 32000, "n_estimators": 320},
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


def _safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    arr = y_true.to_numpy()
    if len(np.unique(arr)) < 2:
        return float("nan")
    return float(roc_auc_score(arr, y_prob))


def _fairness_table(
    y_true: pd.Series,
    y_pred: np.ndarray,
    groups: pd.Series,
    min_group_size: int = 80,
) -> pd.DataFrame:
    df_eval = pd.DataFrame(
        {
            "group": groups.astype(str).fillna("UNKNOWN"),
            "y_true": y_true.astype(int).to_numpy(),
            "y_pred": y_pred.astype(int),
        }
    )

    rows = []
    for group, part in df_eval.groupby("group"):
        n = len(part)
        if n < min_group_size:
            continue
        pred_rate = float(part["y_pred"].mean())
        positives = part[part["y_true"] == 1]
        tpr = float(positives["y_pred"].mean()) if not positives.empty else float("nan")
        rows.append(
            {
                "group": group,
                "n": n,
                "pred_positive_rate": pred_rate,
                "tpr": tpr,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["group", "n", "pred_positive_rate", "tpr"])
    return out.sort_values("pred_positive_rate", ascending=False)


def _gap_or_nan(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.max() - valid.min())


def _evaluate(
    model,
    X_train_enc,
    y_train,
    X_test_enc,
    y_test,
    groups_test: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train_enc, y_train, **fit_kwargs)

    y_prob = model.predict_proba(X_test_enc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    fairness_df = _fairness_table(y_true=y_test, y_pred=y_pred, groups=groups_test)
    dp_gap = _gap_or_nan(fairness_df["pred_positive_rate"]) if not fairness_df.empty else float("nan")
    eopp_gap = _gap_or_nan(fairness_df["tpr"]) if not fairness_df.empty else float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": _safe_auc(y_test, y_prob),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "demographic_parity_gap": dp_gap,
        "equal_opportunity_gap": eopp_gap,
        "max_group_pred_rate": float(fairness_df["pred_positive_rate"].max()) if not fairness_df.empty else float("nan"),
        "min_group_pred_rate": float(fairness_df["pred_positive_rate"].min()) if not fairness_df.empty else float("nan"),
        "max_group_tpr": float(fairness_df["tpr"].max()) if not fairness_df.empty else float("nan"),
        "min_group_tpr": float(fairness_df["tpr"].min()) if not fairness_df.empty else float("nan"),
    }
    return metrics, fairness_df


def _compute_reweighing_weights(y_train: pd.Series, groups_train: pd.Series) -> np.ndarray:
    eps = 1e-6
    train_df = pd.DataFrame({"y": y_train.astype(int).to_numpy(), "g": groups_train.astype(str).fillna("UNKNOWN")})

    p_y = train_df["y"].value_counts(normalize=True).to_dict()
    p_g = train_df["g"].value_counts(normalize=True).to_dict()
    p_yg = train_df.groupby(["y", "g"]).size() / len(train_df)

    overall_pos = float(train_df["y"].mean())
    group_pos = train_df.groupby("g")["y"].mean().to_dict()

    weights = np.ones(len(train_df), dtype=float)
    for i, row in train_df.iterrows():
        y_val = int(row["y"])
        g_val = str(row["g"])

        base = p_y.get(y_val, eps) * p_g.get(g_val, eps) / max(float(p_yg.get((y_val, g_val), eps)), eps)

        g_pos = float(group_pos.get(g_val, overall_pos))
        if y_val == 1:
            tilt = overall_pos / max(g_pos, eps)
        else:
            tilt = (1.0 - overall_pos) / max(1.0 - g_pos, eps)

        weights[i] = base * tilt

    return np.clip(weights, 0.25, 6.0)


def run_q20_fairness_mitigation(
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

    if SENSITIVE_FEATURE not in work_df.columns:
        raise ValueError(f"Sensitive column '{SENSITIVE_FEATURE}' is missing from dataset.")

    drop_cols = [TARGET, *[c for c in LEAKAGE_FEATURES if c in work_df.columns]]
    X = work_df.drop(columns=[c for c in drop_cols if c in work_df.columns])
    y = work_df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    groups_train = X_train[SENSITIVE_FEATURE].astype(str).fillna("UNKNOWN")
    groups_test = X_test[SENSITIVE_FEATURE].astype(str).fillna("UNKNOWN")

    pre = _build_preprocessor(X_train)
    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)

    baseline_model = _build_model(profile=profile, random_state=random_state)
    baseline_metrics, baseline_fairness = _evaluate(
        model=baseline_model,
        X_train_enc=X_train_enc,
        y_train=y_train,
        X_test_enc=X_test_enc,
        y_test=y_test,
        groups_test=groups_test,
    )

    weights = _compute_reweighing_weights(y_train=y_train, groups_train=groups_train)
    mitigated_model = _build_model(profile=profile, random_state=random_state + 11)
    mitigated_metrics, mitigated_fairness = _evaluate(
        model=mitigated_model,
        X_train_enc=X_train_enc,
        y_train=y_train,
        X_test_enc=X_test_enc,
        y_test=y_test,
        groups_test=groups_test,
        sample_weight=weights,
    )

    auc_drop = float(baseline_metrics["roc_auc"] - mitigated_metrics["roc_auc"])
    f1_drop = float(baseline_metrics["f1"] - mitigated_metrics["f1"])
    dp_improved = mitigated_metrics["demographic_parity_gap"] <= baseline_metrics["demographic_parity_gap"]
    eopp_improved = mitigated_metrics["equal_opportunity_gap"] <= baseline_metrics["equal_opportunity_gap"]

    max_auc_drop = 0.03
    max_f1_drop = 0.05
    fairness_gate = dp_improved or eopp_improved
    policy_pass = (auc_drop <= max_auc_drop) and (f1_drop <= max_f1_drop) and fairness_gate

    comparison_df = pd.DataFrame(
        [
            {
                "scenario": "baseline",
                **baseline_metrics,
                "auc_drop_vs_baseline": 0.0,
                "f1_drop_vs_baseline": 0.0,
                "policy_pass": True,
            },
            {
                "scenario": "mitigated_reweighing",
                **mitigated_metrics,
                "auc_drop_vs_baseline": auc_drop,
                "f1_drop_vs_baseline": f1_drop,
                "policy_pass": bool(policy_pass),
            },
        ]
    )

    comparison_path = solutions_dir / "q20_fairness_mitigation_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)

    baseline_group_path = solutions_dir / "q20_fairness_groups_baseline.csv"
    mitigated_group_path = solutions_dir / "q20_fairness_groups_mitigated.csv"
    baseline_fairness.to_csv(baseline_group_path, index=False)
    mitigated_fairness.to_csv(mitigated_group_path, index=False)

    plot_path = figures_dir / "q20_fairness_tradeoff.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.8, 5.2))
    plt.scatter(
        baseline_metrics["demographic_parity_gap"],
        baseline_metrics["roc_auc"],
        color="#1f77b4",
        s=90,
        label="Baseline",
    )
    plt.scatter(
        mitigated_metrics["demographic_parity_gap"],
        mitigated_metrics["roc_auc"],
        color="#d62728",
        s=90,
        label="Mitigated",
    )
    plt.annotate(
        "",
        xy=(mitigated_metrics["demographic_parity_gap"], mitigated_metrics["roc_auc"]),
        xytext=(baseline_metrics["demographic_parity_gap"], baseline_metrics["roc_auc"]),
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.2},
    )
    plt.xlabel("Demographic parity gap (lower is better)")
    plt.ylabel("ROC-AUC (higher is better)")
    plt.title("Q20: Fairness-Performance Tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=220)
    plt.close()

    return {
        "status": "ok",
        "profile": profile,
        "sensitive_feature": SENSITIVE_FEATURE,
        "mitigation_method": "reweighing_with_group_label_tilt",
        "baseline_roc_auc": float(baseline_metrics["roc_auc"]),
        "mitigated_roc_auc": float(mitigated_metrics["roc_auc"]),
        "baseline_f1": float(baseline_metrics["f1"]),
        "mitigated_f1": float(mitigated_metrics["f1"]),
        "baseline_demographic_parity_gap": float(baseline_metrics["demographic_parity_gap"]),
        "mitigated_demographic_parity_gap": float(mitigated_metrics["demographic_parity_gap"]),
        "baseline_equal_opportunity_gap": float(baseline_metrics["equal_opportunity_gap"]),
        "mitigated_equal_opportunity_gap": float(mitigated_metrics["equal_opportunity_gap"]),
        "auc_drop_vs_baseline": auc_drop,
        "f1_drop_vs_baseline": f1_drop,
        "policy_constraint": "auc_drop<=0.03 AND f1_drop<=0.05 AND (dp_gap_improved OR eopp_gap_improved)",
        "policy_pass": bool(policy_pass),
        "comparison_table_path": _rel(comparison_path),
        "baseline_groups_table_path": _rel(baseline_group_path),
        "mitigated_groups_table_path": _rel(mitigated_group_path),
        "tradeoff_plot_path": _rel(plot_path),
    }
