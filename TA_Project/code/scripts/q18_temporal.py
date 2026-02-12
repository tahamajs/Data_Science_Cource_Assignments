#!/usr/bin/env python3
"""Q18: Temporal backtesting with rolling validation and drift-aware decay analysis."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Migration_Status"
LEAKAGE_FEATURES = ("Visa_Approval_Date",)
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILE_TO_CONFIG = {
    "fast": {"sample_size": 12000, "n_folds": 3, "min_train_ratio": 0.55, "test_ratio": 0.15, "n_estimators": 140},
    "balanced": {
        "sample_size": 20000,
        "n_folds": 5,
        "min_train_ratio": 0.52,
        "test_ratio": 0.12,
        "n_estimators": 220,
    },
    "heavy": {"sample_size": 32000, "n_folds": 7, "min_train_ratio": 0.50, "test_ratio": 0.10, "n_estimators": 320},
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


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


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


def _infer_temporal_order(df: pd.DataFrame) -> Tuple[pd.Series, str, str]:
    candidates = ["Year", "Record_Year", "Application_Year", "Timestamp", "Date"]
    for col in candidates:
        if col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) >= 4:
                return series.astype(float), col, "explicit_time_column"
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().sum() > len(df) * 0.7:
                return parsed.view("int64") / 1e9, col, "explicit_datetime_column"

    if "UserID" in df.columns and pd.api.types.is_numeric_dtype(df["UserID"]):
        return df["UserID"].astype(float), "UserID", "fallback_userid_proxy"

    return pd.Series(np.arange(len(df)), index=df.index, dtype=float), "row_index", "fallback_row_index"


def _prepare_data(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy()
    sampled = df.sample(n=sample_size, random_state=random_state)
    return sampled.sort_index().copy()


def _rolling_boundaries(n: int, n_folds: int, min_train_ratio: float, test_ratio: float) -> List[Tuple[int, int]]:
    if n < 600:
        split = max(200, int(n * 0.7))
        return [(split, n)]

    min_train_end = max(250, int(n * min_train_ratio))
    horizon = max(120, int(n * test_ratio))

    bounds: List[Tuple[int, int]] = []
    train_end = min_train_end
    for _ in range(n_folds):
        test_end = min(n, train_end + horizon)
        if test_end - train_end < 80:
            break
        bounds.append((train_end, test_end))
        train_end = test_end
        if train_end >= n - 80:
            break

    if not bounds:
        split = max(200, int(n * 0.7))
        bounds = [(split, n)]
    return bounds


def run_q18_temporal_backtesting(
    df: pd.DataFrame,
    figures_dir: Path,
    solutions_dir: Path,
    profile: str = "balanced",
    random_state: int = RANDOM_STATE,
) -> Dict[str, float | str]:
    if profile not in PROFILE_TO_CONFIG:
        raise ValueError(f"Unknown profile '{profile}'.")

    cfg = PROFILE_TO_CONFIG[profile]
    work_df = _prepare_data(df, sample_size=int(cfg["sample_size"]), random_state=random_state)

    temporal_key, temporal_col, split_strategy = _infer_temporal_order(work_df)
    ordered = work_df.assign(_temporal_key=temporal_key).sort_values("_temporal_key").reset_index(drop=True)

    bounds = _rolling_boundaries(
        n=len(ordered),
        n_folds=int(cfg["n_folds"]),
        min_train_ratio=float(cfg["min_train_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
    )

    rows = []
    numeric_cols = [
        c
        for c in ordered.select_dtypes(include=[np.number]).columns
        if c not in {TARGET, "UserID", "_temporal_key"}
    ]

    for fold_idx, (train_end, test_end) in enumerate(bounds, start=1):
        train_df = ordered.iloc[:train_end].copy()
        test_df = ordered.iloc[train_end:test_end].copy()
        if train_df.empty or test_df.empty:
            continue

        drop_cols = [TARGET, "_temporal_key", *[c for c in LEAKAGE_FEATURES if c in ordered.columns]]
        X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
        y_train = train_df[TARGET].astype(int)
        X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
        y_test = test_df[TARGET].astype(int)

        pre = _build_preprocessor(X_train)
        X_train_enc = pre.fit_transform(X_train)
        X_test_enc = pre.transform(X_test)

        model = _build_model(profile=profile, random_state=random_state + fold_idx)
        model.fit(X_train_enc, y_train)

        y_prob = model.predict_proba(X_test_enc)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fold_psi = []
        for col in numeric_cols:
            fold_psi.append(_compute_psi(train_df[col], test_df[col]))
        psi_values = np.array([v for v in fold_psi if not np.isnan(v)], dtype=float)

        rows.append(
            {
                "fold": fold_idx,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "train_key_end": float(train_df["_temporal_key"].iloc[-1]),
                "test_key_start": float(test_df["_temporal_key"].iloc[0]),
                "test_key_end": float(test_df["_temporal_key"].iloc[-1]),
                "auc": _safe_roc_auc(y_test.to_numpy(), y_prob),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "mean_numeric_psi": float(np.nanmean(psi_values)) if psi_values.size else float("nan"),
            }
        )

    backtest_df = pd.DataFrame(rows)
    if backtest_df.empty:
        # Fallback to a single chronological split to avoid empty artifacts.
        train_df, test_df = train_test_split(ordered, test_size=0.2, random_state=random_state, shuffle=False)
        drop_cols = [TARGET, "_temporal_key", *[c for c in LEAKAGE_FEATURES if c in ordered.columns]]
        X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
        y_train = train_df[TARGET].astype(int)
        X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
        y_test = test_df[TARGET].astype(int)

        pre = _build_preprocessor(X_train)
        X_train_enc = pre.fit_transform(X_train)
        X_test_enc = pre.transform(X_test)
        model = _build_model(profile=profile, random_state=random_state)
        model.fit(X_train_enc, y_train)

        y_prob = model.predict_proba(X_test_enc)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        backtest_df = pd.DataFrame(
            [
                {
                    "fold": 1,
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "train_key_end": float(train_df["_temporal_key"].iloc[-1]),
                    "test_key_start": float(test_df["_temporal_key"].iloc[0]),
                    "test_key_end": float(test_df["_temporal_key"].iloc[-1]),
                    "auc": _safe_roc_auc(y_test.to_numpy(), y_prob),
                    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "mean_numeric_psi": float("nan"),
                }
            ]
        )
        split_strategy = f"{split_strategy}; fallback_single_split"

    first_auc = float(backtest_df["auc"].dropna().iloc[0]) if backtest_df["auc"].notna().any() else float("nan")
    last_auc = float(backtest_df["auc"].dropna().iloc[-1]) if backtest_df["auc"].notna().any() else float("nan")
    first_f1 = float(backtest_df["f1"].iloc[0])
    last_f1 = float(backtest_df["f1"].iloc[-1])

    backtest_df["auc_decay_vs_first"] = backtest_df["auc"] - first_auc if not math.isnan(first_auc) else np.nan
    backtest_df["f1_decay_vs_first"] = backtest_df["f1"] - first_f1

    table_path = solutions_dir / "q18_temporal_backtest.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(table_path, index=False)

    plot_path = figures_dir / "q18_temporal_degradation.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9.0, 5.4))
    ax1.plot(backtest_df["fold"], backtest_df["auc"], marker="o", label="AUC", color="#1f77b4")
    ax1.plot(backtest_df["fold"], backtest_df["f1"], marker="s", label="F1", color="#ff7f0e")
    ax1.set_xlabel("Rolling fold")
    ax1.set_ylabel("Performance")

    ax2 = ax1.twinx()
    ax2.plot(
        backtest_df["fold"],
        backtest_df["mean_numeric_psi"],
        marker="^",
        label="Mean PSI",
        color="#2ca02c",
        linestyle="--",
    )
    ax2.set_ylabel("Drift proxy (Mean PSI)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    plt.title("Q18: Temporal Performance Decay vs Drift")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=220)
    plt.close()

    return {
        "status": "ok",
        "profile": profile,
        "temporal_column": temporal_col,
        "split_strategy": split_strategy,
        "fold_count": float(len(backtest_df)),
        "auc_first_fold": first_auc,
        "auc_last_fold": last_auc,
        "auc_decay_absolute": float(last_auc - first_auc) if not (math.isnan(last_auc) or math.isnan(first_auc)) else float("nan"),
        "f1_first_fold": first_f1,
        "f1_last_fold": last_f1,
        "f1_decay_absolute": float(last_f1 - first_f1),
        "mean_auc": float(backtest_df["auc"].mean(skipna=True)),
        "mean_f1": float(backtest_df["f1"].mean(skipna=True)),
        "mean_numeric_psi": float(backtest_df["mean_numeric_psi"].mean(skipna=True)),
        "backtest_table_path": _rel(table_path),
        "degradation_plot_path": _rel(plot_path),
        "fallback_used": "fallback" in split_strategy,
    }
