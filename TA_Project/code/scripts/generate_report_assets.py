#!/usr/bin/env python3
"""Generate additional figures and statistics used by the full project report."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "GlobalTechTalent_50k.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
SOLUTIONS_DIR = PROJECT_ROOT / "solutions"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Migration_Status" not in df.columns:
        raise ValueError("Expected 'Migration_Status' column in dataset.")
    return df


def plot_target_balance(df: pd.DataFrame, out_path: Path) -> dict[str, float]:
    counts = df["Migration_Status"].value_counts().sort_index()
    labels = ["No Migration (0)", "Migration (1)"]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, counts.values, color=["#4c72b0", "#dd8452"])
    plt.ylabel("Count")
    plt.title("Target Distribution: Migration_Status")

    total = float(counts.sum())
    for bar, value in zip(bars, counts.values):
        pct = 100.0 * float(value) / total
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()

    return {
        "class_0_count": float(counts.get(0, 0)),
        "class_1_count": float(counts.get(1, 0)),
        "class_1_rate": float(counts.get(1, 0) / total),
    }


def plot_missingness(df: pd.DataFrame, out_path: Path) -> dict[str, float]:
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing_pct.head(10)

    plt.figure(figsize=(8.5, 5))
    sns.barplot(x=top_missing.values, y=top_missing.index, hue=top_missing.index, palette="viridis", legend=False)
    plt.xlabel("Missing Percentage")
    plt.ylabel("Column")
    plt.title("Top 10 Columns by Missingness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return {str(k): float(v) for k, v in top_missing.to_dict().items()}


def plot_correlation(df: pd.DataFrame, out_path: Path) -> dict[str, list[str]]:
    numeric = df.select_dtypes(include=[np.number]).copy()
    drop_cols = [c for c in ["UserID", "Visa_Approval_Date"] if c in numeric.columns]
    numeric = numeric.drop(columns=drop_cols, errors="ignore")

    corr = numeric.corr(numeric_only=True)
    target_corr = corr["Migration_Status"].abs().sort_values(ascending=False)
    top_features = [c for c in target_corr.index if c != "Migration_Status"][:8]
    selected = ["Migration_Status", *top_features]
    corr_view = corr.loc[selected, selected]

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_view, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Target + Top Numeric Features)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return {"top_correlated_features": top_features}


def plot_country_migration_rate(df: pd.DataFrame, out_path: Path) -> dict[str, float]:
    country = (
        df.groupby("Country_Origin", as_index=False)
        .agg(sample_count=("Migration_Status", "size"), migration_rate=("Migration_Status", "mean"))
        .query("sample_count >= 100")
        .sort_values("migration_rate", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=country,
        x="migration_rate",
        y="Country_Origin",
        hue="Country_Origin",
        palette="mako",
        legend=False,
    )
    plt.xlim(0, 1)
    plt.xlabel("Migration Rate")
    plt.ylabel("Country_Origin")
    plt.title("Top 15 Countries by Migration Rate (n >= 100)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return {row["Country_Origin"]: float(row["migration_rate"]) for _, row in country.iterrows()}


def main() -> None:
    df = load_dataset(DATA_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

    stats = {
        "dataset_rows": float(len(df)),
        "dataset_columns": float(df.shape[1]),
    }

    stats.update(plot_target_balance(df, FIGURES_DIR / "report_target_balance.png"))
    stats["missingness_top10"] = plot_missingness(df, FIGURES_DIR / "report_missingness_top10.png")
    stats.update(plot_correlation(df, FIGURES_DIR / "report_numeric_correlation.png"))
    stats["country_migration_rates_top15"] = plot_country_migration_rate(
        df, FIGURES_DIR / "report_country_migration_rate.png"
    )

    with (SOLUTIONS_DIR / "report_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Report assets generated.")
    print("Figures in:", FIGURES_DIR)
    print("Stats file:", SOLUTIONS_DIR / "report_stats.json")


if __name__ == "__main__":
    main()
