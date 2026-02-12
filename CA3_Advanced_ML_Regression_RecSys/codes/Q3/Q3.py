#!/usr/bin/env python
"""
Efficient Ensemble Recommender – v2.1 (bagging‑fix)

CLI-friendly and typed. Includes a fast grid mode for quick iteration.
Fix: copying tuned SVD hyper‑params for bagging now uses a whitelist to avoid
passing internal attrs (e.g. ``bsl_options``) that ``SVD`` rejects.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, SVDpp, KNNBaseline
from surprise.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(dir_: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read training and test CSVs from a directory."""
    ratings = pd.read_csv(dir_ / "train_data_movie_rate.csv")
    test = pd.read_csv(dir_ / "test_data.csv")
    return ratings, test


def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows and average duplicate user–item labels."""
    df = df.dropna(subset=["user_id", "item_id", "label"]).copy()
    df["label"] = df["label"].astype(float)
    return df.groupby(["user_id", "item_id"], as_index=False)["label"].mean()


def build_dataset(ratings: pd.DataFrame) -> Dataset:
    """Build a Surprise Dataset from ratings."""
    reader = Reader(rating_scale=(ratings.label.min(), ratings.label.max()))
    return Dataset.load_from_df(ratings[["user_id", "item_id", "label"]], reader)

# ---------------------------------------------------------------------------
# CV tuning helper
# ---------------------------------------------------------------------------

def tune_algo(algo_cls, grid: Dict, data: Dataset, name: str, n_jobs: int = -1):
    """Grid-search RMSE for a Surprise algorithm and return the fitted model plus params/score."""
    gs = GridSearchCV(algo_cls, grid, measures=["rmse"], cv=3, n_jobs=n_jobs)
    gs.fit(data)
    print(f"✓ {name} best RMSE {gs.best_score['rmse']:.4f} params {gs.best_params['rmse']}")
    model = algo_cls(**gs.best_params["rmse"])
    model.fit(data.build_full_trainset())
    return model, gs.best_params["rmse"], gs.best_score["rmse"]

# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)

    return {
        "RMSE": mse ** 0.5,
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Efficient ensemble recommender v2.1")
    parser.add_argument("--data_dir", type=Path, default=Path("./dataset/"), help="Directory containing train/test CSV files.")
    parser.add_argument("--fast", action="store_true", help="Use smaller grids for faster tuning.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallelism for grid search.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    ratings, test_df = load_data(args.data_dir)
    ratings = clean_ratings(ratings)
    data = build_dataset(ratings)
    rmin, rmax = ratings.label.min(), ratings.label.max()

    # ---------------- Hyper‑param grids ----------------------------------
    if args.fast:
        svd_grid = {"n_factors": [160], "lr_all": [0.005], "reg_all": [0.02], "n_epochs": [40]}
        svdpp_grid = {"n_factors": [160], "lr_all": [0.005], "reg_all": [0.05], "n_epochs": [40]}
        knn_grid = {"k": [60], "min_k": [3], "sim_options": {"name": ["pearson_baseline"], "user_based": [False]}}
    else:
        svd_grid = {"n_factors": [120, 160, 200], "lr_all": [0.003, 0.005, 0.007], "reg_all": [0.01, 0.02, 0.04], "n_epochs": [60, 80]}
        svdpp_grid = {"n_factors": [120, 160], "lr_all": [0.003, 0.005, 0.007], "reg_all": [0.03, 0.05], "n_epochs": [40, 60]}
        knn_grid = {"k": [40, 60, 80], "min_k": [3, 5], "sim_options": {"name": ["pearson_baseline"], "user_based": [False]}}

    svd_best, svd_params, svd_cv = tune_algo(SVD, svd_grid, data, "SVD", n_jobs=args.n_jobs)
    svdpp_best, svdpp_params, svdpp_cv = tune_algo(SVDpp, svdpp_grid, data, "SVD++", n_jobs=args.n_jobs)
    knn_best, _, knn_cv = tune_algo(KNNBaseline, knn_grid, data, "KNN‑Baseline", n_jobs=args.n_jobs)

    # ---------------- Bagging SVDs --------------------------------------
    bagging_seeds = [7, 42, 2025]
    param_whitelist = ["n_factors", "n_epochs", "biased", "lr_all", "reg_all", "random_state"]
    base_svd_params = {k: svd_params[k] for k in param_whitelist if k in svd_params}

    svd_bag = []
    for s in bagging_seeds:
        p = base_svd_params.copy()
        p["random_state"] = s
        m = SVD(**p)
        m.fit(data.build_full_trainset())
        svd_bag.append(m)

    base_models = [svd_best, svdpp_best, knn_best] + svd_bag

    # ---------------- Blender training ----------------------------------
    trainset, holdout = train_test_split(data, test_size=0.2, random_state=1)
    for m in base_models:
        m.fit(trainset)

    y_hold, X_hold = [], []
    for uid, iid, r in holdout:
        y_hold.append(r)
        X_hold.append([m.predict(uid, iid).est for m in base_models])
    y_hold = np.array(y_hold)
    X_hold = np.array(X_hold)

    blender = LinearRegression(positive=True, fit_intercept=False)
    blender.fit(X_hold, y_hold)
    weights = blender.coef_ / blender.coef_.sum()
    print("Learned weights:", np.round(weights, 3))

    ens_pred_hold = np.clip(X_hold.dot(weights), rmin, rmax)
    print("Hold‑out ensemble metrics:", metric_dict(y_hold, ens_pred_hold))

    # ---------------- Predict test pairs --------------------------------
    ests = []
    for u, i in test_df[["user_id", "item_id"]].values:
        preds = np.array([m.predict(u, i).est for m in base_models])
        ests.append(float(np.clip(preds.dot(weights), rmin, rmax)))

    submission = pd.DataFrame({"id": test_df["id"], "label": ests})
    sub_path = args.data_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print("Saved submission to", sub_path.resolve())


if __name__ == "__main__":
    main()
