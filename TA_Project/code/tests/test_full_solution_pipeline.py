import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_synthetic_data import generate_dataset
from full_solution_pipeline import leakage_diagnostics, simulate_optimizers, run_q5_unsupervised, write_q1_sql


def test_write_q1_sql_and_leakage(tmp_path):
    sql_path = tmp_path / "q1.sql"
    sql_text = write_q1_sql(sql_path)
    assert sql_path.exists()
    assert "PARTITION BY Country_Origin" in sql_text

    df = generate_dataset(n_rows=500, seed=1)
    diag = leakage_diagnostics(df)
    assert "visa_presence_corr_with_target" in diag
    assert diag["target_rate_if_visa_present"] >= diag["target_rate_if_visa_absent"]


def test_simulate_optimizers_shape_and_keys():
    paths = simulate_optimizers(steps=30)
    assert set(["sgd", "momentum", "adam", "a", "b"]).issubset(paths.keys())
    assert paths["sgd"].shape[0] == 31
    assert paths["momentum"].shape[1] == 2


def test_q5_unsupervised_runs(tmp_path):
    df = generate_dataset(n_rows=1200, seed=2)
    result = run_q5_unsupervised(df, tmp_path)
    assert 0.0 < result["pca_pc1_ratio"] < 1.0
    assert 0.0 < result["pca_pc2_ratio"] < 1.0
    assert result["kmeans_elbow_k"] >= 1
    assert (tmp_path / "q5_kmeans_elbow.png").exists()
