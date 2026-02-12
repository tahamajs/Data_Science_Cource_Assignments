import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_synthetic_data import generate_dataset
from full_solution_pipeline import (
    run_q15_calibration_threshold,
    run_q16_drift_monitoring,
    run_q17_recourse_analysis,
)


def test_q15_calibration_threshold_outputs(tmp_path):
    df = generate_dataset(n_rows=900, seed=21)
    result = run_q15_calibration_threshold(df, tmp_path)

    assert "brier_score" in result
    assert "best_f1_threshold" in result
    assert result["brier_score"] >= 0.0
    assert (tmp_path / "q15_calibration_curve.png").exists()
    assert (tmp_path / "q15_threshold_tradeoff.png").exists()


def test_q16_drift_outputs(tmp_path):
    df = generate_dataset(n_rows=1000, seed=22)
    result = run_q16_drift_monitoring(df, tmp_path, tmp_path)

    assert "top_drift_feature" in result
    assert "top_drift_psi" in result
    assert (tmp_path / "q16_drift_psi.csv").exists()
    assert (tmp_path / "q16_drift_psi_top12.png").exists()


def test_q17_recourse_outputs(tmp_path):
    df = generate_dataset(n_rows=950, seed=23)
    result = run_q17_recourse_analysis(df, tmp_path, tmp_path)

    assert "recourse_success_rate" in result
    assert "successful_recourse_count" in result
    assert (tmp_path / "q17_recourse_examples.csv").exists()
    assert (tmp_path / "q17_recourse_median_deltas.png").exists()
