import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_synthetic_data import generate_dataset
from full_solution_pipeline import run_q4_svm_and_pruning, run_q6_capstone


def test_q4_component_outputs(tmp_path):
    df = generate_dataset(n_rows=900, seed=11)
    result = run_q4_svm_and_pruning(df, tmp_path)

    assert "svm_best_gamma" in result
    assert "tree_best_alpha" in result
    assert result["svm_best_val_accuracy"] >= 0.0
    assert (tmp_path / "q4_svm_gamma_sweep.png").exists()
    assert (tmp_path / "q4_tree_pruning_curve.png").exists()


def test_q6_component_schema(tmp_path):
    df = generate_dataset(n_rows=700, seed=12)
    result = run_q6_capstone(df, tmp_path, tmp_path)

    assert "model_name" in result
    assert "accuracy" in result
    assert "roc_auc" in result
    assert "candidate_prediction_probability" in result
    assert "shap_status" in result
    assert (tmp_path / "q6_fairness_country_rates.csv").exists()
