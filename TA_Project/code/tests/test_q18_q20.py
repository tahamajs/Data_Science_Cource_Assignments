import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_synthetic_data import generate_dataset
from q18_temporal import run_q18_temporal_backtesting
from q19_uncertainty import run_q19_uncertainty_quantification
from q20_fairness_mitigation import run_q20_fairness_mitigation


def test_q18_temporal_artifacts_and_metrics(tmp_path):
    df = generate_dataset(n_rows=1400, seed=31)
    out = run_q18_temporal_backtesting(df, figures_dir=tmp_path, solutions_dir=tmp_path, profile="fast")

    assert out["status"] == "ok"
    assert out["fold_count"] >= 1
    assert "split_strategy" in out
    assert (tmp_path / "q18_temporal_backtest.csv").exists()
    assert (tmp_path / "q18_temporal_degradation.png").exists()


def test_q19_uncertainty_outputs(tmp_path):
    df = generate_dataset(n_rows=1500, seed=32)
    out = run_q19_uncertainty_quantification(df, figures_dir=tmp_path, solutions_dir=tmp_path, profile="fast")

    assert out["status"] == "ok"
    assert out["method"].startswith("split_conformal")
    assert out["max_under_coverage_gap"] >= 0.0
    assert (tmp_path / "q19_uncertainty_coverage.csv").exists()
    assert (tmp_path / "q19_coverage_vs_alpha.png").exists()


def test_q20_fairness_mitigation_outputs(tmp_path):
    df = generate_dataset(n_rows=1600, seed=33)
    out = run_q20_fairness_mitigation(df, figures_dir=tmp_path, solutions_dir=tmp_path, profile="fast")

    assert out["status"] == "ok"
    assert out["sensitive_feature"] == "Country_Origin"
    assert "policy_pass" in out
    assert (tmp_path / "q20_fairness_mitigation_comparison.csv").exists()
    assert (tmp_path / "q20_fairness_tradeoff.png").exists()
