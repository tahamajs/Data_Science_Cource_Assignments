import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_synthetic_data import generate_dataset
from full_solution_pipeline import run_all


def test_run_summary_schema_v2_and_new_artifacts(tmp_path):
    data_path = tmp_path / "synthetic.csv"
    figures_dir = tmp_path / "figures"
    solutions_dir = tmp_path / "solutions"

    df = generate_dataset(n_rows=1800, seed=44)
    df.to_csv(data_path, index=False)

    summary = run_all(
        data_path=data_path,
        figures_dir=figures_dir,
        solutions_dir=solutions_dir,
        profile="fast",
        enable_q18=True,
        enable_q19=True,
        enable_q20=True,
    )

    assert summary["run_summary_version"] == 2
    assert summary["runtime_profile"] == "fast"
    assert "data_split_strategy" in summary
    assert summary["metric_export_version"]

    for key in ["q18", "q19", "q20"]:
        assert key in summary
        assert "status" in summary[key]

    run_summary_path = solutions_dir / "run_summary.json"
    assert run_summary_path.exists()

    loaded = json.loads(run_summary_path.read_text(encoding="utf-8"))
    assert loaded["run_summary_version"] == 2

    expected_files = [
        solutions_dir / "q18_temporal_backtest.csv",
        solutions_dir / "q19_uncertainty_coverage.csv",
        solutions_dir / "q20_fairness_mitigation_comparison.csv",
        figures_dir / "q18_temporal_degradation.png",
        figures_dir / "q19_coverage_vs_alpha.png",
        figures_dir / "q20_fairness_tradeoff.png",
        solutions_dir / "latex_metrics.json",
        solutions_dir / "latex_metrics.tex",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing artifact: {path}"
