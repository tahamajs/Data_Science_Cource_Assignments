# UT-ECE Data Science Final Project

## Analyzing Global Tech Talent Migration (Q1-Q6)

This directory contains a full professional implementation of the Spring 2025 final assessment pipeline, including:

- executable solutions for all six questions
- an extended assignment pack covering full Spring 2024/2025 topic breadth
- a reproducible notebook
- generated grading artifacts (SQL/figures/answer key)
- tests and CI-ready structure
- LaTeX assignment and solution manuals

## Directory Map

- `data/GlobalTechTalent_50k.csv`: source dataset (50,000 records)
- `scripts/generate_synthetic_data.py`: synthetic dataset generator
- `scripts/train_and_explain.py`: baseline modeling + SHAP example
- `scripts/full_solution_pipeline.py`: complete Q1-Q6 pipeline runner
- `notebooks/Solution_Notebook.ipynb`: integrated code + written answers
- `solutions/`: generated answer key and summary outputs
- `figures/`: generated figures from solution runs
- `latex/assignment.tex`: assignment handout
- `latex/solution_manual.tex`: TA grading solution manual
- `latex/assignment_extended.tex`: long-form comprehensive assignment
- `latex/solution_manual_extended.tex`: extended TA solution guide
- `latex/assignment_fa.tex`: Persian comprehensive assignment
- `latex/solution_manual_fa.tex`: Persian TA solution guide
- `latex/project_report_full_fa.tex`: Persian full report with figures
- `tests/`: unit and smoke tests
- `docs/`: report-quality documentation

## Installation

From repository root:

```bash
source <path-to-venv>/bin/activate
make install
```

Or directly:

```bash
pip install -r code/requirements.txt
```

## Reproducible Runs

Run full professional pipeline (Q1-Q6):

```bash
make run
```

Run baseline training/explainability script:

```bash
make baseline
```

Run tests:

```bash
make test
```

Build the complete report with figures:

```bash
make report
```

Build Persian LaTeX assignment + solution:

```bash
make latex-fa
```

Build Persian full report with figures:

```bash
make report-fa
```

## Main Generated Outputs

- `solutions/q1_moving_average.sql`
- `solutions/complete_solution_key.md`
- `solutions/run_summary.json`
- `solutions/q6_fairness_country_rates.csv`
- `figures/q3_ravine_optimizers.png`
- `figures/q4_svm_gamma_sweep.png`
- `figures/q4_tree_pruning_curve.png`
- `figures/q5_kmeans_elbow.png`
- `figures/q6_shap_force_plot.png`
- `figures/q6_shap_summary.png`

## Notes on Capstone Model

- If `xgboost` is installed, Q6 uses XGBoost.
- If `xgboost` is missing, the pipeline automatically falls back to a RandomForest model and still produces SHAP explanations.
- Run metadata records this explicitly in `solutions/run_summary.json`.

## Documentation

- `docs/PROJECT_REPORT.md`: full technical report
- `docs/ASSIGNMENT_FA.md`: full Persian assignment specification
- `solutions/SOLUTION_KEY_FA.md`: full Persian answer key
- `docs/PROJECT_REPORT_FA.md`: full Persian report (markdown)
- `docs/DATA_DICTIONARY.md`: feature-level schema and leakage notes
- `docs/REPRODUCIBILITY.md`: exact commands and environment guidance
- `docs/TOPIC_COVERAGE_FROM_UT_REPOS.md`: mapping from UT public repos to this extended assignment
- `latex/project_report_full.pdf`: compiled full report with embedded figures
- `latex/project_report_full_fa.pdf`: compiled Persian full report with embedded figures
