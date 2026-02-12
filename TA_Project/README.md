# Global Tech Talent Migration Assessment (UT-ECE Data Science)

Professional, end-to-end implementation of the University of Tehran ECE final project:
**Analyzing Global Tech Talent Migration: A Data-Driven Approach**.

## Repository Layout

- `code/`: implementation, notebook, tests, LaTeX assets, generated outputs
- `description/`: assignment design prompt and supporting text
- `.github/workflows/ci.yml`: CI test pipeline

## What This Project Delivers

- Full executable pipeline across Q1-Q6 (`SQL`, inference, optimization, non-linear modeling, clustering, explainability)
- Extended professional assignment package covering full Spring 2024/2025 topic surface
- Complete written answer key and reproducible artifacts
- Solution notebook aligned with assignment questions
- Teaching-assistant LaTeX manual and assignment handout template
- Automated tests and CI integration

## Quick Start

```bash
cd <repo-root>
source <path-to-venv>/bin/activate
make install
make run
make latex
make report
make latex-fa
make report-fa
```

## Outputs

After `make run`, generated artifacts are available in:

- `code/solutions/complete_solution_key.md`
- `code/solutions/run_summary.json`
- `code/solutions/q1_moving_average.sql`
- `code/figures/*.png`

Extended curriculum package files:

- `code/latex/assignment_extended.tex`
- `code/latex/solution_manual_extended.tex`
- `code/solutions/extended_solution_key.md`
- `code/notebooks/Extended_Assignment_Workbook.ipynb`
- `code/docs/TOPIC_COVERAGE_FROM_UT_REPOS.md`

Complete report package:

- `code/docs/PROJECT_REPORT.md`
- `code/latex/project_report_full.tex`
- `code/latex/project_report_full.pdf`

Complete Persian package:

- `code/docs/ASSIGNMENT_FA.md`
- `code/solutions/SOLUTION_KEY_FA.md`
- `code/docs/PROJECT_REPORT_FA.md`
- `code/latex/assignment_fa.tex`
- `code/latex/solution_manual_fa.tex`
- `code/latex/project_report_full_fa.tex`

## Reproducibility

See `code/docs/REPRODUCIBILITY.md` for exact commands and environment notes.
