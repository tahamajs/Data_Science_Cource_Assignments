# Global Tech Talent Migration Assessment (UT-ECE Data Science)

Professional, end-to-end implementation of the University of Tehran ECE final project:
**Analyzing Global Tech Talent Migration: A Data-Driven Approach**.

## Repository Layout

- `code/`: implementation, notebook, tests, LaTeX assets, generated outputs
- `description/`: assignment design prompt and supporting text
- `.github/workflows/ci.yml`: CI test pipeline

## What This Project Delivers

- Full executable pipeline across Q1-Q6 (`SQL`, inference, optimization, non-linear modeling, clustering, explainability)
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
```

## Outputs

After `make run`, generated artifacts are available in:

- `code/solutions/complete_solution_key.md`
- `code/solutions/run_summary.json`
- `code/solutions/q1_moving_average.sql`
- `code/figures/*.png`

## Reproducibility

See `code/docs/REPRODUCIBILITY.md` for exact commands and environment notes.
