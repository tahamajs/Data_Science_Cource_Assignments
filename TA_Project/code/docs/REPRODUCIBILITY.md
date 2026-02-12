# Reproducibility Guide

## Environment

Recommended activation:

```bash
source <path-to-venv>/bin/activate
```

Install dependencies:

```bash
pip install -r code/requirements.txt
```

## Full Pipeline (Q1-Q6)

```bash
MPLCONFIGDIR=/tmp/mpl LOKY_MAX_CPU_COUNT=4 \
python code/scripts/full_solution_pipeline.py \
  --data code/data/GlobalTechTalent_50k.csv \
  --figures-dir code/figures \
  --solutions-dir code/solutions
```

## Baseline Pipeline

```bash
MPLCONFIGDIR=/tmp/mpl LOKY_MAX_CPU_COUNT=4 \
python code/scripts/train_and_explain.py \
  --data code/data/GlobalTechTalent_50k.csv \
  --figdir code/figures
```

## Tests

```bash
python -m pytest code/tests -q
```

## Determinism Notes

- Random seeds are fixed in data generation and modeling scripts.
- If `xgboost` is not installed, Q6 falls back to RandomForest and this is logged in `code/solutions/run_summary.json`.
- `MPLCONFIGDIR=/tmp/mpl` avoids matplotlib cache permission issues in restricted environments.
