SHELL := /bin/bash
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

PROJECT_ROOT := $(CURDIR)
CODE_DIR := $(PROJECT_ROOT)/code

.PHONY: install test run baseline notebook-check compile

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r $(CODE_DIR)/requirements.txt

test:
	$(PYTHON) -m pytest $(CODE_DIR)/tests -q

run:
	MPLCONFIGDIR=/tmp/mpl LOKY_MAX_CPU_COUNT=4 $(PYTHON) $(CODE_DIR)/scripts/full_solution_pipeline.py

baseline:
	MPLCONFIGDIR=/tmp/mpl LOKY_MAX_CPU_COUNT=4 $(PYTHON) $(CODE_DIR)/scripts/train_and_explain.py

notebook-check:
	$(PYTHON) - <<'PY'
import nbformat
nb = nbformat.read('code/notebooks/Solution_Notebook.ipynb', as_version=4)
print('Notebook cells:', len(nb.cells))
PY

compile:
	$(PYTHON) -m py_compile $(CODE_DIR)/scripts/full_solution_pipeline.py $(CODE_DIR)/scripts/train_and_explain.py $(CODE_DIR)/scripts/generate_synthetic_data.py
