SHELL := /bin/bash
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

PROJECT_ROOT := $(CURDIR)
CODE_DIR := $(PROJECT_ROOT)/code

.PHONY: install test run baseline notebook-check compile latex latex-fa report-assets report report-fa

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
	$(PYTHON) -c "import nbformat; nb=nbformat.read('code/notebooks/Solution_Notebook.ipynb', as_version=4); print('Notebook cells:', len(nb.cells))"

compile:
	$(PYTHON) -m py_compile $(CODE_DIR)/scripts/full_solution_pipeline.py $(CODE_DIR)/scripts/train_and_explain.py $(CODE_DIR)/scripts/generate_synthetic_data.py $(CODE_DIR)/scripts/generate_report_assets.py

latex:
	cd $(CODE_DIR)/latex && xelatex -interaction=nonstopmode assignment.tex && xelatex -interaction=nonstopmode solution_manual.tex && xelatex -interaction=nonstopmode assignment_extended.tex && xelatex -interaction=nonstopmode solution_manual_extended.tex

latex-fa:
	cd $(CODE_DIR)/latex && xelatex -interaction=nonstopmode assignment_fa.tex && xelatex -interaction=nonstopmode solution_manual_fa.tex

report-assets:
	$(PYTHON) $(CODE_DIR)/scripts/generate_report_assets.py

report: report-assets
	cd $(CODE_DIR)/latex && xelatex -interaction=nonstopmode project_report_full.tex && xelatex -interaction=nonstopmode project_report_full.tex

report-fa: report-assets
	cd $(CODE_DIR)/latex && xelatex -interaction=nonstopmode project_report_full_fa.tex && xelatex -interaction=nonstopmode project_report_full_fa.tex
