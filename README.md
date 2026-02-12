# Data Science Course – University of Tehran (Complete Repository)

[![Python](https://img.shields.io/badge/Python-3.8%E2%80%933.12-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-0db7ed.svg)](https://www.docker.com/)
[![Kafka](https://img.shields.io/badge/Apache-Kafka-black.svg)](https://kafka.apache.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

A single, end-to-end home for every assignment, notebook, lecture slide, resource pack, and capstone project delivered in the University of Tehran Data Science course. This README is intentionally exhaustive so new contributors, students, and reviewers can find exactly what they need without hunting through folders.

---

## Table of Contents
- [Quick Overview](#quick-overview)
- [Repository Layout](#repository-layout)
- [Environment & Tooling](#environment--tooling)
- [Setup Guide](#setup-guide)
- [Working With Notebooks & Data](#working-with-notebooks--data)
- [Coursework Modules](#coursework-modules)
  - [CA0 — Statistical Inference & Monte Carlo](#ca0--statistical-inference--monte-carlo)
  - [CA1 — Data Visualization & Score-Based Sampling](#ca1--data-visualization--score-based-sampling)
  - [CA2 — Real-Time Streaming with Kafka](#ca2--real-time-streaming-with-kafka)
  - [CA3 — Advanced ML, Regression & Recommender Systems](#ca3--advanced-ml-regression--recommender-systems)
  - [CA4 — Deep Learning (MLP, CNN, RNN)](#ca4--deep-learning-mlp-cnn-rnn)
  - [CA5&6 — NLP & Semi-Supervised Learning](#ca56--nlp--semi-supervised-learning)
- [Capstone: Uber Demand Prediction](#capstone-uber-demand-prediction)
- [TA Project (Teaching Assistant Resources)](#ta-project-teaching-assistant-resources)
- [Lectures, Cheat Sheets & Supplementary Resources](#lectures-cheat-sheets--supplementary-resources)
- [Coding Standards, Naming & Submission Rules](#coding-standards-naming--submission-rules)
- [Data & Storage Notes](#data--storage-notes)
- [Support & Contact](#support--contact)
- [License](#license)
- [Last Updated](#last-updated)

---

## Quick Overview
- **Scope**: 6 core coursework assignments (CA0–CA6), a full production-style capstone, lecture slide deck, Python foundations notebooks, cheat sheets, and a TA-grade reference implementation.
- **Languages & Tools**: Python 3.8–3.12, Jupyter, scikit-learn, PyTorch/TensorFlow (DL tasks), Apache Kafka (streaming), Docker & Compose (capstone), MySQL (capstone DB), GitHub Actions (CI/CD).
- **Audience**: Students following the course, instructors/graders, and collaborators who need a single entry point to all artifacts.

---

## Repository Layout
Top-level directories are organized by assignment and resource type:

```
/Users/tahamajs/Documents/uni/DS
├── CA0_Statistical_Inference_Monte_Carlo/    # Prob/inference & Monte Carlo simulation
├── CA1_Data_Visualization_Score_Sampling/    # Visualization & score-based sampling
├── CA2_Real_Time_Streaming_Kafka/            # Event streaming with Apache Kafka
├── CA3_Advanced_ML_Regression_RecSys/        # Regression + recommender systems
├── CA4_Deep_Learning_Neural_Networks/        # MLP, CNN, RNN assignments
├── CA56_NLP_Semi_Supervised_Learning/        # NLP + semi-supervised text modeling
├── Data_Science_Final_Project/               # Capstone (phase 2 & 3) + presentation
├── TA_Project/                               # TA reference solution for migration case study
├── Python for Data Science Notebooks/        # Intro Python → pandas → matplotlib notebooks
├── Cheetsheet/                               # PDF quick references
├── lectures/                                 # Lecture slide PDFs (01–14)
├── More Resources/                           # Extra material (PowerBI, web scraping, etc.)
├── LICENSE                                   # Academic license
└── README.md                                 # You are here
```

---

## Environment & Tooling
- **Python**: 3.8–3.12 are used across assignments. Prefer 3.10+ for best library support.
- **Package managers**: `pip` + `venv` (per-assignment requirements files where applicable).
- **Jupyter**: Run notebooks locally or in VS Code/JupyterLab.
- **Docker & Docker Compose**: Required for the capstone (`Data_Science_Final_Project/phase3`).
- **Apache Kafka**: Required for CA2 streaming labs; install locally or run via Docker (not included here).
- **Databases**: MySQL 8.0 for the capstone; SQLite/lightweight storage for small labs where noted.
- **Git LFS**: Recommended if you add large models or datasets.

---

## Setup Guide
1. **Clone**
   ```bash
   git clone <repo-url>
   cd DS
   ```
2. **Create a virtual environment** (recommended per assignment)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   - For foundational notebooks: `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`
   - For each assignment or phase, check the local `requirements.txt` or the README inside its folder.
4. **Run notebooks**
   ```bash
   jupyter notebook
   ```
   Open the relevant `.ipynb` inside the assignment directory.
5. **Docker-based workflows (capstone)**
   ```bash
   cd Data_Science_Final_Project/phase3
   docker-compose up --build
   ```
6. **Kafka workflows (CA2)**
   - Start a Kafka broker & Zookeeper locally (or via Docker).
   - Update broker addresses in `CA2_Real_Time_Streaming_Kafka/codes/config/` if needed.

---

## Working With Notebooks & Data
- Keep large datasets in their provided `datasets/` or `data/` subfolders to preserve relative paths used in notebooks.
- When copying notebooks, also copy companion assets (config files, encoders, scalers, joblib artifacts) that live beside the notebook.
- Many notebooks assume UTF-8 CSVs; if paths contain spaces, wrap them in quotes when running scripts.
- For reproducibility, seed-setting is included in most notebooks (`numpy`, `random`, `torch`/`tf` where applicable).

---

## Coursework Modules
### CA0 — Statistical Inference & Monte Carlo
**Path**: `CA0_Statistical_Inference_Monte_Carlo/`  
**Focus**: Law of Large Numbers, Central Limit Theorem, confidence intervals, hypothesis testing, and Monte Carlo simulation.  
**Key assets**: `codes/notebook.ipynb`, `datasets/2016-general-election-trump-vs-clinton.csv`, `datasets/drug_safety.csv`, `description/CA0.pdf`.  
**Run**: `cd CA0_Statistical_Inference_Monte_Carlo && jupyter notebook codes/notebook.ipynb`

### CA1 — Data Visualization & Score-Based Sampling
**Path**: `CA1_Data_Visualization_Score_Sampling/`  
**Focus**: Score function estimation, unadjusted Langevin dynamics, Gaussian mixture sampling, and Airbnb price/geo analysis.  
**Key assets**: `code/code.ipynb`, `dataset/Airbnb_Listings.xlsx`, `dataset/Neighborhood_Locations.xlsx`, `description/CA1.pdf`.  
**Run**: `cd CA1_Data_Visualization_Score_Sampling && jupyter notebook code/code.ipynb`

### CA2 — Real-Time Streaming with Kafka
**Path**: `CA2_Real_Time_Streaming_Kafka/`  
**Focus**: Event-driven architecture, Kafka producers/consumers, Poisson-based synthetic transaction generation, monitoring, and storage patterns.  
**Key assets**: producer/consumer scripts under `codes/`, Poisson generator `base_codes/darooghe_pulse.py`, architecture/report PDFs in `description/` and `report/`.  
**Run (example)**:
```bash
# Terminal 1: start broker externally
cd CA2_Real_Time_Streaming_Kafka/codes/producers && python transactions_producer.py
# Terminal 2: consume/process
cd CA2_Real_Time_Streaming_Kafka/codes/consumers && python transactions_consumer.py
```
Update topic/broker settings in `codes/config/` to match your Kafka instance.

### CA3 — Advanced ML, Regression & Recommender Systems
**Path**: `CA3_Advanced_ML_Regression_RecSys/`  
**Focus**: Bike-demand regression with rich feature engineering, collaborative-filtering recommender ensembles (SVD, SVD++, KNN Baseline), and visualization/statistical analysis tasks.  
**Key assets**: `codes/Q1.zip` (bike sharing), `codes/Q2.py` (movie recommender), `codes/Q3.py` + `codes/visualizations/`, assignment PDFs under `descriptions/`, analytic reports under `reports/`.  
**Run (recommender)**:
```bash
cd CA3_Advanced_ML_Regression_RecSys/codes
python Q2.py --data_dir ./dataset  # adjust data path if needed
```

### CA4 — Deep Learning (MLP, CNN, RNN)
**Path**: `CA4_Deep_Learning_Neural_Networks/`  
**Focus**: Implementing and training MLPs, CNNs, and RNN/LSTM/GRU models; image classification and sequence forecasting; transfer learning and data augmentation.  
**Key assets**: `codes/Task1/Task1_MLP.ipynb`, `codes/Task2/Task2_CNNs.ipynb`, `codes/Task3/Task3_RNNs.ipynb`, datasets (`datasets/BTC-USD.csv`, `datasets/matches.csv`), `description/DS-CA4.pdf`.  
**Run**: open notebooks in Jupyter; GPU optional but beneficial for CNN/RNN sections.

### CA5&6 — NLP & Semi-Supervised Learning
**Path**: `CA56_NLP_Semi_Supervised_Learning/`  
**Focus**: Text vectorization (SentenceTransformers, Word2Vec), supervised baselines, pseudo-labeling, active learning, and co-training on limited labeled data for game-review score prediction.  
**Key assets**: `code/` notebooks and scripts, `datasets/` (labeled/unlabeled CSVs, JSON format), `description/` for assignment brief.  
**Run**: `cd CA56_NLP_Semi_Supervised_Learning && jupyter notebook code/<notebook>.ipynb`

---

## Capstone: Uber Demand Prediction
**Path**: `Data_Science_Final_Project/`  
**Phases**:
- **Phase 2** (`phase2/`): earlier pipeline prototype with `pipeline.py`, `requirements.txt`, and automation scripts (`run.sh`, `Makefile`).
- **Phase 3** (`phase3/`): production-grade pipeline with Dockerized services, MySQL schema, CI/CD, health checks, and logging.

**Highlights (Phase 3)**
- Handles 14M+ NYC Uber trip records (2015) with weather and spatial enrichment.
- Automated ETL → preprocessing → feature engineering → model training/evaluation.
- Models: Gradient Boosting, Random Forest, XGBoost, Neural Nets; peak-time classification + demand regression.
- Containers: `docker-compose.yml` spins up app, MySQL, phpMyAdmin, and Jupyter; entrypoint managed by `docker-entrypoint.sh`.

**Quick Start (Phase 3)**
```bash
cd Data_Science_Final_Project/phase3
cp .env.example .env   # adjust DB credentials/ports if needed
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Local run
python pipeline.py
# Or containerized
docker-compose up --build
```
Detailed architecture, schema, and task descriptions live in `phase3/README.md` and the presentation deck in `presentation/Project-Presentation.pdf`.

---

## TA Project (Teaching Assistant Resources)
**Path**: `TA_Project/`  
A fully worked, CI-enabled solution set for the "Global Tech Talent Migration" assessment. Includes implementation, answer keys, LaTeX assets, and automation via `make`. See `TA_Project/README.md` for commands (e.g., `make install`, `make run`) and output locations under `code/solutions/`.

---

## Lectures, Cheat Sheets & Supplementary Resources
- **Lecture slides** (`lectures/`): 14 PDF decks covering the entire course sequence (lifecycle, probability, SQL, ML, DL, applications).
- **Cheat sheets** (`Cheetsheet/`): Quick-reference PDFs for stats, ML, and tooling.
- **Python foundations** (`Python for Data Science Notebooks/`): 8 progressive notebooks from data types through NumPy/pandas/Matplotlib.
- **More Resources/**: Additional material (e.g., PowerBI samples, web scraping examples, extra CA0/CA1 references).

---

## Coding Standards, Naming & Submission Rules
- **Submission naming** (per course policy): `DS_CA{NUMBER}_{ID1}_{ID2}_{ID3}.zip`.
- **Reproducibility**: Prefer deterministic seeds; document package versions in `requirements.txt` when adding code.
- **Notebooks**: Keep outputs cleared when committing unless plots/tables are needed for grading.
- **Data paths**: Use relative paths inside notebooks/scripts so they run from the assignment root.
- **Style**: PEP 8 for Python; SQL files formatted with uppercase keywords.

---

## Data & Storage Notes
- Large CSVs and model artifacts are kept within each assignment’s `datasets/`, `data/`, or `models/` folders to avoid cross-coupling.
- If you add new data, avoid committing files >100MB unless Git LFS is configured.
- Model binaries (e.g., `*.joblib`, encoders, scalers) for the capstone live in `Data_Science_Final_Project/phase3/models/` (see `model_metadata.json`).

---

## Support & Contact
- For course-related questions: reach out via University of Tehran channels or the course issue tracker (if enabled).
- For repository issues or suggestions: open a GitHub issue referencing the relevant assignment folder.

---

## License
This repository is distributed for academic purposes under the terms described in `LICENSE`.

---

## Last Updated
February 12, 2026

<div align="center">
  <strong>⭐ If this collection helps you, consider starring the repository.</strong>
</div>
