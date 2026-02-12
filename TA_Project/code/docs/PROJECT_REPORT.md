# Complete Project Report

## Project
Global Tech Talent Migration Assessment  
University of Tehran - ECE Data Science (Spring 2025)

## Authors
TA Project Implementation Package

## 1. Executive Summary
This project delivers a full, reproducible, end-to-end data science pipeline for predicting `Migration_Status` using `GlobalTechTalent_50k.csv`.

The project covers:
- data engineering and SQL window analytics
- statistical inference and regularized modeling
- optimization behavior analysis (SGD/Momentum/Adam)
- non-linear model tuning and pruning
- unsupervised learning (PCA + KMeans elbow)
- explainability (SHAP local + global)
- fairness slice reporting and governance notes

Final capstone model in the latest run uses **XGBoost** with:
- Accuracy: **0.5835**
- ROC-AUC: **0.5495**
- F1: **0.2475**

## 2. Scope and Alignment
The report and implementation align with topic coverage observed in:
- [DataScience-Spring2024](https://github.com/DataScience-ECE-UniversityOfTehran/DataScience-Spring2024)
- [DataScience-Spring2025](https://github.com/DataScience-ECE-UniversityOfTehran/DataScience-Spring2025)

Detailed mapping is documented in:
- `code/docs/TOPIC_COVERAGE_FROM_UT_REPOS.md`
- `code/docs/EXTENDED_RUBRIC_MAPPING.md`

## 3. Dataset and Problem Statement

### 3.1 Dataset
- File: `code/data/GlobalTechTalent_50k.csv`
- Rows: **50,000**
- Columns: **15**
- Target: `Migration_Status` (1 = migration, 0 = no migration)

### 3.2 Class Balance
- Class 0 count: **29,467**
- Class 1 count: **20,533**
- Positive rate: **41.07%**

![Target Distribution](../figures/report_target_balance.png)

### 3.3 Missingness
Main missingness is expected in `Visa_Approval_Date` (58.93%), which is a leakage-sensitive field.

![Missingness Top 10](../figures/report_missingness_top10.png)

### 3.4 Correlation Snapshot
Top numeric correlations with target include `GitHub_Activity`, `Research_Citations`, and `Remote_Work`.

![Correlation Heatmap](../figures/report_numeric_correlation.png)

### 3.5 Country-Level Outcome Variation
Country-level migration rates vary within a relatively tight band (for countries with adequate sample size), reinforcing the need for fairness-aware interpretation.

![Country Migration Rates](../figures/report_country_migration_rate.png)

## 4. Data Engineering and Leakage Controls

### 4.1 SQL Window Function Deliverable
The required Q1A SQL moving-average query is delivered in:
- `code/solutions/q1_moving_average.sql`

### 4.2 Leakage Policy
- `Visa_Approval_Date` is excluded from training features.
- Temporal leakage is explicitly discussed for `Last_Login_Region` and `Passport_Renewal_Status`.

Latest diagnostics (from `run_summary.json`):
- corr(visa_present, migration_status) = **1.000**
- P(Migration=1 | visa present) = **1.000**
- P(Migration=1 | visa absent) = **0.000**

These values validate the leakage risk and justify exclusion.

## 5. Statistical Inference and Linear Modeling
The project includes formal Elastic Net derivation and interpretation guidance in:
- `code/latex/solution_manual.tex`
- `code/solutions/complete_solution_key.md`
- `code/solutions/extended_solution_key.md`

Core derivation concept:
- MSE gradient + L1 subgradient + L2 gradient
- explicit subgradient handling at coefficient value zero

## 6. Optimization Analysis
A synthetic ravine objective is used to compare optimization dynamics.

Final losses:
- SGD: **0.403329**
- Momentum: **0.000823**
- Adam: **0.000034**

Interpretation:
- Momentum reduces oscillation in high-curvature directions.
- Adam adapts effective step sizes per parameter and converges faster on this objective.

![Ravine Optimizer Trajectories](../figures/q3_ravine_optimizers.png)

## 7. Non-Linear Modeling Results

### 7.1 SVM Gamma Tuning
- Best gamma: **0.005**
- Best validation accuracy: **0.600**
- Worst validation accuracy: **0.591**

This supports the expected recommendation: reduce `gamma` under overfitting risk.

![SVM Gamma Sweep](../figures/q4_svm_gamma_sweep.png)

### 7.2 Decision Tree Cost-Complexity Pruning
- Best `ccp_alpha`: **0.009639**
- Best validation accuracy: **0.600**

![Decision Tree Pruning Curve](../figures/q4_tree_pruning_curve.png)

## 8. Unsupervised Learning
PCA + KMeans elbow diagnostics:
- EVR(PC1): **0.277**
- EVR(PC2): **0.145**
- EVR(PC1 + PC2): **0.422**
- Elbow-selected K: **4**

![KMeans Elbow](../figures/q5_kmeans_elbow.png)

## 9. Capstone Explainability (SHAP)

### 9.1 Model and Metrics
Latest capstone model:
- Model: **XGBoost**
- Accuracy: **0.5835**
- ROC-AUC: **0.5495**
- F1: **0.2475**

### 9.2 Local Explanation
Candidate-level explanation details:
- Candidate index: **27343**
- Predicted migration probability: **0.3892**
- SHAP base value: **-0.3494**
- SHAP output value: **-0.4507**
- Output space: **log-odds**
- Top local contributor: `num__Research_Citations`

![SHAP Local Force/Waterfall](../figures/q6_shap_force_plot.png)

### 9.3 Global Explainability
Global feature-importance view is provided in:

![SHAP Global Importance](../figures/q6_shap_summary.png)

## 10. Fairness and Governance
Fairness slice output:
- `code/solutions/q6_fairness_country_rates.csv`

Report policy:
- treat model output as predictive, not causal
- audit subgroup performance before deployment
- keep human review for high-impact decisions
- monitor for data/label/policy drift

## 11. Reproducibility and Engineering Quality

### 11.1 Execution
- Full pipeline: `make run`
- Tests: `make test`
- Compile checks: `make compile`
- LaTeX builds: `make latex`

### 11.2 Artifacts
Core outputs:
- `code/solutions/complete_solution_key.md`
- `code/solutions/extended_solution_key.md`
- `code/solutions/run_summary.json`
- `code/solutions/report_stats.json`
- `code/solutions/q1_moving_average.sql`
- `code/figures/*.png`

### 11.3 CI
CI workflow validates install, compilation, and tests:
- `.github/workflows/ci.yml`

## 12. Limitations
- This dataset does not encode all real-world migration drivers (e.g., family, geopolitics, policy shocks).
- Fairness analysis is slice-based and should be extended with threshold-sensitive audits and intervention policy.
- XAI interpretations are descriptive and not causal evidence.

## 13. Future Work
- Add causal framing (DAG + sensitivity checks).
- Add temporal validation by year for drift-robust evaluation.
- Add calibration and cost-sensitive decision threshold policy.
- Add scenario-based LLM-agent evaluation with safety guardrails.

## 14. Conclusion
This project is fully implemented, reproducible, and report-complete across technical, methodological, and governance dimensions. It includes all major deliverables expected in a professional university capstone: code, tests, figures, explainability, fairness analysis, extended curriculum alignment, and publication-ready documentation.

---

## Appendix A: Key Files
- `code/scripts/full_solution_pipeline.py`
- `code/scripts/train_and_explain.py`
- `code/scripts/generate_report_assets.py`
- `code/notebooks/Solution_Notebook.ipynb`
- `code/notebooks/Extended_Assignment_Workbook.ipynb`
- `code/latex/assignment.tex`
- `code/latex/solution_manual.tex`
- `code/latex/assignment_extended.tex`
- `code/latex/solution_manual_extended.tex`

## Appendix B: Evidence Sources in This Repo
- `code/solutions/run_summary.json`
- `code/solutions/report_stats.json`
- `code/solutions/q6_fairness_country_rates.csv`
- `code/docs/TOPIC_COVERAGE_FROM_UT_REPOS.md`
