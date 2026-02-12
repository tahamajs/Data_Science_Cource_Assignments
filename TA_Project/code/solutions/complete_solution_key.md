# Global Tech Talent Migration — Complete Solution Key

This document provides complete answers for Q1-Q6 and links each answer to executable artifacts in this repository.

## Q1. Advanced Data Engineering & SQL

### Q1A. 3-year moving average + rank query
The SQL answer is saved in `code/solutions/q1_moving_average.sql`.

### Q1B. Data leakage diagnosis
Leaked feature(s):
- `Visa_Approval_Date` is **direct leakage** because it encodes post-outcome bureaucratic status.
- `Last_Login_Region` can become **temporal leakage** if captured after migration and reflects destination behavior.
- `Passport_Renewal_Status` is potentially leaky if timestamps are post-decision.
- `Years_Since_Degree` is generally safe if computed from pre-decision records.

Dataset evidence from this run:
- corr(visa_present, migration_status) = **1.000**
- P(Migration=1 | visa present) = **1.000**
- P(Migration=1 | visa absent) = **0.000**

## Q2. Statistical Inference & Linear Models

### Q2A. Elastic Net gradient derivation
For
\[
J(\theta)=\frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2 + \lambda_1\sum_{j=1}^n|\theta_j| + \frac{\lambda_2}{2}\sum_{j=1}^n\theta_j^2,
\]
for parameter \(\theta_j\):
\[
\nabla_{\theta_j}J(\theta)=\frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} + \lambda_1\,\partial|\theta_j| + \lambda_2\theta_j,
\]
where
\[
\partial|\theta_j|=\begin{cases}
+1 & \theta_j>0\\
-1 & \theta_j<0\\
[-1,1] & \theta_j=0
\end{cases}
\]
At \(\theta_j=0\), coordinate-descent solvers use this subgradient set and can keep coefficients exactly at zero (feature selection).

### Q2B. Interpretation of \(\beta=0.52\), p-value \(=0.003\), 95% CI \([0.18, 0.86]\)
- Because p-value < 0.05, reject \(H_0: \beta=0\).
- The confidence interval excludes 0, confirming statistical significance.
- The positive interval implies higher `GitHub_Activity` is associated with higher migration propensity (under model assumptions and conditioning on other covariates).

## Q3. Optimization & Gradient Descent

Ravine behavior: steep curvature in one dimension and shallow curvature in another causes vanilla SGD to zig-zag and progress slowly.

From this run (final losses on the toy ravine):
- SGD final loss: **0.403329**
- Momentum final loss: **0.000823**
- Adam final loss: **0.000034**

Interpretation:
- Momentum damps oscillations by accumulating velocity, so opposing gradients on steep walls cancel over time.
- Adam additionally rescales updates per-parameter using first/second moments, usually improving stability when feature scales differ.

Figure: `code/figures/q3_ravine_optimizers.png`

## Q4. Non-Linear Models & Kernels

### Q4A. SVM with RBF kernel (overfitting case)
If the model overfits, **decrease \(\gamma\)**.
- High \(\gamma\): narrow influence radius around each point, very wiggly boundary, high variance.
- Lower \(\gamma\): broader influence, smoother boundary, lower variance.

Run metrics:
- Best validation gamma: **0.005**
- Best validation accuracy: **0.600**
- Worst validation accuracy: **0.591**

Figure: `code/figures/q4_svm_gamma_sweep.png`

### Q4B. Cost-complexity pruning
\[
R_\alpha(T)=R(T)+\alpha|T|
\]
- Increasing \(\alpha\) increases penalty for leaf count, producing smaller trees.
- Small \(\alpha\): low bias, high variance.
- Large \(\alpha\): higher bias, lower variance.

Run metrics:
- Best \(\alpha\): **0.009639**
- Best validation accuracy after pruning: **0.600**

Figure: `code/figures/q4_tree_pruning_curve.png`

## Q5. Unsupervised Learning

### Q5A. PCA explained variance ratio
For covariance matrix eigenvalues \(\lambda_1, \lambda_2, \lambda_3\):
\[
\text{EVR}_k = \frac{\lambda_k}{\lambda_1+\lambda_2+\lambda_3}
\]
Eigenvalue interpretation: variance captured along principal component \(k\).

Run results:
- EVR(PC1): **0.277**
- EVR(PC2): **0.145**
- EVR(PC1+PC2): **0.422**

### Q5B. K-Means elbow method rationale
- WCSS decreases monotonically with larger \(K\) because each additional centroid can only reduce or keep the same minimum squared distances.
- The elbow approximates the point of diminishing returns where marginal WCSS reduction drops sharply.
- Geometrically, this is near maximal curvature on the WCSS-vs-\(K\) curve.

Run result:
- Elbow-selected \(K\): **4**

Figure: `code/figures/q5_kmeans_elbow.png`

## Q6. Capstone Explainability (SHAP)

Model used in this run: **XGBoost**
- Accuracy: **0.584**
- ROC-AUC: **0.550**
- F1: **0.248**

Candidate explanation details:
- Candidate index: **27343**
- Predicted migration probability: **0.389**
- SHAP status: **ok**
- Base value: **-0.34942**
- Output value: **-0.45072**
- Output space: **log_odds**
- Probability implied by SHAP output: **0.38919**

Interpretation:
- `base_value` is the model's average output over the background set.
- `output_value` is the local output for the candidate.
- Their difference is the sum of per-feature SHAP contributions for that candidate.
- Positive SHAP values push toward migration; negative values push toward no migration.

Artifacts:
- Local force/waterfall plot: `figures/q6_shap_force_plot.png`
- Global SHAP summary plot: `figures/q6_shap_summary.png`
- Country fairness slice: `solutions/q6_fairness_country_rates.csv`

## Q15. Calibration and Threshold Policy (New)

Why this matters:
- A high AUC model can still be poorly calibrated.
- Decision threshold should be chosen by utility/cost, not only default 0.5.

Run results:
- Model: **XGBoost**
- ROC-AUC: **0.541**
- Brier score: **0.2436**
- Expected calibration error: **0.0327**
- Best threshold by F1: **0.25** (F1=0.585)
- Best threshold by expected cost: **0.25**
- Minimum expected cost per sample: **0.5823**

Artifacts:
- Calibration curve: `figures/q15_calibration_curve.png`
- Threshold tradeoff plot: `figures/q15_threshold_tradeoff.png`

## Q16. Drift Monitoring and Data Stability (New)

Drift diagnostics use PSI (Population Stability Index) across reference/current windows.

Run results:
- Split rule: **random half split**
- Reference size: **25000**
- Current size: **25000**
- Top drift feature: **Visa_Approval_Date**
- Top drift PSI: **0.0013**
- High-drift features (PSI >= 0.25): **0**
- Moderate-drift features (0.10 <= PSI < 0.25): **0**
- Country distribution JS divergence: **0.0002**

Artifacts:
- Drift table: `solutions/q16_drift_psi.csv`
- Drift plot: `figures/q16_drift_psi_top12.png`

## Q17. Counterfactual Recourse Analysis (New)

Question addressed:
- For near-boundary non-migrant predictions, what is the minimum actionable change needed to flip decision to migration-positive?

Run results:
- Model: **XGBoost**
- Decision threshold: **0.50**
- Candidates considered: **120**
- Successful recourse count: **120**
- Recourse success rate: **1.000**
- Median delta (GitHub\_Activity): **2.000**
- Median delta (Research\_Citations): **50.000**
- Median delta (Industry\_Experience): **0.500**

Artifacts:
- Recourse examples table: `solutions/q17_recourse_examples.csv`
- Recourse effort plot: `figures/q17_recourse_median_deltas.png`

## Fairness note for grading discussion
Even with strong predictive metrics, model decisions can mirror historical policy constraints. Country-level predicted positive rates should be audited against domain knowledge before any deployment.
