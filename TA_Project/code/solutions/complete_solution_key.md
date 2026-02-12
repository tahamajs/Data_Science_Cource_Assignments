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
J(	heta)=rac{1}{2m}\sum_{i=1}^m (h_	heta(x^{(i)})-y^{(i)})^2 + \lambda_1\sum_{j=1}^n|	heta_j| + rac{\lambda_2}{2}\sum_{j=1}^n	heta_j^2,
\]
for parameter \(	heta_j\):
\[

abla_{	heta_j}J(	heta)=rac{1}{m}\sum_{i=1}^m (h_	heta(x^{(i)})-y^{(i)})x_j^{(i)} + \lambda_1\,\partial|	heta_j| + \lambda_2	heta_j,
\]
where
\[
\partial|	heta_j|=egin{cases}
+1 & 	heta_j>0\
-1 & 	heta_j<0\
[-1,1] & 	heta_j=0
\end{cases}
\]
At \(	heta_j=0\), coordinate-descent solvers use this subgradient set and can keep coefficients exactly at zero (feature selection).

### Q2B. Interpretation of \(eta=0.52\), p-value \(=0.003\), 95% CI \([0.18, 0.86]\)
- Because p-value < 0.05, reject \(H_0: eta=0\).
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
- Best validation accuracy: **0.581**
- Worst validation accuracy: **0.574**

Figure: `code/figures/q4_svm_gamma_sweep.png`

### Q4B. Cost-complexity pruning
\[
R_lpha(T)=R(T)+lpha|T|
\]
- Increasing \(lpha\) increases penalty for leaf count, producing smaller trees.
- Small \(lpha\): low bias, high variance.
- Large \(lpha\): higher bias, lower variance.

Run metrics:
- Best \(lpha\): **0.006833**
- Best validation accuracy after pruning: **0.581**

Figure: `code/figures/q4_tree_pruning_curve.png`

## Q5. Unsupervised Learning

### Q5A. PCA explained variance ratio
For covariance matrix eigenvalues \(\lambda_1, \lambda_2, \lambda_3\):
\[
	ext{EVR}_k = rac{\lambda_k}{\lambda_1+\lambda_2+\lambda_3}
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

Model used in this run: **RandomForest (XGBoost fallback)**
- Accuracy: **0.594**
- ROC-AUC: **0.573**
- F1: **0.261**

Candidate explanation details:
- Candidate index: **13530**
- Predicted migration probability: **0.509**
- SHAP status: **ok**
- Base value: **0.50000**
- Output value: **0.50857**
- Sigmoid(output value): **0.62447**

Interpretation:
- `base_value` is the model's average output over the background set.
- `output_value` is the local output for the candidate.
- Their difference is the sum of per-feature SHAP contributions for that candidate.
- Positive SHAP values push toward migration; negative values push toward no migration.

Artifacts:
- Local force/waterfall plot: `/Users/tahamajs/Documents/uni/DS/TA_Project/code/figures/q6_shap_force_plot.png`
- Global SHAP summary plot: `/Users/tahamajs/Documents/uni/DS/TA_Project/code/figures/q6_shap_summary.png`
- Country fairness slice: `/Users/tahamajs/Documents/uni/DS/TA_Project/code/solutions/q6_fairness_country_rates.csv`

## Fairness note for grading discussion
Even with strong predictive metrics, model decisions can mirror historical policy constraints. Country-level predicted positive rates should be audited against domain knowledge before any deployment.
