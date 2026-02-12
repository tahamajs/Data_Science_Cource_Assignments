# Extended Comprehensive Solution Key

This key corresponds to `latex/assignment_extended.tex` and covers the full topic surface extracted from UT-ECE Spring 2024/2025 repositories.

## Q1. Lifecycle and Problem Framing

A complete answer should define:
- Objective: predict `Migration_Status` for policy/program planning.
- Success metrics: ROC-AUC, F1, calibration error, subgroup fairness gap.
- Lifecycle: problem framing -> ingestion -> quality checks -> feature engineering -> model training -> validation -> explainability -> deployment -> monitoring.
- Failure modes: leakage, label delay, policy drift, cohort shift, and proxy bias.

## Q2. Python, Data Operations, and EDA

Expected implementation:
- robust schema checks (`dtype`, null map, duplicates, basic constraints)
- outlier diagnostics (IQR/z-score/context-aware)
- at least 6 targeted plots with insights (not decorative)
- one reusable preprocessing function + tests

## Q3. Scientific Studies and Inference

High-grade response:
- clearly states this is observational data unless intervention exists
- distinguishes association from causation
- reports one CI and one hypothesis test with assumptions and effect interpretation

## Q4. Visualization Design and Storytelling

Must include:
- KPI definitions linked to decisions
- preattentive and accessibility-aware design choices
- one misleading visualization example and corrected alternative

## Q5. SQL Advanced Querying

Reference query pattern:

```sql
WITH citation_velocity AS (
  SELECT
    UserID,
    Country_Origin,
    Year,
    Research_Citations,
    AVG(Research_Citations) OVER (
      PARTITION BY Country_Origin
      ORDER BY Year
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_citations
  FROM Professionals_Data
)
SELECT
  *,
  DENSE_RANK() OVER (
    PARTITION BY Country_Origin
    ORDER BY moving_avg_citations DESC
  ) AS country_rank
FROM citation_velocity;
```

Also expected:
- percentile bucketing query
- one cohort/transition CTE query

## Q6. Leakage and Big-Data Architecture

Leakage classification:
- `Visa_Approval_Date`: direct leakage
- `Last_Login_Region`: potentially leaky depending on collection time
- `Passport_Renewal_Status`: possible temporal leakage
- `Years_Since_Degree`: usually safe if available at inference time

Architecture answer should include:
- batch + streaming data flow
- feature store with point-in-time correctness
- train/serve parity and drift monitoring

## Q7. Regression and Elastic Net

Elastic Net coordinate gradient:

\[
\nabla_{\theta_j}J = \frac{1}{m}\sum_i(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} + \lambda_1\partial|\theta_j| + \lambda_2\theta_j
\]

Subgradient at zero:

\[
\partial|\theta_j| = [-1, 1] \quad \text{if } \theta_j = 0
\]

Interpretation of statistical significance must be coherent with CI/p-value logic.

## Q8. Optimization (Ravines)

Required explanation:
- SGD oscillates in high-curvature directions
- Momentum accumulates velocity and reduces zig-zag behavior
- Adam adapts per-parameter steps via first/second moments

## Q9. Supervised Model Family Comparison

Expected protocol:
- consistent split + cross-validation
- tuned SVM/KNN, DT/RF, and one boosting model
- unified metric table
- error analysis and threshold discussion

## Q10. Dimensionality Reduction

Expected:
- PCA explained variance with interpretation
- one additional method (random projection, t-SNE, UMAP)
- discussion of interpretability vs geometry preservation

## Q11. Clustering

Expected:
- K-Means with elbow and silhouette evidence
- DBSCAN (or equivalent) with sensitivity analysis
- practical interpretation of cluster meaning and stability

## Q12. Neural Networks and Sequence Models

Expected:
- one tabular NN baseline
- one sequence/text model (CNN/RNN/LSTM/GRU) or equivalent
- training diagnostics and overfitting controls

## Q13. LMs and LLM Agents

Expected:
- agent workflow design (plan/retrieve/reason/verify)
- objective evaluation criteria
- safety/faithfulness governance constraints

## Q14. Ethics, Fairness, and Governance

Expected:
- subgroup metric audit
- proxy-risk discussion
- human-in-the-loop and override policy
- monitoring and incident response expectations

## Q15. Calibration and Threshold Policy (New)

Required analysis:
- calibration curve and reliability interpretation
- Brier score and/or expected calibration error (ECE)
- two threshold policies:
  - threshold maximizing F1
  - threshold minimizing asymmetric expected cost (e.g., FN cost > FP cost)

Recommended conclusion format:
- report both thresholds,
- explain why the operational threshold should follow cost structure rather than default 0.5.

## Q16. Drift Detection and Monitoring (New)

Required analysis:
- split data into reference/current windows (time-based when available)
- compute PSI for numeric features and rank them
- compute one categorical drift signal (e.g., JS divergence)
- define monitoring trigger policy

PSI interpretation guideline:
- PSI < 0.10: low drift
- 0.10 <= PSI < 0.25: moderate drift
- PSI >= 0.25: high drift / investigate + retraining decision

## Q17. Counterfactual Recourse (New)

Required analysis:
- select actionable features
- for near-threshold negative predictions, estimate minimum feature change to cross threshold
- summarize:
  - recourse success rate
  - median required intervention by feature
  - practical/ethical constraints of intervention

High-quality answer must explicitly acknowledge that some features are not truly actionable in real policy settings.

## Q18. Temporal Backtesting and Rolling Validation (New)

Required analysis:
- build chronological rolling folds when valid temporal signal exists
- if no valid temporal field exists, document and justify fallback ordering
- report fold-wise `AUC`/`F1` and decay versus first fold
- add drift-aware interpretation (e.g., mean PSI per fold)

Expected artifacts:
- `code/solutions/q18_temporal_backtest.csv`
- `code/figures/q18_temporal_degradation.png`

## Q19. Uncertainty Quantification and Coverage (New)

Required analysis:
- implement split-conformal (or equivalent) interval/uncertainty method
- evaluate empirical coverage for multiple confidence levels
- report interval width and under-coverage gaps
- provide decision policy for low-confidence predictions

Expected artifacts:
- `code/solutions/q19_uncertainty_coverage.csv`
- `code/figures/q19_coverage_vs_alpha.png`

## Q20. Fairness Mitigation Experiment (New)

Required analysis:
- compute baseline subgroup fairness metrics
- apply one mitigation intervention (reweighing/thresholding/other justified method)
- compare pre/post fairness and utility
- enforce and report explicit policy constraints (e.g., maximum tolerable AUC drop)

Expected artifacts:
- `code/solutions/q20_fairness_mitigation_comparison.csv`
- `code/figures/q20_fairness_tradeoff.png`

## Block J (Bonus): Advanced Extensions
- Causal DAG with identifiability discussion and adjustment sets.
- Conformal/uncertainty intervals with empirical coverage.
- Temporal validation vs random split with degradation analysis.
- Online/streaming serving design with freshness, SLA, OOD/drift guardrail, and rollback.

## Capstone Integrated Requirements

A complete capstone submission must provide:
- leakage-safe model pipeline
- explainability: local + global
- fairness slice table
- reproducible outputs and environment metadata
- stakeholder-facing summary and deployment recommendation

## Professional Evaluation Notes

Award higher marks for:
- reproducible code and deterministic evaluation
- explicit assumptions and limits
- strong interpretation quality
- responsible AI considerations integrated into deployment plan
