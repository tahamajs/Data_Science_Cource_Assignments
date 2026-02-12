# Assignment 0: Statistical Inference & Monte Carlo Simulation

---

## 📊 Project Overview

This project explores fundamental concepts of **statistical inference**, **Monte Carlo simulation**, and **hypothesis testing**. The assignment consists of three major parts, each focusing on critical data science concepts: probability theory, confidence intervals, and hypothesis testing.

---

## 🎯 Learning Objectives

### Part 1: Roulette Simulation & Law of Large Numbers
- Understand the **Law of Large Numbers** through practical simulation
- Implement Monte Carlo simulation for a roulette game
- Explore the **Central Limit Theorem (CLT)**
- Calculate confidence intervals and standard errors

### Part 2: Electoral Data Analysis (2016 US Election)
- Apply statistical inference to real-world polling data
- Calculate **confidence intervals** for population proportions
- Perform **hypothesis testing** for statistical significance
- Compare theoretical results with Monte Carlo simulations

### Part 3: Drug Safety Testing
- Conduct **independent t-tests** for clinical trial data
- Compare treatment vs. control groups
- Interpret **p-values** and statistical significance
- Analyze adverse effects and safety metrics

---

## 🔬 Core Concepts & Techniques

### 1. Probability Foundations
- **Gaussian Distribution**: Bell curve and normal probability
- **Law of Large Numbers**: Sample mean converges to population mean
- **Central Limit Theorem**: Distribution of sample means becomes normal
- **Standard Error**: Measure of estimate variability

### 2. Confidence Intervals
```
CI = μ̂ ± z × SE
```
- **z**: Critical value from standard normal distribution
- **SE**: Standard error of the estimate
- **95% Confidence Interval**: Contains true parameter with 95% probability

### 3. Hypothesis Testing
- **Null Hypothesis (H₀)**: Assumption of no difference
- **Alternative Hypothesis (H₁)**: Assumption of a difference
- **p-value**: Probability of observing results under H₀
- **Significance Level (α)**: Typically 0.05

### 4. Monte Carlo Simulation
- Use random sampling to estimate distributions
- Repeat experiments many times
- Compare empirical results with theoretical predictions

---

## 📁 Project Structure

```
Statistical_Inference_Monte_Carlo/
├── codes/
│   └── notebook.ipynb          # Main project notebook
├── datasets/
│   ├── 2016-general-election-trump-vs-clinton.csv
│   └── drug_safety.csv
├── description/
│   └── CA0.pdf                 # Assignment description
└── README.md                   # This file
```

---

## 🛠️ Technologies & Libraries

### Python Libraries
```python
numpy                 # Numerical computations
pandas                # Data manipulation
matplotlib            # Plotting and visualization
scipy.stats           # Statistical functions
random                # Random number generation
```

**Quick start**
```bash
cd CA0_Statistical_Inference_Monte_Carlo/codes
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib scipy
jupyter notebook notebook.ipynb
```

### Statistical Techniques
- **Monte Carlo Simulation**
- **Confidence Intervals**
- **Hypothesis Testing**
- **Independent t-tests**
- **Central Limit Theorem**
- **Law of Large Numbers**

---

## 📊 Problems Solved

### Problem 1: Roulette Simulation (6 Questions)

#### 1.1: Simulation Function
Implemented roulette game with 18 red, 18 black, and 2 green slots

#### 1.2: Analysis with Variable N
Simulated for N = {10, 25, 100, 1000}
- Convergence to normal distribution
- Standard error decreases as N increases

#### 1.3: Average Winnings Distribution
Calculated \( S_N/N \) and analyzed convergence

#### 1.4: Comparison with Theoretical Results
```
E[X] = -1/19 ≈ -0.0526
Var[X] = E[X²] - (E[X])²
```

#### 1.5: Probability of Casino Loss
Used CLT to calculate probabilities

#### 1.6: Probability vs. N Plot
Why do casinos encourage continued play?

---

### Problem 2: 2016 US Election Analysis (9 Questions)

#### Dataset
Polling data for Trump vs. Clinton election

#### Analyses Performed
1. **Confidence intervals** for candidate support proportions
2. **Monte Carlo simulation** to validate CI coverage
3. **Data cleaning** with missing value removal
4. **Time series plots** of candidate support
5. **Proportion estimates** across all polls
6. **95% confidence intervals** for each candidate
7. **Spread calculation**: \( d = 2p - 1 \)
8. **Hypothesis testing** for significance of spread
9. **Results interpretation** and comparison with actual outcome

---

### Problem 3: Drug Safety Testing (Clinical Trial)

#### Experimental Design
- **Treatment Group (Drug)** vs **Control Group (Placebo)**
- Measured variables:
  - White Blood Cell count (WBC)
  - Red Blood Cell count (RBC)
  - Number of Adverse Effects

#### Statistical Tests Performed

```python
# Independent t-test
from scipy.stats import ttest_ind

# Compare drug group with placebo
t_stat, p_value = ttest_ind(drug_group, placebo_group, 
                             equal_var=False, 
                             alternative='two-sided')
```

#### Test Scenarios
1. **Two-sided** with α = 0.05
2. **One-sided (less)** with α = 0.05
3. **One-sided (greater)** with α = 0.05
4. **Two-sided** with α = 0.1 (higher significance level)

#### Key Results
- No significant differences in most variables
- RBC showed significance only with one-sided test at α=0.1
- Interpretation: Drug appears to be safe

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy pandas matplotlib scipy
```

### Run the Notebook
```bash
cd codes/
jupyter notebook notebook.ipynb
```

---

## 📈 Results & Findings

### Key Findings

#### 1. Roulette Simulation
- As number of rounds increases, results approach **expected value**
- Distribution of average winnings becomes **normal** with larger N
- Standard error decreases with \( \sqrt{N} \)
- Casino profit probability **increases** with N

#### 2. Election Analysis
- 95% confidence intervals calculated from polling data
- Monte Carlo results **matched** theoretical calculations
- Spread between candidates was **statistically significant**
- However, final outcome **differed** from predictions (sampling error)

#### 3. Drug Trial
- No significant differences in most metrics between drug and placebo
- This indicates **relative safety** of the drug
- Importance of choosing appropriate significance level

---

## 📚 Advanced Concepts Covered

### 1. Type I and Type II Errors
- **Type I Error (α)**: Rejecting true null hypothesis
- **Type II Error (β)**: Accepting false null hypothesis
- **Statistical Power**: \( 1 - β \)

### 2. Effect Size
```
Cohen's d = (μ₁ - μ₂) / σ
```

### 3. Confidence Intervals vs. Hypothesis Tests
- Confidence intervals: **Range of plausible values**
- Hypothesis tests: **Binary decision**

---

## 🎓 Key Takeaways

1. **Law of Large Numbers** guarantees convergence to truth with more data
2. **Central Limit Theorem** allows us to use normal distribution for inference
3. **Monte Carlo simulation** is a powerful tool for understanding statistical concepts
4. **p-value** is a decision metric, **not** a measure of effect size
5. **Confidence intervals** provide more information than hypothesis tests

---

## 🔍 Real-World Applications

- **Quality Control**: Product testing
- **Pharmaceuticals**: Clinical trials
- **Finance**: Risk assessment
- **Polling**: Public opinion analysis
- **A/B Testing**: Product optimization

---

## 📖 References & Resources

### Books
- *All of Statistics* by Larry Wasserman
- *Statistical Inference* by Casella & Berger
- *Probability and Statistics* by DeGroot & Schervish

### Courses
- MIT OpenCourseWare: Introduction to Probability and Statistics
- Stanford CS109: Probability for Computer Scientists

### Tools
- [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [NumPy Documentation](https://numpy.org/doc/)

---

## 👥 Team Members

- Mohammad Taha Majlesi - 810101504
- Mohammad Hossein Mazhari - 810101520
- Alireza Karimi - 810101492

---

## 📝 Important Notes

⚠️ **Key Points**:
- Always **set significance level** before analysis
- Report **confidence intervals** along with point estimates
- **Sample size** directly impacts estimation accuracy
- Check **test assumptions** (e.g., normality)

---

**Created**: Fall 2024-2025  
**Last Updated**: January 2025
