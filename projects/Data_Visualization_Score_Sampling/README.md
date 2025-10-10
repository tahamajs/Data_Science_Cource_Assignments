# Assignment 1: Data Visualization & Score-Based Sampling

---

## 📊 Project Overview

This project explores advanced **sampling techniques** with a focus on **Langevin Dynamics** and **Score-Based Sampling**. The main objective is to understand how to sample from complex probability distributions using the gradient of the log-density function (Score Function).

---

## 🎯 Learning Objectives

### Part 1: Fundamentals

- Understand **Score Function**: \( \nabla_x \log p(x) \)
- Calculate gradient field for Gaussian distributions
- Visualize Score Function vector fields

### Part 2: Langevin Dynamics

- Implement **Unadjusted Langevin Algorithm (ULA)**
- Sample from 2D Gaussian distributions
- Compare with direct methods (NumPy)

### Part 3: Gaussian Mixture Models (GMM)

- Calculate Score Function for mixture distributions
- Sample from multi-modal distributions
- Qualitative and quantitative analysis

### Part 4: Airbnb Data Analysis

- Data processing and cleaning
- Price distribution visualization
- Geographic distribution analysis

---

## 🔬 Core Concepts & Techniques

### 1. Score Function

**Definition**: Gradient of log probability density function

```python
score(x) = ∇_x log p(x) = ∇_x p(x) / p(x)
```

**For Gaussian Distribution**:

```python
score(x) = -Σ⁻¹(x - μ)
```

**Applications**:

- Advanced sampling algorithms
- Generative models
- Score matching

---

### 2. Langevin Dynamics

**Update Formula**:

```
x_{t+1} = x_t + ε·∇_x log p(x_t) + √(2ε)·η
```

where:

- **ε**: step size
- **η**: Gaussian noise ~ N(0, I)

**Advantages**:

- No need for normalized density
- Only requires Score Function
- Efficient for complex distributions

**Key Parameters**:

- **Number of steps**: Balance speed vs accuracy
- **Step size**: Smaller = more accurate but slower

---

### 3. Gaussian Mixture Model (GMM)

**Definition**:

```
p(x) = α·N(x; μ₁, Σ₁) + (1-α)·N(x; μ₂, Σ₂)
```

**Score Function for GMM**:

```python
∇_x log p(x) = [α·p₁(x)·∇log p₁(x) + (1-α)·p₂(x)·∇log p₂(x)] / p(x)
```

**Challenges**:

- Sampling from multi-modal distributions
- Mode hopping between clusters
- Parameter tuning for exploring all modes

---

## 📁 Project Structure

```
Data_Visualization_Score_Sampling/
├── code/
│   └── code.ipynb              # Main project notebook
├── dataset/
│   ├── Airbnb_Listings.xlsx    # Listing data
│   └── Neighborhood_Locations.xlsx  # Geographic coordinates
├── description/
│   └── CA1.pdf                 # Assignment description
├── note/
│   └── CA1.pdf                 # Notes and grades
└── README.md                   # This file
```

---

## 🛠️ Technologies & Libraries

### Python Libraries

```python
# Numerical & Statistical Computing
numpy                 # Numerical operations
scipy.stats          # Statistical distributions
sklearn              # KMeans clustering

# Visualization
matplotlib           # Basic plotting
seaborn              # Advanced statistical plots

# Data Processing
pandas               # DataFrame manipulation
```

### Implemented Algorithms

1. **Score Computation**

   ```python
   def compute_score(point, mu, inv_cov):
       return -inv_cov @ (point - mu)
   ```

2. **Langevin Sampler**

   ```python
   def run_langevin(init_pts, score_fn, num_steps, step_size):
       for t in range(num_steps):
           scores = score_fn(current_pts)
           noise = np.random.randn() * sqrt(2*step)
           current_pts += step * scores + noise
       return current_pts
   ```

3. **GMM Score Function**
   ```python
   def mixture_grad(x, mu1, mu2, Sigma1, Sigma2, alpha):
       grad1 = -inv_Sigma1 @ (x - mu1)
       grad2 = -inv_Sigma2 @ (x - mu2)
       pdf1 = multivariate_normal(mu1, Sigma1).pdf(x)
       pdf2 = multivariate_normal(mu2, Sigma2).pdf(x)
       numerator = alpha*pdf1*grad1 + (1-alpha)*pdf2*grad2
       denominator = alpha*pdf1 + (1-alpha)*pdf2
       return numerator / denominator
   ```

---

## 📊 Tasks & Analysis

### Task 1: Gaussian Distribution Sampling

#### 1.1: Score Field Calculation & Visualization

- Create 2D grid for Score Function evaluation
- Calculate gradient at each point
- Plot contour plot with quiver (vector field)

**Result**:

- Vectors point towards distribution mean
- Vector magnitude increases with distance from mean

#### 1.2: Langevin Dynamics Implementation

- `run_langevin` function with tunable parameters
- Start from random points
- Iterative update using Score + noise

#### 1.3: Sampling Trajectory Visualization

- Plot trajectory from start to convergence
- Show step-by-step movement towards high-density region

#### 1.4: Precise vs Fast Langevin Comparison

**Precise Settings**:

- Step size: 0.05
- Number of steps: 5000

**Fast Settings**:

- Step size: 0.5
- Number of steps: 20

**Evaluation Metrics**:

```python
# Mean comparison
mean_numpy = np.mean(numpy_samples, axis=0)
mean_precise = np.mean(langevin_precise, axis=0)
mean_fast = np.mean(langevin_fast, axis=0)

# Covariance comparison
cov_numpy = np.cov(numpy_samples.T)
cov_precise = np.cov(langevin_precise.T)

# Wasserstein distance
w_distance_x = wasserstein_distance(numpy_samples[:, 0],
                                     langevin_samples[:, 0])

# Kolmogorov-Smirnov test
ks_stat, p_value = ks_2samp(numpy_samples[:, 0],
                             langevin_samples[:, 0])
```

**Results**:

- **Precise Langevin**:
  - Mean very close to ground truth
  - p-value > 0.05 → Similar distributions
  - Low Wasserstein distance
- **Fast Langevin**:
  - Higher deviation from mean
  - p-value ≈ 0 → Significant difference
  - Higher variance

---

### Task 1 - Question 5 (Bonus): GMM Sampling

#### Theoretical Analysis of GMM Score Function

**Main Challenge**: Score Function of a mixture is a weighted combination of component scores:

```
∇log p(x) = [α·p₁(x)·∇log p₁(x) + (1-α)·p₂(x)·∇log p₂(x)] / p(x)
```

**Key Properties**:

- Near each mode, that mode's score dominates
- In between regions, weights determined by density ratios
- Mode hopping requires sufficient noise

#### Implementation & Testing

**GMM Settings**:

```python
mu_A = [-5, 5]
mu_B = [5, -5]
Sigma_A = Sigma_B = 5*I
mixing_coefficient = 0.5
```

**Langevin Parameters**:

- Number of samples: 100
- Number of steps: 1000
- Step size: 0.05

#### Quantitative Evaluation

**Mean Comparison**:

```
Ground Truth Mode 1: [-5, 5]
Ground Truth Mode 2: [5, -5]
Langevin Mean: [0.34, 0.32]
Theoretical Mixture Mean: [0, 0]
```

**Covariance Analysis**:

```
Per-Mode Covariance: [[5, 0], [0, 5]]
Langevin Covariance: [[29.9, -23.3], [-23.3, 25.8]]
```

- Higher variance indicates spread between two modes
- Negative correlation shows diagonal orientation

**Distance Metrics**:

```
Wasserstein Distance:
  X-axis: 0.606
  Y-axis: 0.705

Kolmogorov-Smirnov Test:
  X-axis p-value: 0.815
  Y-axis p-value: 0.815
```

- p-value > 0.05 → No significant difference in marginals

#### Visualization Results

1. **Scatter Plot**: Display samples in 2D space
2. **Histogram Comparison**: Compare marginal distributions
3. **KDE Plot**: Non-parametric density estimation

---

### Task 2: Airbnb Data Analysis

#### Objective

Analyze and visualize Airbnb listing data, exploring price distributions across neighborhoods.

#### Datasets

1. **Airbnb_Listings.xlsx**: Listing information
   - Name, price, neighborhood, ratings, amenities
2. **Neighborhood_Locations.xlsx**: Geographic coordinates

#### Analyses Performed

**1. Data Processing**:

```python
# Clean prices
df['price_clean'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)

# Remove outliers
Q1 = df['price_clean'].quantile(0.25)
Q3 = df['price_clean'].quantile(0.75)
IQR = Q3 - Q1
df_filtered = df[(df['price_clean'] >= Q1 - 1.5*IQR) &
                  (df['price_clean'] <= Q3 + 1.5*IQR)]
```

**2. Descriptive Statistics**:

- Mean, median, standard deviation of price
- Distribution by property type
- Distribution by neighborhood

**3. Visualizations**:

- Price distribution histogram
- Box plot of price by neighborhood
- Geographic scatter plot
- Correlation heatmap

---

## 🚀 How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn openpyxl
```

### Run the Notebook

```bash
cd code/
jupyter notebook code.ipynb
```

---

## 📈 Results & Findings

### Key Findings - Sampling

#### 1. Langevin Dynamics is Effective

- With proper tuning, produces accurate samples
- Only requires Score Function (not normalized density)

#### 2. Speed-Accuracy Trade-off

| Method  | Speed      | Accuracy   | Use Case               |
| ------- | ---------- | ---------- | ---------------------- |
| Precise | ⭐⭐       | ⭐⭐⭐⭐⭐ | Research, benchmarking |
| Fast    | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | Rapid generation       |

#### 3. GMM Sampling is Challenging

- Requires careful parameter tuning
- Risk of getting stuck in one mode
- Needs more steps for full exploration

#### 4. Quantitative Evaluation is Essential

- Visualization necessary but insufficient
- Statistical metrics: Wasserstein, KS-test
- Mean and covariance comparison

---

### Key Findings - Airbnb

#### Price Distribution

- Right-skewed distribution
- Significant outliers present
- Meaningful differences across neighborhoods

#### Price Factors

1. **Geographic location** (most important)
2. Property type
3. User ratings
4. Number of amenities

---

## 📚 Advanced Concepts

### 1. Score-Based Generative Models

**Main Idea**: Model Score Function instead of density itself

```python
s_θ(x) ≈ ∇_x log p(x)
```

**Advantages**:

- No normalization needed
- Scalable to complex models
- High-quality sample generation

**Applications**:

- Image generation
- Audio and music generation
- Molecular design

---

### 2. Annealed Langevin Dynamics

**Problem**: Multi-modal distributions

**Solution**: Use sequence of smoothed distributions

```
p_σ(x) ∝ p(x) * exp(-||x||²/(2σ²))
```

**Process**:

1. Start with large σ (smooth distribution)
2. Gradually decrease σ
3. Converge to original distribution

---

### 3. Metropolis-Adjusted Langevin Algorithm (MALA)

**Improvement over ULA**: Add Metropolis-Hastings acceptance step

```python
# Calculate acceptance ratio
alpha = min(1, p(x_prop)/p(x_curr) * ...transition ratio...)

# Accept or reject
if random.uniform(0,1) < alpha:
    x_curr = x_prop
```

**Benefit**: Exact convergence to target distribution

---

## 🎓 Key Takeaways

### Theoretical Lessons

1. **Score Function** is powerful for sampling
2. **Langevin Dynamics** bridges physics and probability
3. **GMM** is good example of multi-modal distributions
4. **Wasserstein Distance** is meaningful metric for comparing distributions

### Practical Lessons

1. **Parameter tuning** is critical (step size, iterations)
2. **Quantitative evaluation** always necessary
3. **Visualization** aids understanding
4. **Baseline comparison** (e.g., NumPy) important

---

## 🔍 Real-World Applications

### Generative Models

- **Denoising Diffusion Models**: DALL-E, Stable Diffusion
- **Score-Based Models**: Various NCSN variants

### Advanced MCMC

- **Bayesian sampling**: Parameter inference
- **Computational physics**: Molecular simulation

### Business Data Visualization

- **Pricing analysis**: Airbnb, Hotels
- **Market analysis**: Real Estate
- **CRM**: Customer pattern identification

---

## 🐛 Common Issues & Solutions

### Issue 1: Slow Convergence

**Cause**: Step size too small  
**Solution**: Increase ε or use annealing

### Issue 2: High Oscillation

**Cause**: Step size too large  
**Solution**: Decrease ε or increase iterations

### Issue 3: Stuck in One Mode

**Cause**: Unable to hop between modes  
**Solution**:

- Increase noise
- Use Parallel Tempering
- Restart from different points

### Issue 4: Heavy Computation

**Solutions**:

- Use GPU acceleration
- Code optimization (vectorization)
- Reduce number of samples

---

## 📖 References & Resources

### Key Papers

1. **Score-Based Generative Modeling through SDEs**  
   Yang Song et al., ICLR 2021

2. **Denoising Diffusion Probabilistic Models**  
   Ho et al., NeurIPS 2020

3. **Generative Modeling by Estimating Gradients**  
   Song & Ermon, NeurIPS 2019

### Books

- _Pattern Recognition and Machine Learning_ by Bishop
- _Monte Carlo Statistical Methods_ by Robert & Casella

### Courses

- Stanford CS236: Deep Generative Models
- MIT 6.S897: Machine Learning for Healthcare

### Reference Implementations

- [Score-Based Generative Models (Official)](https://github.com/yang-song/score_sde)
- [PyTorch Diffusion Models](https://github.com/lucidrains/denoising-diffusion-pytorch)

---

## 👥 Team Members

- **Mohammad Taha Majlesi** - 810101504
- **Mohammad Hossein Mazhari** - 810101520
- **Alireza Karimi** - 810101492

---

## 📧 Contact & Support

### Technical Questions

Contact course TAs for implementation questions.

### Theoretical Questions

Consult instructors for conceptual clarification.

**Instructors**: Dr. Bahrak, Dr. Yaghoobzadeh  
**TAs**: Mohammad Reza Alavi, Mohammad Kavian, Fatemeh Mohammadi

---

**Created**: Fall 2024-2025  
**Last Updated**: January 2025

---

> **Note**: This project provides foundation for understanding modern generative models like Stable Diffusion and DALL-E. Mastering these concepts enables working with more complex models.
