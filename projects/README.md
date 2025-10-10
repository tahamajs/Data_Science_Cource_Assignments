# Data Science Projects - University of Tehran

This repository contains all assignments and projects from the Data Science course at University of Tehran.

---

## 📚 Projects Overview

### [CA0: Statistical Inference & Monte Carlo Simulation](./CA0_Statistical_Inference_Monte_Carlo/)
**Topics**: Probability Theory, Hypothesis Testing, Confidence Intervals  
**Concepts**:
- Monte Carlo simulation for roulette games
- Electoral polling data analysis (2016 US Election)
- Clinical trial statistical testing
- Law of Large Numbers and Central Limit Theorem

---

### [CA1: Data Visualization & Score-Based Sampling](./CA1_Data_Visualization_Score_Sampling/)
**Topics**: Advanced Sampling, Langevin Dynamics, Data Visualization  
**Concepts**:
- Score Function computation
- Langevin Dynamics sampling
- Gaussian Mixture Models
- Airbnb data analysis and visualization

---

### [CA2: Real-Time Data Streaming with Kafka](./CA2_Real_Time_Streaming_Kafka/)
**Topics**: Distributed Systems, Event-Driven Architecture  
**Concepts**:
- Apache Kafka producer implementation
- Real-time transaction event generation
- Poisson process for realistic event arrivals
- Stream processing patterns

---

### [CA3: Advanced Machine Learning](./CA3_Advanced_ML_Regression_RecSys/)
**Topics**: Regression, Feature Engineering, Recommender Systems  
**Concepts**:
- **Q1**: Bike-sharing demand prediction with ensemble methods
- **Q2**: Movie recommender system using collaborative filtering
- **Q3**: Statistical visualization and data analysis

---

### [CA4: Deep Learning & Neural Networks](./CA4_Deep_Learning_Neural_Networks/)
**Topics**: Neural Network Architectures  
**Concepts**:
- **Task 1**: Multi-Layer Perceptrons (MLP) for classification
- **Task 2**: Convolutional Neural Networks (CNN) for images
- **Task 3**: Recurrent Neural Networks (RNN/LSTM) for sequences

---

### [CA56: NLP & Semi-Supervised Learning](./CA56_NLP_Semi_Supervised_Learning/)
**Topics**: Natural Language Processing, Limited Labeled Data  
**Concepts**:
- Text vectorization (SentenceTransformers, Word2Vec)
- Supervised learning baselines
- Pseudo-labeling and self-training
- Active learning strategies

---

## 🛠️ Technologies Used

### Programming Languages & Core Libraries
- **Python 3.8+**
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn

### Machine Learning
- scikit-learn
- LightGBM, XGBoost
- PyTorch, TensorFlow

### Specialized Tools
- Apache Kafka (Confluent)
- Gensim (Word2Vec)
- SentenceTransformers
- Surprise (Recommender Systems)

---

## 📊 Skills Developed

### Statistical Analysis
- Hypothesis testing (t-tests, ANOVA)
- Confidence intervals and p-values
- Monte Carlo simulation
- Probability distributions

### Machine Learning
- **Supervised Learning**: Regression, Classification
- **Unsupervised Learning**: Clustering, Dimensionality Reduction
- **Semi-Supervised Learning**: Pseudo-labeling, Active Learning
- **Deep Learning**: MLP, CNN, RNN/LSTM
- **Ensemble Methods**: Stacking, Bagging, Boosting

### Data Engineering
- Real-time streaming pipelines
- Event-driven architecture
- Data preprocessing and feature engineering
- ETL processes

### Advanced Topics
- Score-based sampling
- Langevin dynamics
- Collaborative filtering
- Transfer learning
- Time-series forecasting

---

## 📁 Repository Structure

```
DS/
├── projects/
│   ├── CA0_Statistical_Inference_Monte_Carlo/
│   │   ├── codes/
│   │   ├── datasets/
│   │   ├── description/
│   │   └── README.md
│   ├── CA1_Data_Visualization_Score_Sampling/
│   │   ├── code/
│   │   ├── dataset/
│   │   ├── description/
│   │   └── README.md
│   ├── CA2_Real_Time_Streaming_Kafka/
│   │   ├── codes/
│   │   ├── description/
│   │   └── README.md
│   ├── CA3_Advanced_ML_Regression_RecSys/
│   │   ├── codes/
│   │   ├── descriptions/
│   │   ├── reports/
│   │   └── README.md
│   ├── CA4_Deep_Learning_Neural_Networks/
│   │   ├── codes/
│   │   │   ├── Task1/
│   │   │   ├── Task2/
│   │   │   └── Task3/
│   │   ├── datasets/
│   │   ├── description/
│   │   └── README.md
│   ├── CA56_NLP_Semi_Supervised_Learning/
│   │   ├── code/
│   │   ├── datasets/
│   │   ├── description/
│   │   └── README.md
│   └── Data_Science_Project/
│       └── [Final project]
├── Materials/
│   └── [Lecture slides and resources]
├── cheatsheet/
│   └── [Reference materials]
└── README.md (this file)
```

---

## 🚀 How to Use This Repository

### Clone the Repository
```bash
git clone <repository-url>
cd DS/projects
```

### Install Dependencies

Each project has its own requirements. Generally:

```bash
# Core dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn

# For specific projects:
# CA2: Apache Kafka
pip install confluent-kafka

# CA4: Deep Learning
pip install torch torchvision

# CA56: NLP
pip install sentence-transformers gensim nltk
```

### Run Individual Projects

Navigate to each project's directory and follow the instructions in its README file.

---

## 📖 Learning Path

Recommended order to explore these projects:

1. **CA0** - Build statistical foundations
2. **CA1** - Understand advanced sampling and visualization
3. **CA3** - Learn ML pipelines and feature engineering
4. **CA4** - Deep dive into neural networks
5. **CA56** - Apply advanced NLP and SSL techniques
6. **CA2** - Understand production systems

---

## 👥 Course Information

**Course**: Data Science  
**University**: University of Tehran  
**Semester**: Fall 2024-2025  
**Instructors**: Dr. Bahrak, Dr. Yaghoobzadeh

---

## 📧 Contact

For questions about specific projects, refer to the individual README files in each project directory.

---

**Last Updated**: January 2025

