# Data Science Course - University of Tehran

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

This repository contains all coursework, projects, and materials for the Data Science course at the University of Tehran. The course covers comprehensive topics from statistical foundations to advanced machine learning and deep learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Course Materials](#course-materials)
- [Projects & Assignments](#projects--assignments)
  - [CA0: Introduction to Data Science](#ca0-introduction-to-data-science)
  - [CA1: Statistical Analysis](#ca1-statistical-analysis)
  - [CA3: Recommender Systems](#ca3-recommender-systems)
  - [CA4: Advanced Topics](#ca4-advanced-topics)
  - [CA5&6: Natural Language Processing](#ca56-natural-language-processing)
  - [Final Project: NYC Taxi Demand Prediction](#final-project-nyc-taxi-demand-prediction)
- [Technologies & Tools](#technologies--tools)
- [Installation & Setup](#installation--setup)
- [Team Members](#team-members)
- [License](#license)

---

## 🎯 Overview

This Data Science course provides a comprehensive journey through:

- **Statistical Foundations**: Probability theory, inference, and hypothesis testing
- **Data Visualization**: Statistical charts, design principles, and storytelling with data
- **Machine Learning**: Supervised and unsupervised learning algorithms
- **Deep Learning**: Neural Networks, CNNs, RNNs, and Transformers
- **Natural Language Processing**: Language models and text analysis
- **Real-world Applications**: End-to-end data science projects with CI/CD pipelines

---

## 📁 Repository Structure

```
DS_UT/
├── cheetsheet/                 # Quick reference guides and cheat sheets
│   └── *.pdf                  # Course cheat sheets
│
├── Materials/                  # All lecture slides and course materials
│   ├── Lecture 02-14/         # Complete lecture series
│   └── Sample Questions       # Final exam preparation
│
├── projects/                   # All course assignments and projects
│   ├── CA0/                   # Assignment 0: Introduction
│   ├── CA1/                   # Assignment 1: Statistical Analysis
│   ├── CA3/                   # Assignment 3: Recommender Systems
│   ├── CA4/                   # Assignment 4: Advanced ML
│   ├── CA56/                  # Assignment 5&6: NLP
│   └── phase las/             # Final Project: NYC Taxi Prediction
│       └── Data_Science_Phase2_Project_CI-CD/
│
└── README.md                  # This file
```

---

## 📚 Course Materials

The `Materials/` directory contains comprehensive lecture slides covering:

### Core Topics

1. **Statistical Foundations**

   - Lecture 02: Statistical Charts
   - Lecture 03: Probability Theory Review
   - Lecture 04: Foundations for Inference & Visualization Design

2. **Machine Learning Fundamentals**

   - Lecture 05: Linear Regression, Dashboards & Storytelling
   - Lecture 06: SQL (Parts 1 & 2)
   - Lecture 07: Data Preprocessing
   - Lecture 08: Gradient Descent & Logistic Regression

3. **Advanced Machine Learning**

   - Lecture 09-01: Sklearn & Feature Engineering
   - Lecture 09-02: Logistic Regression (Advanced)
   - Lecture 09-03: Cross-Validation & Regularization
   - Lecture 09-04: SVM & KNN
   - Lecture 09-05: Decision Trees & Random Forests

4. **Deep Learning**

   - Lecture 10: Neural Networks & CNNs
   - Lecture 11: RNNs & NLP Fundamentals
   - Lecture 12: Language Models

5. **Advanced Topics**
   - Lecture 13: Unsupervised Learning
   - Lecture 14: Data Science Applications

---

## 🚀 Projects & Assignments

### CA0: Introduction to Data Science

**Location**: `projects/CA0/`

Initial assignment introducing basic data science concepts and Python programming fundamentals.

**Topics Covered**:

- Python basics
- Data structures
- Introduction to pandas and numpy

---

### CA1: Statistical Analysis

**Location**: `projects/CA1/`

**Team Members**: 810101504, 810101492, 810101520

Statistical analysis project focusing on:

- Exploratory Data Analysis (EDA)
- Statistical hypothesis testing
- Data visualization
- Probability distributions

---

### CA3: Recommender Systems

**Location**: `projects/CA3/`

**Description**: Implementation of an ensemble-based movie recommendation system using collaborative filtering.

**Key Features**:

- **Algorithms Implemented**:
  - SVD (Singular Value Decomposition)
  - SVD++ (Enhanced SVD)
  - KNN-Baseline (K-Nearest Neighbors)
- **Advanced Techniques**:
  - Grid search hyperparameter tuning
  - Ensemble learning with bagging
  - Linear regression blending
  - Model stacking

**Main File**: `Q3.py`

**Metrics**:

- RMSE (Root Mean Squared Error)
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

**Usage**:

```bash
python Q3.py --data_dir ./dataset/
```

**Performance Optimization**:

- Cross-validation with 3 folds
- Parallel processing for grid search
- Weighted ensemble predictions
- Rating clipping for boundary constraints

---

### CA4: Advanced Topics

**Location**: `projects/CA4/`

Advanced machine learning assignment exploring complex algorithms and techniques.

---

### CA5&6: Natural Language Processing

**Location**: `projects/CA56/`

**Description**: Persian Question-Answering system using NLP techniques.

**Contents**:

- `labeled-data.csv`: Training dataset with labeled examples
- `unlabeled-data.csv`: Test dataset for predictions
- `PerCQA_JSON_Format.json`: Persian conversational QA format
- `DS_CA56 (1).ipynb`: Implementation notebook

**Topics**:

- Text preprocessing for Persian language
- Question answering systems
- Natural language understanding
- Model evaluation and validation

**Key Challenges**:

- Persian language processing
- Context understanding
- Answer extraction

---

### Final Project: NYC Taxi Demand Prediction

**Location**: `projects/phase las/Data_Science_Phase2_Project_CI-CD/`

**Team Members**: 810101504, 810101492, 810101520

A comprehensive end-to-end machine learning project predicting taxi demand in New York City with complete CI/CD pipeline integration.

#### 🎯 Project Objectives

Predict taxi demand patterns based on:

- **Location**: Geographic zones in NYC
- **Time**: Peak hours and time-based patterns
- **Weather**: Weather conditions and their impact
- **Historical Trends**: Past demand patterns

#### 📊 Project Components

**1. Data Pipeline** (`pipeline.py`)

- Automated data loading from multiple sources
- Data preprocessing and cleaning
- Feature engineering
- Performance monitoring and logging

**2. Database** (`database/`)

- MySQL database schema
- Taxi zone coordinates
- Weather data integration
- Efficient data storage and retrieval

**3. Machine Learning Models** (`models/`)

Multiple model configurations optimized for different scenarios:

- **Base Performance Model**
  - General demand prediction
  - Comprehensive feature set
- **Location-Based Model**
  - Zone-specific demand patterns
  - Geographic feature engineering
- **Peak Time Model**
  - Time-series analysis
  - Rush hour predictions
- **Weather-Dependent Model**

  - Weather impact analysis
  - Conditional demand forecasting

- **Neural Network Model**
  - Deep learning approach
  - Non-linear pattern recognition

**Model Files**:

```
models/
├── *_best_model.joblib          # Trained models
├── *_encoder.joblib              # Feature encoders
├── *_scaler.joblib/.pkl          # Data scalers
├── model_metadata.json           # Model configurations
└── model_performance_comparison.json  # Evaluation results
```

**4. Scripts** (`scripts/`)

Modular Python scripts for:

- `database_connection.py`: Database connectivity
- `load_data.py`: Data ingestion
- `preprocess.py`: Data cleaning and transformation
- `feature_engineering.py`: Feature creation
- `seed_database.py`: Database initialization

**5. Notebooks**

- `Phase3.ipynb`: Main analysis and model development
- `Weather_Complete_Report_Final.ipynb`: Weather data analysis
- `clean.ipynb`: Data cleaning procedures
- `import.ipynb`: Data import processes
- `scrapping.ipynb`: Web scraping for additional data

**6. Visualizations** (`visualizations/`, `weather_visualizations/`)

- Demand heatmaps
- Time-series plots
- Weather correlation charts
- Geographic distribution maps
- Model performance comparisons

**7. SQL Queries** (`Query/`)

- Complex analytical queries
- Data aggregation scripts
- Performance optimization queries

#### 🐳 Docker Deployment

Complete containerization setup:

**Files**:

- `Dockerfile`: Container configuration
- `docker-compose.yml`: Multi-container orchestration
- `docker-entrypoint.sh`: Initialization script

**Quick Start**:

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

#### 📦 Dependencies

**Core Libraries** (`requirements.txt`):

```
pandas              # Data manipulation
sqlalchemy         # Database ORM
pymysql            # MySQL connector
scikit-learn       # Machine learning
matplotlib         # Plotting
seaborn            # Statistical visualizations
numpy              # Numerical computing
cryptography       # Secure connections
mysql-connector-python  # MySQL driver
```

#### 🔄 CI/CD Pipeline

Automated workflow featuring:

- **Continuous Integration**:
  - Automated testing
  - Code quality checks
  - Dependency management
- **Continuous Deployment**:
  - Docker image building
  - Automated deployments
  - Environment configuration

#### 📈 Model Performance

The project includes comprehensive model evaluation:

- Cross-validation metrics
- Hold-out test performance
- Comparison across different approaches
- Performance visualization

**Evaluation Metrics**:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

#### 🗺️ Data Sources

1. **NYC Taxi Trip Data**

   - Pickup/dropoff locations
   - Trip timestamps
   - Trip distances

2. **Weather Data** (Scraped)

   - Temperature
   - Precipitation
   - Wind conditions
   - Visibility

3. **Geographic Data**
   - Taxi zone boundaries
   - Zone coordinates
   - Borough information

#### 🚀 Running the Pipeline

```bash
# Setup environment
pip install -r requirements.txt

# Run complete pipeline
python pipeline.py

# Or run individual steps
python scripts/load_data.py
python scripts/preprocess.py
python scripts/feature_engineering.py
```

#### 📊 Key Insights

The project demonstrates:

- Weather significantly impacts demand (especially precipitation)
- Clear peak hours: morning (7-9 AM) and evening (5-7 PM) rushes
- Geographic patterns: Manhattan shows highest demand
- Seasonal variations in demand patterns
- Holiday effects on taxi usage

#### 🎓 Learning Outcomes

This project showcases:

- End-to-end ML pipeline development
- Real-world data challenges and solutions
- Model selection and optimization
- Production-ready code practices
- Docker containerization
- CI/CD implementation
- Database design and management
- Data visualization best practices

---

## 🛠️ Technologies & Tools

### Programming Languages

- **Python 3.8+**: Primary language for data analysis and ML

### Data Science Libraries

- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Surprise**: Recommender systems

### Visualization

- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical visualizations

### Deep Learning (if applicable)

- **TensorFlow/PyTorch**: Neural networks
- **Keras**: High-level neural networks API

### Database

- **SQL**: Data querying
- **MySQL**: Relational database
- **SQLAlchemy**: Python SQL toolkit

### Development Tools

- **Jupyter Notebook**: Interactive development
- **Docker**: Containerization
- **Git**: Version control
- **GitHub Actions**: CI/CD

### Web Scraping

- **BeautifulSoup/Scrapy**: Data collection

---

## 💻 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook
- Git
- Docker (for final project)

### Basic Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd DS
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

For general coursework:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

For CA3 (Recommender Systems):

```bash
pip install scikit-surprise
```

For Final Project:

```bash
cd projects/phase\ las/Data_Science_Phase2_Project_CI-CD/
pip install -r requirements.txt
```

### Running Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to the desired `.ipynb` file and run the cells.

### Docker Setup (Final Project)

```bash
cd projects/phase\ las/Data_Science_Phase2_Project_CI-CD/
docker-compose up --build
```

---

## 👥 Team Members

**Student IDs**:

- 810101504
- 810101492
- 810101520

---

## 📝 Assignment Submission Guidelines

Each assignment follows the naming convention:

```
DS_CA{NUMBER}_{ID1}_{ID2}_{ID3}.zip
```

---

## 🎓 Course Information

**Institution**: University of Tehran  
**Course**: Data Science  
**Academic Year**: 2024-2025  
**Semester**: Fall 2024

---

## 📖 Additional Resources

### Cheat Sheets

The `cheetsheet/` directory contains quick reference guides for:

- Python syntax
- Pandas operations
- Machine learning algorithms
- Statistical methods

### Documentation

- Course lecture slides in `Materials/`
- Project documentation in respective folders
- Code comments and docstrings

---

## 🔍 Project Highlights

### Key Achievements

1. **Comprehensive Coverage**: From basic statistics to advanced deep learning
2. **Real-world Applications**: NYC taxi demand prediction with production-ready code
3. **Best Practices**:

   - Modular code design
   - Comprehensive documentation
   - Version control
   - CI/CD pipelines
   - Docker containerization

4. **Advanced Techniques**:
   - Ensemble learning
   - Feature engineering
   - Hyperparameter tuning
   - Model evaluation and selection
   - Time series analysis

---

## 🚀 Future Enhancements

Potential improvements and extensions:

- [ ] Real-time prediction API
- [ ] Interactive dashboard for visualizations
- [ ] Mobile application integration
- [ ] Additional data sources integration
- [ ] Advanced deep learning models
- [ ] A/B testing framework
- [ ] Automated model retraining

---

## 📞 Contact & Support

For questions or collaboration opportunities:

- Repository Issues: Use GitHub Issues
- Team Communication: [Contact through university channels]

---

## 📄 License

This project is part of academic coursework at the University of Tehran. All rights reserved for educational purposes.

---

## 🙏 Acknowledgments

- **Instructors**: University of Tehran Data Science Faculty
- **Teaching Assistants**: For guidance and support
- **Open Source Community**: For libraries and tools used in this course

---

## 📊 Repository Statistics

- **Total Assignments**: 6 (CA0, CA1, CA3, CA4, CA5&6, Final Project)
- **Programming Languages**: Python, SQL
- **Total Notebooks**: 8+
- **Lecture Materials**: 14 lectures
- **Team Size**: 3 members

---

## 🔄 Last Updated

October 10, 2025

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ by the DS Team**

</div>
