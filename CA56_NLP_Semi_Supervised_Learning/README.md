# Assignment 5 & 6: NLP & Semi-Supervised Learning
## Text Classification with Limited Labeled Data

---

## 📊 Project Overview

This comprehensive assignment explores **Natural Language Processing (NLP)** and **Semi-Supervised Learning (SSL)** techniques for text classification with limited labeled data. The project focuses on predicting video game review scores from textual summaries using both traditional and advanced machine learning approaches.

### Key Challenge
**How to leverage large amounts of unlabeled text data when labeled data is scarce?**

This is a common real-world scenario where obtaining labels is expensive or time-consuming.

---

## 🎯 Learning Objectives

### Part 1: Text Vectorization
- Transform text into numerical representations
- Implement **SentenceTransformers** for dense embeddings
- Train custom **Word2Vec** models
- Compare embedding strategies

### Part 2: Supervised Learning Baselines
- Build classification and regression models
- Implement proper train/validation/test splits
- Evaluate multiple algorithms (LR, SVM, LightGBM)
- Establish baseline performance

### Part 3: Semi-Supervised Learning
- **Pseudo-Labeling**: Use model predictions to expand training set
- **Active Learning**: Strategically select samples for labeling
- **Co-Training**: Leverage multiple views of data
- Compare SSL techniques with supervised baselines

---

## 📁 Project Structure

```
NLP_Semi_Supervised_Learning/
├── code/
│   └── DS_CA56_final.ipynb     # Main project notebook
├── datasets/
│   ├── labeled-data.csv        # Labeled reviews with scores
│   ├── unlabeled-data.csv      # Unlabeled reviews
│   └── PerCQA_JSON_Format.json # Additional dataset
├── description/
│   └── CA5&6.pdf               # Assignment description
└── README.md                   # This file
```

---

## 🔬 Core Concepts & Techniques

### 1. Text Vectorization

#### A. SentenceTransformers (Pre-trained Embeddings)

**Advantages**:
- Pre-trained on massive corpora
- Captures semantic meaning
- Ready to use, no training needed
- Dense representations (384-768 dimensions)

**Implementation**:
```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = ["Great game!", "Terrible experience"]
embeddings = model.encode(texts)

# Shape: (n_samples, 384)
print(embeddings.shape)
```

**Model Details**:
- **all-MiniLM-L6-v2**: 
  - Embedding dimension: 384
  - Speed: ~14,000 sentences/sec
  - Quality: High semantic similarity capture

---

#### B. Word2Vec (Custom-trained Embeddings)

**Advantages**:
- Domain-specific embeddings
- Captures domain vocabulary
- Controllable embedding size
- Can handle out-of-vocabulary words (to some extent)

**Implementation**:
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Tokenize texts
tokenized_texts = [word_tokenize(text.lower()) for text in all_texts]

# Train Word2Vec
w2v_model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=100,        # Embedding dimension
    window=5,               # Context window
    min_count=2,            # Ignore rare words
    workers=4,              # Parallel training
    sg=1,                   # Skip-gram (vs CBOW)
    epochs=10
)

# Get sentence embedding by averaging word vectors
def get_sentence_vector(sentence, model):
    words = word_tokenize(sentence.lower())
    vectors = [model.wv[word] for word in words if word in model.wv]
    
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Generate embeddings
embeddings = np.array([get_sentence_vector(text, w2v_model) 
                       for text in texts])
```

**Hyperparameters**:
- `vector_size`: Embedding dimension (50-300 typical)
- `window`: Context window size (3-10 typical)
- `sg`: Skip-gram (1) or CBOW (0)
- `min_count`: Minimum word frequency

---

#### C. Comparison: SentenceTransformers vs Word2Vec

| Aspect | SentenceTransformers | Word2Vec |
|--------|---------------------|----------|
| **Training Time** | None (pre-trained) | Minutes to hours |
| **Data Required** | None | Large corpus preferred |
| **Semantic Quality** | High (pre-trained on billions of texts) | Varies (domain-dependent) |
| **Embedding Size** | Fixed (384-768) | Configurable (50-300) |
| **OOV Handling** | Better (subword tokenization) | Limited |
| **Domain Specificity** | General purpose | Highly specific |

---

### 2. Supervised Learning Baselines

#### A. Data Splitting Strategy

**80-10-10 Split with Stratification**:
```python
from sklearn.model_selection import train_test_split

# First split: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,           # Maintain class distribution
    random_state=42
)

# Second split: 80% train, 20% val (of temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125,      # 0.125 * 0.8 = 0.1 of total
    stratify=y_temp,
    random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

**Why Stratification?**
- Maintains class distribution in all splits
- Critical for imbalanced datasets
- Ensures representative validation/test sets

---

#### B. Classification Models

**Logistic Regression**:
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,                # Regularization strength
    random_state=42
)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)
```

**SVM (Support Vector Machine)**:
```python
from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',         # Radial basis function
    C=1.0,
    gamma='scale',
    probability=True,     # Enable probability estimates
    random_state=42
)

svm.fit(X_train, y_train)
```

**LightGBM (Gradient Boosting)**:
```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
```

---

#### C. Regression Models

**Linear Regression**:
```python
from sklearn.linear_model import LinearRegression

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
```

**Ridge Regression** (L2 regularization):
```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
```

**SVR (Support Vector Regression)**:
```python
from sklearn.svm import SVR

svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train, y_train)
```

**LightGBM Regressor**:
```python
lgb_reg = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    random_state=42
)

lgb_reg.fit(X_train, y_train)
```

---

#### D. Evaluation Metrics

**Classification Metrics**:
```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(classification_report(y_true, y_pred))
```

**Regression Metrics**:
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
```

---

### 3. Semi-Supervised Learning Techniques

#### A. Pseudo-Labeling (Self-Training)

**Concept**: Use confident model predictions on unlabeled data to expand training set

**Algorithm**:
```
1. Train initial model on labeled data
2. Predict labels for unlabeled data
3. Select high-confidence predictions
4. Add pseudo-labeled samples to training set
5. Retrain model
6. Repeat until convergence or max iterations
```

**Implementation**:
```python
def pseudo_labeling(model, X_labeled, y_labeled, X_unlabeled, 
                    confidence_threshold=0.9, max_iterations=5):
    """
    Pseudo-labeling algorithm
    
    Args:
        model: Sklearn classifier with predict_proba
        X_labeled: Labeled features
        y_labeled: Labels
        X_unlabeled: Unlabeled features
        confidence_threshold: Minimum confidence for pseudo-labels
        max_iterations: Maximum pseudo-labeling iterations
    
    Returns:
        Trained model
    """
    X_train = X_labeled.copy()
    y_train = y_labeled.copy()
    X_pool = X_unlabeled.copy()
    
    for iteration in range(max_iterations):
        # Train model
        model.fit(X_train, y_train)
        
        # Predict on unlabeled data
        probas = model.predict_proba(X_pool)
        confidences = np.max(probas, axis=1)
        predictions = np.argmax(probas, axis=1)
        
        # Select high-confidence predictions
        high_conf_mask = confidences >= confidence_threshold
        
        if np.sum(high_conf_mask) == 0:
            print(f"No samples above threshold at iteration {iteration}")
            break
        
        # Add pseudo-labeled samples to training set
        X_pseudo = X_pool[high_conf_mask]
        y_pseudo = predictions[high_conf_mask]
        
        X_train = np.vstack([X_train, X_pseudo])
        y_train = np.concatenate([y_train, y_pseudo])
        
        # Remove from pool
        X_pool = X_pool[~high_conf_mask]
        
        print(f"Iteration {iteration+1}: Added {np.sum(high_conf_mask)} "
              f"pseudo-labeled samples. Pool size: {len(X_pool)}")
        
        if len(X_pool) == 0:
            break
    
    # Final training
    model.fit(X_train, y_train)
    return model
```

**Hyperparameters**:
- `confidence_threshold`: 0.7-0.95 typical (higher = more conservative)
- `max_iterations`: 3-10 typical
- Can add samples incrementally or in batches

---

#### B. Active Learning

**Concept**: Strategically select the most informative unlabeled samples for human labeling

**Query Strategies**:

**1. Uncertainty Sampling** (Least Confident):
```python
def uncertainty_sampling(model, X_pool, n_samples=10):
    """
    Select samples with lowest prediction confidence
    """
    probas = model.predict_proba(X_pool)
    confidences = np.max(probas, axis=1)
    
    # Select least confident samples
    uncertain_indices = np.argsort(confidences)[:n_samples]
    
    return uncertain_indices
```

**2. Margin Sampling**:
```python
def margin_sampling(model, X_pool, n_samples=10):
    """
    Select samples with smallest margin between top 2 classes
    """
    probas = model.predict_proba(X_pool)
    sorted_probas = np.sort(probas, axis=1)
    
    # Margin = difference between top 2 probabilities
    margins = sorted_probas[:, -1] - sorted_probas[:, -2]
    
    # Select smallest margins
    margin_indices = np.argsort(margins)[:n_samples]
    
    return margin_indices
```

**3. Entropy Sampling**:
```python
def entropy_sampling(model, X_pool, n_samples=10):
    """
    Select samples with highest prediction entropy
    """
    probas = model.predict_proba(X_pool)
    
    # Calculate entropy
    entropies = -np.sum(probas * np.log(probas + 1e-10), axis=1)
    
    # Select highest entropy
    entropy_indices = np.argsort(entropies)[-n_samples:][::-1]
    
    return entropy_indices
```

**Active Learning Loop**:
```python
def active_learning(model, X_labeled, y_labeled, X_unlabeled, 
                    n_queries=10, n_iterations=5, query_strategy='uncertainty'):
    """
    Active learning loop
    """
    X_train = X_labeled.copy()
    y_train = y_labeled.copy()
    X_pool = X_unlabeled.copy()
    
    # Simulated oracle (in practice, this would be human labeling)
    # Here we assume we have true labels for evaluation
    
    for iteration in range(n_iterations):
        # Train model
        model.fit(X_train, y_train)
        
        # Query selection
        if query_strategy == 'uncertainty':
            query_indices = uncertainty_sampling(model, X_pool, n_queries)
        elif query_strategy == 'margin':
            query_indices = margin_sampling(model, X_pool, n_queries)
        elif query_strategy == 'entropy':
            query_indices = entropy_sampling(model, X_pool, n_queries)
        else:
            # Random sampling baseline
            query_indices = np.random.choice(len(X_pool), n_queries, replace=False)
        
        # Simulate oracle labeling (in practice, human would label these)
        X_queried = X_pool[query_indices]
        y_queried = oracle_labels[query_indices]  # Simulated
        
        # Add to training set
        X_train = np.vstack([X_train, X_queried])
        y_train = np.concatenate([y_train, y_queried])
        
        # Remove from pool
        X_pool = np.delete(X_pool, query_indices, axis=0)
        
        print(f"Iteration {iteration+1}: Queried {n_queries} samples. "
              f"Train size: {len(X_train)}, Pool size: {len(X_pool)}")
    
    return model
```

---

#### C. Co-Training

**Concept**: Train multiple models on different views of data and have them teach each other

**Requirements**:
- Two (or more) independent feature sets (views)
- Views should be conditionally independent given the label

**Example Views**:
- View 1: SentenceTransformer embeddings
- View 2: Word2Vec embeddings + TF-IDF features

**Implementation**:
```python
def co_training(model1, model2, X1_labeled, X2_labeled, y_labeled,
                X1_unlabeled, X2_unlabeled, n_iterations=5, n_samples=10):
    """
    Co-training with two views
    
    Args:
        model1: Classifier for view 1
        model2: Classifier for view 2
        X1_labeled, X2_labeled: Labeled data (different views)
        y_labeled: Labels
        X1_unlabeled, X2_unlabeled: Unlabeled data (different views)
        n_iterations: Number of co-training iterations
        n_samples: Samples to add per iteration
    """
    X1_train, X2_train = X1_labeled.copy(), X2_labeled.copy()
    y_train = y_labeled.copy()
    
    X1_pool, X2_pool = X1_unlabeled.copy(), X2_unlabeled.copy()
    
    for iteration in range(n_iterations):
        # Train both models
        model1.fit(X1_train, y_train)
        model2.fit(X2_train, y_train)
        
        # Model 1 labels for Model 2
        probas1 = model1.predict_proba(X1_pool)
        conf1 = np.max(probas1, axis=1)
        pred1 = np.argmax(probas1, axis=1)
        
        # Select top confident from model1
        top_indices1 = np.argsort(conf1)[-n_samples:]
        
        # Model 2 labels for Model 1
        probas2 = model2.predict_proba(X2_pool)
        conf2 = np.max(probas2, axis=1)
        pred2 = np.argmax(probas2, axis=1)
        
        # Select top confident from model2
        top_indices2 = np.argsort(conf2)[-n_samples:]
        
        # Add model1's confident predictions to model2's training
        X2_train = np.vstack([X2_train, X2_pool[top_indices1]])
        X1_train = np.vstack([X1_train, X1_pool[top_indices1]])
        y_train = np.concatenate([y_train, pred1[top_indices1]])
        
        # Add model2's confident predictions to model1's training
        X2_train = np.vstack([X2_train, X2_pool[top_indices2]])
        X1_train = np.vstack([X1_train, X1_pool[top_indices2]])
        y_train = np.concatenate([y_train, pred2[top_indices2]])
        
        # Remove from pools
        all_indices = np.unique(np.concatenate([top_indices1, top_indices2]))
        X1_pool = np.delete(X1_pool, all_indices, axis=0)
        X2_pool = np.delete(X2_pool, all_indices, axis=0)
        
        print(f"Iteration {iteration+1}: Added {len(all_indices)} samples")
    
    return model1, model2
```

---

### 4. Visualization & Analysis

#### A. Embedding Visualization (PCA)

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Review Score')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Text Embeddings Visualization (PCA)')
plt.show()
```

#### B. Learning Curves

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn
pip install sentence-transformers gensim nltk
pip install lightgbm matplotlib seaborn
pip install torch  # If using neural networks
```

### Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Run the Notebook

```bash
cd code/
jupyter notebook DS_CA56_final.ipynb
```

---

## 📈 Expected Results

### Baseline Performance (Supervised Only)

**Classification**:
| Model | Accuracy | F1-Macro | Notes |
|-------|----------|----------|-------|
| Logistic Regression | 40-45% | 0.35-0.40 | Fast, interpretable |
| SVM | 42-48% | 0.38-0.43 | Good with small data |
| LightGBM | 45-52% | 0.42-0.48 | Best baseline |

**Regression**:
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 1.8-2.2 | 1.4-1.7 | 0.25-0.35 |
| Ridge | 1.7-2.1 | 1.3-1.6 | 0.30-0.40 |
| LightGBM | 1.5-1.9 | 1.2-1.5 | 0.40-0.50 |

---

### Semi-Supervised Improvements

**Pseudo-Labeling**:
- Expected improvement: +3-7% accuracy
- Works best with: High initial accuracy, confident predictions
- Risks: Label noise accumulation

**Active Learning**:
- Expected improvement: +5-10% accuracy with 20% additional labels
- Works best with: Informative queries, diverse pool
- Most efficient SSL method

**Co-Training**:
- Expected improvement: +4-8% accuracy
- Works best with: Independent views, moderate agreement
- Requires careful view design

---

## 🎓 Key Takeaways

### Text Representation
1. **Pre-trained embeddings** (SentenceTransformers) often outperform custom embeddings
2. **Domain-specific Word2Vec** can capture nuanced vocabulary
3. **Embedding quality** directly impacts downstream performance
4. **Dimensionality** affects both performance and computation

### Semi-Supervised Learning
1. **SSL helps** when labeled data is scarce (< 1000 samples)
2. **Quality over quantity**: 100 well-chosen labels > 1000 random labels
3. **Confidence thresholds** critical for pseudo-labeling
4. **Active learning** most label-efficient method
5. **View independence** crucial for co-training success

### Practical Insights
1. **Start with strong baseline** before trying SSL
2. **Monitor label quality** in pseudo-labeling
3. **Diverse queries** better than uncertainty alone
4. **Ensemble SSL methods** for best results
5. **Early stopping** prevents overfitting on pseudo-labels

---

## 🔍 Real-World Applications

### Semi-Supervised NLP
- **Review analysis**: Product/movie reviews with limited ratings
- **Document classification**: Legal, medical documents
- **Sentiment analysis**: Social media monitoring
- **Intent detection**: Chatbots and virtual assistants

### Active Learning Use Cases
- **Medical diagnosis**: Expert labeling is expensive
- **Content moderation**: Human review is time-consuming
- **Autonomous driving**: Annotation of edge cases
- **Scientific literature**: Domain expert classification

---

## 📖 References & Resources

### Papers
- **"Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method"** (Lee, 2013)
- **"A Survey on Semi-Supervised Learning"** (Van Engelen & Hoos, 2020)
- **"Active Learning Literature Survey"** (Settles, 2009)
- **"Co-Training for Natural Language"** (Blum & Mitchell, 1998)

### Books
- *Speech and Language Processing* by Jurafsky & Martin
- *Natural Language Processing with Python* (NLTK Book)
- *Semi-Supervised Learning* by Chapelle, Scholkopf & Zien

### Libraries & Tools
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/)
- [Scikit-learn Semi-Supervised Learning](https://scikit-learn.org/stable/modules/semi_supervised.html)

### Quick start
```bash
cd CA56_NLP_Semi_Supervised_Learning/code
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn gensim sentence-transformers nltk matplotlib seaborn
jupyter notebook DS_CA56_final.ipynb
```
If using SentenceTransformers for the first time, the model download will happen on first run.

---

## 👥 Team Members

Team assignment (check specific requirements).

---

**Created**: Fall 2024-2025  
**Last Updated**: January 2025

---

> **Note**: Semi-supervised learning is particularly valuable in domains where labeling is expensive. This assignment demonstrates how to maximize model performance with minimal labeled data - a critical skill for practical machine learning applications.

