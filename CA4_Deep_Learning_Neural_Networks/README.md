# Assignment 4: Deep Learning & Neural Networks
## MLP, CNN, and RNN Architectures

---

## 📊 Project Overview

This assignment provides hands-on experience with three fundamental **deep learning architectures**: **Multi-Layer Perceptrons (MLP)**, **Convolutional Neural Networks (CNN)**, and **Recurrent Neural Networks (RNN)**. Each task demonstrates practical applications of these architectures on different types of data and problems.

---

## 🎯 Learning Objectives

### Task 1: Multi-Layer Perceptrons (MLP)
- Build fully connected neural networks
- Implement **backpropagation** from scratch
- Understand **activation functions** and their impact
- Master **hyperparameter tuning**
- Visualize loss landscapes and training dynamics

### Task 2: Convolutional Neural Networks (CNN)
- Design CNN architectures for **image classification**
- Implement **convolutional**, **pooling**, and **batch normalization** layers
- Apply **data augmentation** techniques
- Use **transfer learning** with pre-trained models
- Optimize for image-based tasks

### Task 3: Recurrent Neural Networks (RNN)
- Build sequence models with **LSTM** and **GRU**
- Handle **time-series data** and **sequential patterns**
- Implement **bidirectional RNNs**
- Address **vanishing gradient** problems
- Apply to forecasting and sequence prediction

---

## 📁 Project Structure

```
Deep_Learning_Neural_Networks/
├── codes/
│   ├── Task1/
│   │   └── Task1_MLP.ipynb          # MLP implementation
│   ├── Task2/
│   │   ├── Task2_CNNs.ipynb         # CNN for images
│   │   ├── Part2.ipynb              # Advanced CNN techniques
│   │   └── VQ3.pdf                  # Results and analysis
│   └── Task3/
│       ├── Task3_RNNs.ipynb         # RNN/LSTM implementation
│       └── Task3ـRNNs.ipynb         # Additional RNN tasks
├── datasets/
│   ├── BTC-USD.csv                  # Bitcoin price data
│   └── matches.csv                  # Match results data
├── description/
│   └── DS-CA4.pdf                   # Assignment description
└── README.md                        # This file
```

---

## 🧠 Task 1: Multi-Layer Perceptron (MLP)

### Overview
Implement and train fully-connected neural networks for **classification** and **regression** tasks. Understand the mathematics behind backpropagation and gradient descent.

---

### Key Concepts

#### 1. Neural Network Architecture

**Basic MLP Structure**:
```
Input Layer → Hidden Layer(s) → Output Layer
```

**Mathematical Formulation**:
```python
# Forward pass
z1 = W1 @ x + b1
a1 = activation(z1)
z2 = W2 @ a1 + b2
y_pred = activation(z2)

# Loss
L = loss_function(y_pred, y_true)
```

---

#### 2. Activation Functions

**ReLU (Rectified Linear Unit)**:
```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

**Sigmoid**:
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

**Tanh**:
```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

**Softmax** (for multi-class classification):
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

---

#### 3. Backpropagation

**Chain Rule Application**:
```python
# Output layer gradients
dL_dz2 = y_pred - y_true
dL_dW2 = dL_dz2 @ a1.T
dL_db2 = np.sum(dL_dz2, axis=1, keepdims=True)

# Hidden layer gradients
dL_da1 = W2.T @ dL_dz2
dL_dz1 = dL_da1 * activation_derivative(z1)
dL_dW1 = dL_dz1 @ x.T
dL_db1 = np.sum(dL_dz1, axis=1, keepdims=True)
```

---

#### 4. Optimization

**Stochastic Gradient Descent (SGD)**:
```python
W = W - learning_rate * dL_dW
b = b - learning_rate * dL_db
```

**Adam Optimizer**:
```python
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
W = W - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
```

---

#### 5. Regularization

**L2 Regularization (Weight Decay)**:
```python
loss = mse_loss + lambda_reg * np.sum(W**2)
gradient = gradient + 2 * lambda_reg * W
```

**Dropout**:
```python
def dropout(x, drop_rate=0.5, training=True):
    if training:
        mask = np.random.binomial(1, 1-drop_rate, x.shape) / (1-drop_rate)
        return x * mask
    return x
```

---

### Implementation Example

```python
import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        # Initialize weights
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i+1], dims[i]) * 0.01
            b = np.zeros((dims[i+1], 1))
            self.layers.append({'W': W, 'b': b})
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i, layer in enumerate(self.layers):
            z = layer['W'] @ self.activations[-1] + layer['b']
            self.z_values.append(z)
            
            # Apply activation (ReLU for hidden, sigmoid for output)
            if i < len(self.layers) - 1:
                a = np.maximum(0, z)  # ReLU
            else:
                a = 1 / (1 + np.exp(-z))  # Sigmoid
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[1]
        
        # Output layer gradient
        dz = self.activations[-1] - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            dW = (1/m) * dz @ self.activations[i].T
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            
            # Update parameters
            self.layers[i]['W'] -= learning_rate * dW
            self.layers[i]['b'] -= learning_rate * db
            
            if i > 0:
                da = self.layers[i]['W'].T @ dz
                dz = da * (self.z_values[i-1] > 0)  # ReLU derivative
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = -np.mean(y * np.log(y_pred + 1e-8) + 
                           (1 - y) * np.log(1 - y_pred + 1e-8))
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
```

---

## 🖼️ Task 2: Convolutional Neural Networks (CNN)

### Overview
Design and train CNNs for **image classification** tasks. Learn about spatial feature extraction and hierarchical representations.

---

### Key Concepts

#### 1. Convolutional Layer

**Convolution Operation**:
```python
import torch.nn as nn

conv = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=64,    # 64 filters
    kernel_size=3,      # 3x3 filter
    stride=1,
    padding=1
)
```

**Feature Map Calculation**:
```
Output size = (Input size - Kernel size + 2*Padding) / Stride + 1
```

---

#### 2. Pooling Layers

**Max Pooling**:
```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# Reduces spatial dimensions by half
```

**Average Pooling**:
```python
pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

---

#### 3. CNN Architecture Example

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
```

---

#### 4. Data Augmentation

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

#### 5. Transfer Learning

```python
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Only train the new layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

---

#### 6. Training Loop

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc
```

---

## 🔄 Task 3: Recurrent Neural Networks (RNN)

### Overview
Build sequence models for **time-series prediction** and **sequential data processing**. Learn to handle temporal dependencies.

---

### Key Concepts

#### 1. Basic RNN Cell

**RNN Equation**:
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

**Implementation**:
```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, hidden = self.rnn(x)
        # out shape: (batch, seq_len, hidden_size)
        
        # Use last time step
        out = self.fc(out[:, -1, :])
        return out
```

---

#### 2. LSTM (Long Short-Term Memory)

**LSTM Gates**:
```
Forget gate: f_t = σ(W_f [h_{t-1}, x_t] + b_f)
Input gate:  i_t = σ(W_i [h_{t-1}, x_t] + b_i)
Cell state:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C [h_{t-1}, x_t] + b_C)
Output gate: o_t = σ(W_o [h_{t-1}, x_t] + b_o)
Hidden:      h_t = o_t ⊙ tanh(C_t)
```

**PyTorch Implementation**:
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), 
                         self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), 
                         self.lstm.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state
        out = self.fc(out[:, -1, :])
        return out
```

---

#### 3. GRU (Gated Recurrent Unit)

**GRU Equations** (simpler than LSTM):
```
Update gate: z_t = σ(W_z [h_{t-1}, x_t])
Reset gate:  r_t = σ(W_r [h_{t-1}, x_t])
Candidate:   h̃_t = tanh(W [r_t ⊙ h_{t-1}, x_t])
Hidden:      h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Implementation**:
```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
```

---

#### 4. Bidirectional RNN

```python
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Key difference
            dropout=0.3
        )
        
        # hidden_size * 2 because bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```

---

#### 5. Time Series Forecasting

**Data Preparation**:
```python
def create_sequences(data, seq_length):
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)

# Example: Bitcoin price prediction
seq_length = 30  # Use past 30 days to predict next day
X, y = create_sequences(btc_prices, seq_length)
```

**Training**:
```python
model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
```

---

## 🛠️ Technologies & Libraries

### Deep Learning Frameworks
```python
torch >= 1.10          # PyTorch deep learning
torchvision >= 0.11    # Computer vision datasets & models
tensorflow >= 2.8      # Alternative framework
keras >= 2.8           # High-level API
```

### Data Processing
```python
numpy >= 1.20          # Numerical operations
pandas >= 1.3          # Data manipulation
scikit-learn >= 1.0    # Preprocessing & metrics
```

### Visualization
```python
matplotlib >= 3.4      # Plotting
seaborn >= 0.11        # Statistical viz
tensorboard >= 2.8     # Training monitoring
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

### Quick start
```bash
cd CA4_Deep_Learning_Neural_Networks/codes
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

### Task 1: MLP
```bash
cd codes/Task1/
jupyter notebook Task1_MLP.ipynb
```

### Task 2: CNN
```bash
cd codes/Task2/
jupyter notebook Task2_CNNs.ipynb
```

### Task 3: RNN
```bash
cd codes/Task3/
jupyter notebook Task3_RNNs.ipynb
```

GPU is optional but recommended for the CNN/RNN notebooks; set the PyTorch device inside the notebooks if available.

---

## 📈 Expected Results

### Task 1: MLP Performance
- **MNIST**: 97-98% accuracy
- **Training time**: 5-10 minutes
- **Convergence**: ~20-30 epochs

### Task 2: CNN Performance
- **CIFAR-10**: 85-92% accuracy
- **ImageNet (Transfer Learning)**: 90-95% accuracy
- **Training time**: 1-2 hours (with GPU)

### Task 3: RNN Performance
- **Bitcoin Prediction RMSE**: 500-1000
- **Sequence Classification Accuracy**: 85-90%
- **Training time**: 30-60 minutes

---

## 🎓 Key Takeaways

### MLP Insights
1. **Depth helps** but requires careful initialization
2. **Activation functions** dramatically affect learning
3. **Regularization** essential to prevent overfitting
4. **Batch normalization** speeds up training

### CNN Insights
1. **Convolutional layers** learn hierarchical features
2. **Data augmentation** critical for generalization
3. **Transfer learning** often outperforms training from scratch
4. **Batch size** affects both speed and accuracy

### RNN Insights
1. **LSTM/GRU** solve vanishing gradient problem
2. **Sequence length** is critical hyperparameter
3. **Bidirectional** models improve performance
4. **Teacher forcing** helps training but may hurt inference

---

## 🔍 Real-World Applications

### MLPs
- Tabular data classification
- Feature transformation
- Embedding layers

### CNNs
- Image classification & detection
- Medical image analysis
- Self-driving cars
- Face recognition

### RNNs
- Stock price prediction
- Natural language processing
- Speech recognition
- Video analysis

---

## 📖 References & Resources

### Books
- *Deep Learning* by Goodfellow, Bengio, and Courville
- *Dive into Deep Learning* by Zhang et al.
- *Neural Networks and Deep Learning* by Michael Nielsen

### Courses
- Stanford CS231n: CNNs for Visual Recognition
- Stanford CS224n: NLP with Deep Learning
- Deep Learning Specialization (Coursera)

### Documentation
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)

---

## 👥 Team Members

Individual or team assignment (check specific requirements).

---

**Created**: Fall 2024-2025  
**Last Updated**: January 2025

---

> **Note**: Deep learning requires significant computational resources. Using Google Colab or similar platforms with GPU acceleration is recommended for faster training.
