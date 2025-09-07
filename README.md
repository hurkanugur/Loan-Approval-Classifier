# 📚 Loan Approval Classifier with PyTorch

## 📖 Overview
This project predicts **loan approval outcomes (Approved/Rejected)** using a neural network built with **PyTorch**.  
It demonstrates a full machine learning pipeline from data loading to inference, including:

- 🧠 **Neural Network** with multiple hidden layers using LeakyReLU activation function  
- ⚖️ **Binary Cross-Entropy (BCEWithLogitsLoss)** for training
- 🚀 Adam optimizer for gradient updates 
- 🔀 **Mini-batch training** with `DataLoader`  
- 📊 **Train/Validation/Test split** for robust evaluation  
- 📈 **Live training & validation loss monitoring**  
- ✅ **Sigmoid activation on the output** to produce probabilities, with a threshold for Approved/Rejected decision

---

## 🧩 Libraries
- **PyTorch** – model, training, and inference  
- **pandas** – data handling  
- **matplotlib** – loss visualization  
- **pickle** – saving/loading normalization params and trained model

---

## ⚙️ Requirements

- Python **3.13+**
- Recommended editor: **VS Code**

---

## 📦 Installation

- Clone the repository
```bash
git clone https://github.com/hurkanugur/Loan-Approval-Classifier.git
```

- Navigate to the `Loan-Approval-Classifier` directory
```bash
cd Car_Price_Predictor
```

- Install dependencies
```bash
pip install -r requirements.txt
```

- Navigate to the `Loan-Approval-Classifier/src` directory
```bash
cd src
```

---

## 🔧 Setup Python Environment in VS Code

1. `View → Command Palette → Python: Create Environment`  
2. Choose **Venv** and your **Python version**  
3. Select **requirements.txt** to install dependencies  
4. Click **OK**

---

## 📂 Project Structure

```bash
data/
└── loan_data.csv                         # Raw dataset

model/
└── loan_approval_classifier.pth          # Trained model (after training)

src/
├── config.py                             # Paths, hyperparameters, split ratios
├── dataset.py                            # Data loading & preprocessing
├── main_train.py                         # Training & model saving
├── main_inference.py                     # Inference pipeline
├── model.py                              # Neural network definition
├── visualize.py                          # Training/validation plots

requirements.txt                          # Python dependencies
```

---

## 📂 Model Architecture

```bash
Input → Linear(128) → LeakyReLU(0.01) → Dropout(0.2)
      → Linear(64)  → LeakyReLU(0.01) → Dropout(0.1)
      → Linear(32)  → LeakyReLU(0.01)
      → Linear(8)   → LeakyReLU(0.01)
      → Linear(1)   → Sigmoid(Output)
```

---

## 📂 Train the Model
```bash
python main_train.py
```
or
```bash
python3 main_train.py
```

---

## 📂 Run Predictions on Real Data
```bash
python main_inference.py
```
or
```bash
python3 main_inference.py
```
