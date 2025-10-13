# 📚 Loan Approval Classifier with PyTorch

## 📖 Overview
This project predicts **loan approval outcomes (Approved/Rejected)** using a neural network built with **PyTorch**.  

- 🧠 **Neural Network** with multiple hidden layers using **LeakyReLU** activation function and **Dropout**  
- ⚖️ **Binary Cross-Entropy (BCEWithLogitsLoss)** for training
- 🚀 Adam optimizer for gradient updates 
- 🔀 **Mini-batch training** with `DataLoader`  
- 📊 **Train/Validation/Test split** for robust evaluation  
- 📈 **Live training & validation loss monitoring**  
- ✅ **Sigmoid activation on the output** to produce probabilities, with a threshold for Approved/Rejected decision
- 🎨 **Interactive Gradio Interface** for real-time prediction

---

## 🖼️ Application Screenshot

Below is a preview of the **Gradio Interface** used for real-time classification:

![Application Screenshot](assets/app_screenshot.png)

---

## 🧩 Libraries
- **PyTorch** – model, training, and inference  
- **pandas** – data handling  
- **matplotlib** – loss visualization  
- **pickle** – saving/loading normalization params and trained model
- **Gradio** — interactive web interface for real-time model demos 

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
cd Loan-Approval-Classifier
```

- Install dependencies
```bash
pip install -r requirements.txt
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
assets/
└── app_screenshot.png                # Screenshot of the application

data/
└── loan_data.csv                     # Raw dataset

model/
├── loan_approval_classifier.pth      # Trained model (after training)
├── feature_transformer.pkl           # Fitted preprocessing transformer
└── statistics.pkl                    # Preprocessing stats

src/
├── config.py                         # Paths, hyperparameters, split ratios
├── dataset.py                        # Data loading & preprocessing
├── device_manager.py                 # Selects and manages compute device
├── train.py                          # Training pipeline
├── inference.py                      # Inference pipeline
├── model.py                          # Neural network definition
└── visualize.py                      # Training/validation plots

main/
├── main_train.py                     # Entry point for training
└── main_inference.py                 # Entry point for inference

requirements.txt                      # Python dependencies
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
Navigate to the project directory:
```bash
cd Loan-Approval-Classifier
```

Run the training script:
```bash
python -m main.main_train
```
or
```bash
python3 -m main.main_train
```

---

## 📂 Run Inference / Make Predictions
Navigate to the project directory:
```bash
cd Loan-Approval-Classifier
```

Run the app:
```bash
python -m main.main_inference
```
or
```bash
python3 -m main.main_inference
```
