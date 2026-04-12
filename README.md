# 🔒 Credit Card Fraud Detection Using Machine Learning

> **With Imbalanced Data Handling and Web Deployment**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)

---

## 📌 Project Overview

This is a **production-level machine learning project** that builds an accurate **credit card fraud detection system** capable of identifying fraudulent transactions from a highly imbalanced dataset. The project implements a complete end-to-end data science pipeline—from data exploration and preprocessing through model training, evaluation, and deployment via a professional **Flask web application**.

### Key Highlights

- 🎯 **4 ML models** trained and compared (Logistic Regression, Random Forest, SVM, KNN)
- ⚖️ **SMOTE** (Synthetic Minority Over-sampling Technique) for handling extreme class imbalance
- 📊 **Comprehensive evaluation** using 7+ metrics (Accuracy, Precision, Recall, F1, ROC-AUC, MCC)
- 🏆 **Automatic best model selection** based on ROC-AUC score
- 🌐 **Flask web application** with premium dark glassmorphic UI for real-time fraud prediction
- 📈 **Rich visualizations** including ROC curves, confusion matrices, correlation heatmaps, and more

---

## 📁 Project Structure

```
ml/
├── model_training.py        # Complete ML pipeline (EDA → SMOTE → Training → Evaluation → Save)
├── app.py                   # Flask web application
├── model.pkl                # Saved best model (generated after training)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── creditcard.csv           # Dataset (download from Kaggle)
├── templates/
│   └── index.html           # Main web page template
├── static/
│   ├── css/
│   │   └── style.css        # Premium dark glassmorphic theme
│   └── js/
│       └── app.js           # Client-side logic & interactivity
└── plots/                   # Generated visualizations
    ├── class_distribution.png
    ├── amount_distribution.png
    ├── correlation_heatmap.png
    ├── feature_correlation_class.png
    ├── smote_comparison.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── model_comparison.png
```

---

## 📊 Dataset

**Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Details |
|---|---|
| **Total Transactions** | 284,807 |
| **Fraudulent** | 492 (0.172%) |
| **Legitimate** | 284,315 (99.828%) |
| **Features** | 30 (V1–V28 PCA, Time, Amount) |
| **Target** | Class (0 = Legitimate, 1 = Fraud) |
| **Imbalance Ratio** | ~1:578 |

### Feature Description

- **V1–V28**: Principal Component Analysis (PCA) transformed features (confidential original features)
- **Time**: Seconds elapsed between each transaction and the first transaction in the dataset
- **Amount**: Transaction monetary amount
- **Class**: Target variable (0 = Legitimate, 1 = Fraudulent)

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Imbalance Handling** | Imbalanced-Learn (SMOTE) |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Flask |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+) |
| **UI Design** | Glassmorphism, Dark Theme, CSS Animations |
| **Model Serialization** | Pickle |

---

## 🚀 How to Run

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Navigate to Project Directory

```bash
cd path/to/ml
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset

1. Go to [Kaggle Dataset Page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the project root directory (`ml/`)

### Step 4: Train the Model

```bash
python model_training.py
```

This will:
- Perform complete EDA with visualizations
- Apply SMOTE for class balancing
- Train 4 ML models (Logistic Regression, Random Forest, SVM, KNN)
- Generate evaluation metrics and comparison tables
- Automatically select the best model
- Save the model as `model.pkl`
- Save all plots to `./plots/`

### Step 5: Launch the Web Application

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

---

## 📈 Results Summary

### Model Performance Comparison

The following metrics were evaluated for each model after applying SMOTE:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | ~0.97 | ~0.06 | ~0.92 | ~0.11 | ~0.98 | ~0.23 |
| Random Forest | ~0.999 | ~0.88 | ~0.82 | ~0.85 | ~0.98 | ~0.85 |
| SVM | ~0.98 | ~0.10 | ~0.90 | ~0.18 | ~0.98 | ~0.30 |
| KNN | ~0.998 | ~0.72 | ~0.80 | ~0.76 | ~0.96 | ~0.76 |

> **Note:** Exact values depend on random state and data splits.

### Key Insights

1. **Accuracy is misleading** on imbalanced data—all models achieve ~99% accuracy
2. **SMOTE significantly improves Recall** (fraud detection rate)
3. **Random Forest** consistently performs best across most metrics
4. **ROC-AUC** is the primary metric for best model selection
5. **MCC** provides the most balanced assessment on imbalanced data

### Why Accuracy Alone Is Not Sufficient

With only 0.172% fraud cases, a classifier that **always predicts "legitimate"** achieves **99.83% accuracy** — yet catches **zero** frauds. For fraud detection, we prioritize **Recall** and **ROC-AUC**.

---

## 🌐 Web Application Features

### Input Modes

1. **Manual Input**: Enter individual transaction features (V1–V28, scaled_time, scaled_amount)
2. **Quick Test**: Generate random test data with legitimate/suspicious/random profiles
3. **CSV Upload**: Drag-and-drop batch prediction on uploaded transaction files

### UI Features

- 🌙 Premium dark glassmorphic theme with animated particle background
- 📊 SVG gauge chart for fraud probability visualization
- 📉 Animated probability comparison bars
- 🎨 Risk level indicator (Low / Medium / High)
- 📋 Batch results table with status badges
- 🔄 Loading states with spinner animations
- 📱 Fully responsive design

---

## 📜 License

This project is developed for **academic and educational purposes**.

---

## 🙏 Acknowledgments

- **Dataset**: [Machine Learning Group – ULB, Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **SMOTE**: Chawla, N.V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- **Scikit-Learn**: Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python"

---

<div align="center">
  <strong>Built with ❤️ for Academic Excellence</strong>
</div>
