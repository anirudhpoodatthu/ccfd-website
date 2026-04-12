"""
=============================================================================
CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
WITH IMBALANCED DATA HANDLING AND WEB DEPLOYMENT
=============================================================================
Author      : ML Engineering Team
Description : Full data science pipeline for fraud detection including EDA,
              SMOTE-based imbalance handling, multi-model training, evaluation,
              and best model selection with serialization.
=============================================================================
"""

# ---------------------------------------------
# 1. IMPORTS
# ---------------------------------------------
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, matthews_corrcoef,
    classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ---------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
SMOTE_STRATEGY = 0.1   # 10% fraud ratio -- fast training, still balanced
MODEL_PATH     = "model.pkl"
PLOTS_DIR      = "plots"
DATA_PATH      = "creditcard.csv"   # place Kaggle CSV here

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the credit card fraud dataset from a CSV file."""
    print(f"\n[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns      : {list(df.columns)}")
    return df


# ---------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------
def perform_eda(df: pd.DataFrame) -> None:
    """Run EDA: summary stats, missing values, class distribution, correlation."""

    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # 4.1 Basic info
    print("\n[INFO] First 5 rows:")
    print(df.head())

    print("\n[INFO] Statistical Summary:")
    print(df.describe())

    print("\n[INFO] Missing Values:")
    print(df.isnull().sum())

    # 4.2 Class distribution
    class_counts = df["Class"].value_counts()
    fraud_pct    = class_counts[1] / len(df) * 100
    print(f"\n[INFO] Class Distribution:\n{class_counts}")
    print(f"[INFO] Fraud percentage: {fraud_pct:.4f}%")

    # -- Plot: Class Distribution ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution - Credit Card Fraud Dataset", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#F44336"]
    axes[0].bar(["Legitimate (0)", "Fraud (1)"], class_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Transaction Count by Class")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    axes[1].pie(
        class_counts.values,
        labels=["Legitimate", "Fraud"],
        autopct="%1.3f%%",
        colors=colors,
        startangle=90,
        explode=(0, 0.1)
    )
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/class_distribution.png", dpi=150)
    plt.close()
    print(f"[SAVED] {PLOTS_DIR}/class_distribution.png")

    # -- Plot: Transaction Amount Distribution -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount Distribution", fontsize=14, fontweight="bold")

    df[df["Class"] == 0]["Amount"].hist(bins=60, ax=axes[0], color="#2196F3", edgecolor="black", alpha=0.8)
    axes[0].set_title("Legitimate Transactions")
    axes[0].set_xlabel("Amount (USD)")
    axes[0].set_ylabel("Frequency")

    df[df["Class"] == 1]["Amount"].hist(bins=60, ax=axes[1], color="#F44336", edgecolor="black", alpha=0.8)
    axes[1].set_title("Fraudulent Transactions")
    axes[1].set_xlabel("Amount (USD)")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/amount_distribution.png", dpi=150)
    plt.close()
    print(f"[SAVED] {PLOTS_DIR}/amount_distribution.png")

    # -- Plot: Correlation Heatmap -----------------------------------------
    plt.figure(figsize=(20, 16))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=False, cmap="coolwarm",
        linewidths=0.3, vmin=-1, vmax=1, square=True
    )
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"[SAVED] {PLOTS_DIR}/correlation_heatmap.png")

    # -- Plot: Top features correlated with Class --------------------------
    class_corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    class_corr.plot(kind="bar", color="#7E57C2", edgecolor="black")
    plt.title("Top 15 Features Correlated with Fraud (Class)", fontsize=13, fontweight="bold")
    plt.xlabel("Feature")
    plt.ylabel("|Correlation|")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_correlation_class.png", dpi=150)
    plt.close()
    print(f"[SAVED] {PLOTS_DIR}/feature_correlation_class.png")


# ---------------------------------------------
# 5. FEATURE ENGINEERING & PREPROCESSING
# ---------------------------------------------
def preprocess(df: pd.DataFrame):
    """
    Scale 'Time' and 'Amount', split into train/test sets (stratified).
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    print("\n[INFO] Preprocessing data ...")

    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"]   = scaler.fit_transform(df[["Time"]])
    df.drop(["Amount", "Time"], axis=1, inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"[INFO] Train size : {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")
    print(f"[INFO] Train fraud: {y_train.sum():,}  |  Test fraud: {y_test.sum():,}")

    return X_train, X_test, y_train, y_test, list(X.columns)


# ---------------------------------------------
# 6. SMOTE - IMBALANCE HANDLING
# ---------------------------------------------
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Apply SMOTE to the training set and visualise before/after distribution.
    Returns: X_resampled, y_resampled
    """
    print("\n[INFO] Applying SMOTE ...")
    print(f"[INFO] Before SMOTE - Class distribution:\n{y_train.value_counts()}")

    smote = SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"[INFO] After  SMOTE - Class distribution:\n{pd.Series(y_res).value_counts()}")

    # -- Plot: Before vs After SMOTE ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution: Before vs After SMOTE", fontsize=14, fontweight="bold")

    before = y_train.value_counts()
    after  = pd.Series(y_res).value_counts()
    colors = ["#2196F3", "#F44336"]

    axes[0].bar(["Legitimate", "Fraud"], before.values, color=colors, edgecolor="black")
    axes[0].set_title("Before SMOTE")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(before.values):
        axes[0].text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")

    axes[1].bar(["Legitimate", "Fraud"], after.values, color=colors, edgecolor="black")
    axes[1].set_title("After SMOTE")
    for i, v in enumerate(after.values):
        axes[1].text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/smote_comparison.png", dpi=150)
    plt.close()
    print(f"[SAVED] {PLOTS_DIR}/smote_comparison.png")

    return X_res, y_res


# ---------------------------------------------
# 7. MODEL DEFINITIONS
# ---------------------------------------------
def get_models() -> dict:
    """Return a dictionary of ML models to train and compare."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.15, max_depth=3,
            random_state=RANDOM_STATE, subsample=0.8
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1, algorithm='ball_tree'
        ),
    }


# ---------------------------------------------
# 8. MODEL EVALUATION HELPER
# ---------------------------------------------
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """
    Compute all evaluation metrics for a fitted model.
    Returns a metrics dictionary.
    """
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    acc         = accuracy_score(y_test, y_pred)
    prec        = precision_score(y_test, y_pred, zero_division=0)
    rec         = recall_score(y_test, y_pred, zero_division=0)
    f1          = f1_score(y_test, y_pred, zero_division=0)
    roc_auc     = roc_auc_score(y_test, y_prob)
    mcc         = matthews_corrcoef(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Model : {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Legitimate','Fraud'])}")

    return {
        "Model"    : name,
        "Accuracy" : round(acc,  4),
        "Precision": round(prec, 4),
        "Recall"   : round(rec,  4),
        "F1-Score" : round(f1,   4),
        "ROC-AUC"  : round(roc_auc, 4),
        "MCC"      : round(mcc,  4),
        "y_prob"   : y_prob,
        "y_pred"   : y_pred,
    }


# ---------------------------------------------
# 9. TRAIN ALL MODELS
# ---------------------------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test) -> tuple[dict, list]:
    """
    Train every model, evaluate on test set, return fitted models dict
    and list of metrics dicts.
    """
    models      = get_models()
    fitted      = {}
    results     = []

    print("\n" + "="*60)
    print("  MODEL TRAINING & EVALUATION")
    print("="*60)

    for name, model in models.items():
        print(f"\n[INFO] Training: {name} ...")
        model.fit(X_train, y_train)
        fitted[name] = model
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

    return fitted, results


# ---------------------------------------------
# 10. VISUALISATIONS - CONFUSION MATRICES & ROC
# ---------------------------------------------
def plot_confusion_matrices(results: list, y_test) -> None:
    """Plot confusion matrix for every model in a grid."""
    n      = len(results)
    ncols  = 2
    nrows  = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes   = axes.flatten()
    fig.suptitle("Confusion Matrices - All Models", fontsize=15, fontweight="bold")

    for i, res in enumerate(results):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"])
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(res["Model"], fontsize=12, fontweight="bold")

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/confusion_matrices.png", dpi=150)
    plt.close()
    print(f"[SAVED] {PLOTS_DIR}/confusion_matrices.png")


def plot_roc_curves(results: list, y_test) -> None:
    """Plot ROC curves for all models on a single figure."""
    plt.figure(figsize=(10, 7))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 lw=2, label=f"{res['Model']} (AUC = {res['ROC-AUC']:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - All Models", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/roc_curves.png", dpi=150)
    plt.close()
    print(f"[SAVED] {PLOTS_DIR}/roc_curves.png")


def plot_model_comparison(results: list) -> None:
    """Bar chart comparing all models across key metrics."""
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC"]
    df_res = pd.DataFrame(results)[["Model"] + metrics_cols].set_index("Model")

    ax = df_res.plot(kind="bar", figsize=(14, 7), edgecolor="black", width=0.75)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticklabels(df_res.index, rotation=20, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_comparison.png", dpi=150)
    plt.close()
    print(f"[SAVED] {PLOTS_DIR}/model_comparison.png")


# ---------------------------------------------
# 11. MODEL COMPARISON TABLE
# ---------------------------------------------
def print_comparison_table(results: list) -> pd.DataFrame:
    """Print a formatted comparison table and return as DataFrame."""
    metrics_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC"]
    df_res = pd.DataFrame(results)[metrics_cols].sort_values("ROC-AUC", ascending=False)

    print("\n" + "="*80)
    print("  MODEL COMPARISON TABLE (sorted by ROC-AUC)")
    print("="*80)
    print(df_res.to_string(index=False))
    print("="*80)

    return df_res


# ---------------------------------------------
# 12. BEST MODEL SELECTION
# ---------------------------------------------
def select_best_model(fitted_models: dict, results: list) -> tuple:
    """
    Automatically select the best model based on ROC-AUC score.
    Returns: (best_model_name, best_model_object)
    """
    best = max(results, key=lambda x: x["ROC-AUC"])
    best_name  = best["Model"]
    best_model = fitted_models[best_name]

    print(f"\n[BEST MODEL] -> {best_name}")
    print(f"  ROC-AUC  : {best['ROC-AUC']:.4f}")
    print(f"  F1-Score : {best['F1-Score']:.4f}")
    print(f"  Recall   : {best['Recall']:.4f}")
    print(f"  MCC      : {best['MCC']:.4f}")

    return best_name, best_model


# ---------------------------------------------
# 13. SAVE MODEL
# ---------------------------------------------
def save_model(model, model_name: str, feature_names: list) -> None:
    """Serialise the best model + metadata using pickle."""
    payload = {
        "model"        : model,
        "model_name"   : model_name,
        "feature_names": feature_names,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n[SAVED] Model serialised -> {MODEL_PATH}")


# ---------------------------------------------
# 14. WHY ACCURACY IS NOT ENOUGH
# ---------------------------------------------
def explain_metrics() -> None:
    print("""
+--------------------------------------------------------------------------+
|         WHY ACCURACY ALONE IS NOT SUFFICIENT FOR FRAUD DETECTION         |
+--------------------------------------------------------------------------+
|                                                                          |
|  The dataset is HIGHLY IMBALANCED (~0.172% fraud cases).                |
|                                                                          |
|  A naive classifier that predicts EVERY transaction as "Legitimate"     |
|  would achieve ~99.83% accuracy -- yet it would NEVER detect any fraud. |
|                                                                          |
|  Better metrics for imbalanced classification:                           |
|                                                                          |
|  * Precision  - Of all predicted frauds, how many are real?             |
|  * Recall     - Of all real frauds, how many did we catch?              |
|  * F1-Score   - Harmonic mean of Precision & Recall.                    |
|  * ROC-AUC    - Ability to distinguish classes across all thresholds.   |
|  * MCC        - Balanced metric even with class imbalance.              |
|                                                                          |
|  In fraud detection, RECALL is critical -- missing a fraud (False       |
|  Negative) is far more costly than a false alarm (False Positive).      |
+--------------------------------------------------------------------------+
""")


# ---------------------------------------------
# 15. MAIN PIPELINE
# ---------------------------------------------
def main():
    explain_metrics()

    # -- Load --------------------------------------------------------------
    df = load_data(DATA_PATH)

    # -- EDA ---------------------------------------------------------------
    perform_eda(df)

    # -- Preprocess --------------------------------------------------------
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    # -- SMOTE -------------------------------------------------------------
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # -- Train & Evaluate --------------------------------------------------
    fitted_models, results = train_and_evaluate(X_train_res, y_train_res, X_test, y_test)

    # -- Visualisations ----------------------------------------------------
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_model_comparison(results)

    # -- Comparison Table --------------------------------------------------
    print_comparison_table(results)

    # -- Best Model --------------------------------------------------------
    best_name, best_model = select_best_model(fitted_models, results)

    # -- Save --------------------------------------------------------------
    save_model(best_model, best_name, feature_names)

    print("\n[DONE] Pipeline complete. All plots saved to ./plots/")


if __name__ == "__main__":
    main()
