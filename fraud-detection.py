#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ============================================================
# FRAUD DETECTION USING PYOD AUTOENCODER
# Unsupervised Deep Learning Experiment
# Dataset: creditcard.csv
# ============================================================

# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

print("Libraries imported successfully.")


# ------------------------------------------------------------
# 2. Load the dataset
# ------------------------------------------------------------
DATA_FILE = "creditcard.csv"

df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Dataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst five rows:")
display(df.head())


# ------------------------------------------------------------
# 3. Check class distribution
# ------------------------------------------------------------
normal_count = int((df["Class"] == 0).sum())
fraud_count = int((df["Class"] == 1).sum())

fraud_rate = fraud_count / len(df)

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print("Normal transactions:", normal_count)
print("Fraudulent transactions:", fraud_count)
print(f"Fraud percentage: {fraud_rate * 100:.4f}%")


# ------------------------------------------------------------
# 4. Separate features and true labels
# ------------------------------------------------------------
# IMPORTANT:
# The 'Class' column is NOT used when training the AutoEncoder.
# It is used only afterward to evaluate the detected anomalies.

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)


# ------------------------------------------------------------
# 5. Split data into training and testing sets
# ------------------------------------------------------------
# Stratification keeps approximately the same fraud percentage
# in the training and testing sets.

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training records:", len(X_train_raw))
print("Testing records:", len(X_test_raw))


# ------------------------------------------------------------
# 6. Standardize the features
# ------------------------------------------------------------
# Fit the scaler only on training data to avoid data leakage.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print("\nFeature scaling completed successfully.")


# ------------------------------------------------------------
# 7. Define contamination
# ------------------------------------------------------------
# Contamination is the approximate expected proportion of
# anomalies in the dataset.
#
# The known fraud percentage is used only to configure the
# unsupervised threshold. Class labels are not given to fit().

contamination = fraud_rate

print(f"Contamination rate: {contamination:.6f}")


# ------------------------------------------------------------
# 8. Create the PyOD AutoEncoder
# ------------------------------------------------------------
model = AutoEncoder(
    contamination=contamination,
    hidden_neuron_list=[32, 16, 8, 16, 32],
    epoch_num=10,
    batch_size=256,
    lr=0.001,
    verbose=1,
    random_state=42
)

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

print("Training PyOD AutoEncoder...")
print("Training may take several minutes.")


# ------------------------------------------------------------
# 9. Train the model
# ------------------------------------------------------------
model.fit(X_train)

print("\nAutoEncoder training completed successfully.")


# ------------------------------------------------------------
# 10. Predict anomalies on the test set
# ------------------------------------------------------------
# Prediction values:
# 0 = normal
# 1 = anomaly / suspected fraud

y_pred = model.predict(X_test)

# Continuous anomaly scores.
# Larger scores indicate more unusual transactions.

decision_scores = model.decision_function(X_test)

print("Predictions completed successfully.")


# ------------------------------------------------------------
# 11. Calculate performance metrics
# ------------------------------------------------------------
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    decision_scores
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# 12. Print fraud detection results
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("FRAUD DETECTION RESULTS")
print("=" * 60)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 13. Extract confusion-matrix values
# ------------------------------------------------------------
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 60)
print("CONFUSION MATRIX DETAILS")
print("=" * 60)

print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ------------------------------------------------------------
# 14. Create confusion-matrix chart
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("PyOD AutoEncoder Fraud Detection Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    ["Normal", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Fraud"]
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=12
        )

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 15. Plot anomaly scores
# ------------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.scatter(
    np.arange(len(decision_scores)),
    decision_scores,
    s=8
)

plt.title("AutoEncoder Reconstruction-Based Anomaly Scores")
plt.xlabel("Transaction Index")
plt.ylabel("Anomaly Score")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 16. Compare actual and detected fraud
# ------------------------------------------------------------
actual_fraud = int(y_test.sum())
flagged_fraud = int(y_pred.sum())

print("\n" + "=" * 60)
print("FRAUD COUNT COMPARISON")
print("=" * 60)

print("Actual fraud transactions in test set:", actual_fraud)
print("Transactions flagged as anomalies:", flagged_fraud)
print("Correctly detected fraud cases:", tp)
print("Fraud cases missed by model:", fn)


# ------------------------------------------------------------
# 17. Create summary table
# ------------------------------------------------------------
summary = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives",
        "Actual Fraud Cases",
        "Flagged Fraud Cases"
    ],
    "Result": [
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(roc_auc, 4),
        tn,
        fp,
        fn,
        tp,
        actual_fraud,
        flagged_fraud
    ]
})

print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

display(summary)




# In[ ]:




