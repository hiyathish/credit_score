import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# ---------------------------------------------------------
# LOAD CSV FILES (same directory as script)
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

glm = pd.read_csv(os.path.join(script_dir, "GLM.csv"))
gbt = pd.read_csv(os.path.join(script_dir, "GBT.csv"))

# ---------------------------------------------------------
# CLEAN TARGET COLUMN (TRUE/FALSE → 1/0)
# ---------------------------------------------------------
def clean_target(df):
    df["loan_status"] = df["loan_status"].astype(str).str.upper()
    df["loan_status"] = df["loan_status"].replace({"TRUE": 1, "FALSE": 0})
    df["loan_status"] = df["loan_status"].astype(int)
    return df

glm = clean_target(glm)
gbt = clean_target(gbt)

# ---------------------------------------------------------
# PROBABILITY COLUMN
# ---------------------------------------------------------
prob_col = "confidence(true)"

# ---------------------------------------------------------
# COMPUTE ROC + AUC
# ---------------------------------------------------------
def compute_roc(df):
    y_true = df["loan_status"]
    y_prob = df[prob_col]
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

glm_fpr, glm_tpr, glm_auc = compute_roc(glm)
gbt_fpr, gbt_tpr, gbt_auc = compute_roc(gbt)

# ---------------------------------------------------------
# PLOT ROC CURVES
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

plt.plot(glm_fpr, glm_tpr, color="blue",
         label=f"GLM ROC (AUC = {glm_auc:.4f})")

plt.plot(gbt_fpr, gbt_tpr, color="red",
         label=f"GBT ROC (AUC = {gbt_auc:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="black",
         label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: GLM vs GBT")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
