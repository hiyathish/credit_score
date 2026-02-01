import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, brier_score_loss

# ---------------------------------------------------------
# LOAD FILES FROM SAME DIRECTORY
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

glm = pd.read_csv(os.path.join(script_dir, "GLM.csv"))
gbt = pd.read_csv(os.path.join(script_dir, "GBT.csv"))

glm["model"] = "GLM"
gbt["model"] = "GBT"

# ---------------------------------------------------------
# CLEAN TARGET COLUMN (loan_status)
# TRUE → 1, FALSE → 0
# ---------------------------------------------------------
def convert_target(df):
    df["loan_status"] = df["loan_status"].astype(str).str.upper()
    df["loan_status"] = df["loan_status"].replace({"TRUE": 1, "FALSE": 0})
    df["loan_status"] = df["loan_status"].astype(int)
    return df

glm = convert_target(glm)
gbt = convert_target(gbt)

# ---------------------------------------------------------
# COMBINE DATA
# ---------------------------------------------------------
df = pd.concat([gbt, glm], ignore_index=True)

prob_col = "confidence(true)"

# ---------------------------------------------------------
# GENERIC BINNING (auto-detect min/max)
# ---------------------------------------------------------
min_prob = df[prob_col].min()
max_prob = df[prob_col].max()

bins = np.linspace(min_prob, max_prob, 11)
df["bin"] = pd.cut(df[prob_col], bins=bins, include_lowest=True)

# ---------------------------------------------------------
# CALIBRATION TABLE FUNCTION
# ---------------------------------------------------------
def calibration_table(data):
    return data.groupby("bin").agg(
        avg_predicted_pd=(prob_col, "mean"),
        actual_default_rate=("loan_status", "mean"),
        count=("loan_status", "count")
    ).reset_index()

glm_cal = calibration_table(df[df["model"] == "GLM"])
gbt_cal = calibration_table(df[df["model"] == "GBT"])

# ---------------------------------------------------------
# METRICS: AUC, Brier Score, KS
# ---------------------------------------------------------
def ks_statistic(df):
    df_sorted = df.sort_values(prob_col)
    cum_good = np.cumsum(df_sorted["loan_status"] == 0) / sum(df_sorted["loan_status"] == 0)
    cum_bad = np.cumsum(df_sorted["loan_status"] == 1) / sum(df_sorted["loan_status"] == 1)
    return max(abs(cum_bad - cum_good))

def model_metrics(df_model):
    y_true = df_model["loan_status"]
    y_prob = df_model[prob_col]

    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ks = ks_statistic(df_model)

    return auc, brier, ks

glm_auc, glm_brier, glm_ks = model_metrics(glm)
gbt_auc, gbt_brier, gbt_ks = model_metrics(gbt)

# ---------------------------------------------------------
# PLOT CALIBRATION CURVES
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(glm_cal["avg_predicted_pd"], glm_cal["actual_default_rate"],
         marker="o", label="GLM", color="blue")

plt.plot(gbt_cal["avg_predicted_pd"], gbt_cal["actual_default_rate"],
         marker="o", label="GBT", color="red")

plt.plot([min_prob, max_prob], [min_prob, max_prob],
         linestyle="--", color="black", label="Perfect Calibration")

plt.xlabel("Predicted Probability (PD)")
plt.ylabel("Actual Default Rate")
plt.title("Calibration Curve: GLM vs GBT")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# TEXT REPORT
# ---------------------------------------------------------
def model_summary(name, auc, brier, ks, cal_table):
    return f"""
MODEL: {name}
----------------------------------------
AUC: {auc:.4f}
Brier Score: {brier:.4f}
KS Statistic: {ks:.4f}

Calibration Summary:
Average Predicted PD: {cal_table['avg_predicted_pd'].mean():.4f}
Average Actual Default Rate: {cal_table['actual_default_rate'].mean():.4f}
Total Observations: {cal_table['count'].sum()}
"""

print(model_summary("GLM", glm_auc, glm_brier, glm_ks, glm_cal))
print(model_summary("GBT", gbt_auc, gbt_brier, gbt_ks, gbt_cal))
