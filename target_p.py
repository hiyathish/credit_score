import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. READ DATA
# ---------------------------------------------------------
df = pd.read_excel("FULL_FINAL_DATA.xlsx")

# Clean column names
df.columns = (
    df.columns.str.lower()
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# ---------------------------------------------------------
# 2. SET VISUAL STYLE
# ---------------------------------------------------------
sns.set(style="whitegrid", palette="Set2")

# ---------------------------------------------------------
# 3. FEATURES TO PLOT
# ---------------------------------------------------------
features = [
    "annual_income",
    "loan_amount",
    "credit_score",
    "debt_to_income_ratio",
    "loan_to_income_ratio"
]

# ---------------------------------------------------------
# 4. CREATE SUBPLOTS (2 rows × 3 columns)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()

# ---------------------------------------------------------
# 5. GENERATE ALL BOXPLOTS WITH SMALLER FONTS
# ---------------------------------------------------------
for i, feature in enumerate(features):
    sns.boxplot(
        x="loan_status",
        y=feature,
        data=df,
        ax=axes[i]
    )
    axes[i].set_title(
        f"{feature.replace('_',' ').title()} vs Loan Status",
        fontsize=12
    )
    axes[i].set_xlabel("Loan Status (0 = Approved, 1 = Rejected)", fontsize=8)
    axes[i].set_ylabel(feature.replace("_", " ").title(), fontsize=8)
    axes[i].tick_params(axis='both', labelsize=7)

# Hide the last empty subplot (6th)
axes[-1].axis("off")

# Add spacing to avoid overlap
plt.tight_layout(pad=2.0, w_pad=2.3, h_pad=2.7)

plt.show()
