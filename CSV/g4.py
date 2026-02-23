import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel("FULL_FINAL_DATA.xlsx")

# Clean column names
df.columns = (
    df.columns.str.lower()
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

sns.set(style="whitegrid", palette="Set2")

# =========================================================
# GROUP 4: Distribution Overlap (Overlaid Histograms)
# =========================================================

overlap_features = [
    "credit_score",
    "annual_income",
    "debt_to_income_ratio"
]

for feature in overlap_features:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[df.loan_status == 0][feature], color="blue", label="Approved", kde=True, alpha=0.5)
    sns.histplot(df[df.loan_status == 1][feature], color="orange", label="Rejected", kde=True, alpha=0.5)
    plt.title(f"{feature.replace('_',' ').title()} Distribution by Loan Status")
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
