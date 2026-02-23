import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = pd.read_excel("FULL_FINAL_DATA.xlsx")

# Clean column names
df.columns = (
    df.columns.str.lower()
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

sns.set(style="whitegrid", palette="Set2")

# ---------------------------------------------------------
# GROUP 1: Categorical vs Target (Bar Charts)
# ---------------------------------------------------------

# Expected categorical columns
categorical_features = [
    "occupation_status",
    "product_type",
    "credit_history_years"
]

print("Available columns in dataset:")
print(df.columns.tolist())
print("\nChecking which categorical columns exist...\n")

# Loop through each feature and check if it exists
for feature in categorical_features:
    if feature not in df.columns:
        print(f"❌ Column NOT found: {feature}")
        continue

    print(f"✔ Plotting: {feature}")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=feature, hue="loan_status")
    plt.title(f"{feature.replace('_',' ').title()} vs Loan Status")
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel("Count")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()
