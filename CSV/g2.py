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

print("Available columns:")
print(df.columns.tolist())
print("\n")

# =========================================================
# GROUP 2: Numeric vs Numeric (Scatterplots)
# =========================================================

numeric_pairs = [
    ("annual_income", "loan_amount"),
    ("annual_income", "current_debt"),
    ("loan_amount", "interest_rate"),
    ("credit_score", "interest_rate")
]

print("Checking numeric pairs...\n")

for x, y in numeric_pairs:
    if x not in df.columns:
        print(f"❌ Column NOT found: {x}")
        continue
    if y not in df.columns:
        print(f"❌ Column NOT found: {y}")
        continue

    print(f"✔ Plotting: {x} vs {y}")

    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
    plt.title(f"{x.replace('_',' ').title()} vs {y.replace('_',' ').title()}")
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.tight_layout()
    plt.show()
