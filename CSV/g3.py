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
# GROUP 3A: Boxplot grouped by Category
# =========================================================

if "product_type" in df.columns and "loan_to_income_ratio" in df.columns:
    print("✔ Plotting: loan_to_income_ratio by product_type")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="product_type", y="loan_to_income_ratio")
    plt.title("Loan To Income Ratio by Product Type")
    plt.xlabel("Product Type")
    plt.ylabel("Loan To Income Ratio")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()
else:
    print("❌ Missing columns for Group 3A")


# =========================================================
# GROUP 3B: Overlaid Histograms (Credit Score by Loan Status)
# =========================================================

if "credit_score" in df.columns and "loan_status" in df.columns:
    print("✔ Plotting: credit_score distribution by loan_status")

    plt.figure(figsize=(7, 4))
    sns.histplot(df[df.loan_status == 0]["credit_score"], color="green", label="Approved", kde=True, alpha=0.5)
    sns.histplot(df[df.loan_status == 1]["credit_score"], color="red", label="Rejected", kde=True, alpha=0.5)
    plt.title("Credit Score Distribution by Loan Status")
    plt.xlabel("Credit Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("❌ Missing columns for Group 3B")


# =========================================================
# GROUP 3C: Scatterplot grouped by Category
# =========================================================

if "annual_income" in df.columns and "loan_amount" in df.columns and "product_type" in df.columns:
    print("✔ Plotting: annual_income vs loan_amount grouped by product_type")

    plt.figure(figsize=(7, 4))
    sns.scatterplot(
        data=df,
        x="annual_income",
        y="loan_amount",
        hue="product_type",
        alpha=0.7
    )
    plt.title("Income vs Loan Amount by Product Type")
    plt.xlabel("Annual Income")
    plt.ylabel("Loan Amount")
    plt.tight_layout()
    plt.show()
else:
    print("❌ Missing columns for Group 3C")
