import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# If running in RapidMiner, replace this with:
# df = input_data
df = pd.read_excel("FULL_FINAL_DATA.xlsx")

# Convert loan_status to readable labels (optional)
df['loan_status'] = df['loan_status'].map({0: 'No Default', 1: 'Default'})

# Plot
plt.figure(figsize=(8, 5))
sns.boxplot(x='loan_status', y='annual_income', data=df, palette='Set2')

plt.title("Annual Income vs Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Annual Income")
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()
