import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "train.csv")
output_dir = os.path.join(base_dir, "outputs", "analysis")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load data
if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
    exit(1)

df = pd.read_csv(data_path)
print(f"Loaded data: {df.shape}")

# 1. SalePrice Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, bins=30, color='blue')
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price ($)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "price_distribution.png"))
plt.close()
print("Saved price_distribution.png")

# 2. Correlation Heatmap (Top 10 features)
plt.figure(figsize=(12, 10))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])
corr = numeric_df.corr()
k = 10 # number of variables for heatmap
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
            yticklabels=cols.values, xticklabels=cols.values, cmap='coolwarm')
plt.title('Top 10 Features Correlated with Sale Price')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()
print("Saved correlation_heatmap.png")

# 3. Scatter Plots for Key Features
key_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']

for feature in key_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[feature], y=df['SalePrice'], alpha=0.6)
    plt.title(f'{feature} vs Sale Price')
    plt.xlabel(feature)
    plt.ylabel('Sale Price ($)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"scatter_{feature}.png"))
    plt.close()
    print(f"Saved scatter_{feature}.png")

print(f"\nAnalysis complete. Charts saved to {output_dir}")
