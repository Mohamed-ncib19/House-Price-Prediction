import pandas as pd
import os

def load_and_preprocess_data():
    # Base directory of the project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "train.csv")
    
    # Load data
    train = pd.read_csv(train_path)
    
    # Drop Id column
    train = train.drop("Id", axis=1)
    
    # Separate target variable
    X = train.drop("SalePrice", axis=1)
    y = train["SalePrice"]
    
    # Handle missing values
    X = X.fillna(X.mean(numeric_only=True))
    X = X.fillna("Unknown")
    
    # One-hot encoding
    X = pd.get_dummies(X)
    
    # Save cleaned data for UI inspection
    cleaned_path = os.path.join(base_dir, "data", "cleaned_train.csv")
    cleaned_df = X.copy()
    cleaned_df['SalePrice'] = y
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")
    
    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    print(X.head())