import pandas as pd
import joblib
import os

# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")
outputs_dir = os.path.join(base_dir, "outputs")

import sys

# Get model name from argument or default to random_forest
model_name = sys.argv[1] if len(sys.argv) > 1 else "random_forest"

# Load trained model
model_path = os.path.join(models_dir, f"{model_name}.pkl")
if not os.path.exists(model_path):
    print(f"Error: Model {model_name} not found at {model_path}")
    sys.exit(1)

model = joblib.load(model_path)

# Load test data
test_path = os.path.join(base_dir, "data", "test.csv")
test = pd.read_csv(test_path)

# Drop ID column
test_ids = test["Id"]
test.drop("Id", axis=1, inplace=True)

# Handle missing values
test = test.fillna(test.mean(numeric_only=True))
test = test.fillna("Unknown")

# Encode categorical variables
test = pd.get_dummies(test)

# Align columns with training data
train_columns_path = os.path.join(models_dir, "train_columns.pkl")
train_columns = joblib.load(train_columns_path)
test = test.reindex(columns=train_columns, fill_value=0)

# Predict prices
predictions_usd = model.predict(test)
USD_TO_TND_RATE = 2.91
predictions_tnd = predictions_usd * USD_TO_TND_RATE

# Save output
os.makedirs(outputs_dir, exist_ok=True)
output = pd.DataFrame({
    "Id": test_ids,
    "SalePrice_USD": predictions_usd,
    "SalePrice_TND": predictions_tnd
})

# Save to general file AND model-specific file
output_file = os.path.join(outputs_dir, "predictions_multi_currency.csv")
model_specific_file = os.path.join(outputs_dir, f"predictions_{model_name}.csv")

output.to_csv(output_file, index=False)
output.to_csv(model_specific_file, index=False)

print(f"Predictions for {model_name} saved to {model_specific_file}")
print(f"Formula used: TND = USD * {USD_TO_TND_RATE}")
