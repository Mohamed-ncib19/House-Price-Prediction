import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")

# Load test data
test_data_path = os.path.join(models_dir, "test_data.pkl")
X_test, y_test = joblib.load(test_data_path)

models = {
    "Linear Regression": joblib.load(os.path.join(models_dir, "linear_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(models_dir, "decision_tree.pkl")),
    "Random Forest": joblib.load(os.path.join(models_dir, "random_forest.pkl"))
}

print("\nMODEL EVALUATION RESULTS\n")

for name, model in models.items():
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{name}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.2f}")
    print("-" * 30)

# Save metrics for web interface
results_data = {}
for name, model in models.items():
    predictions = model.predict(X_test)
    results_data[name.lower().replace(" ", "_")] = {
        "MAE": round(mean_absolute_error(y_test, predictions), 2),
        "MSE": round(mean_squared_error(y_test, predictions), 2),
        "R2": round(r2_score(y_test, predictions), 2)
    }

metrics_json_path = os.path.join(models_dir, "metrics.json")
with open(metrics_json_path, 'w') as f:
    json.dump(results_data, f)
print(f"Dynamic metrics saved to {metrics_json_path}")
