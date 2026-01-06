import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")
outputs_dir = os.path.join(base_dir, "outputs")
charts_dir = os.path.join(outputs_dir, "charts")

# Create charts directory if not exists
os.makedirs(charts_dir, exist_ok=True)

# Load test data
test_data_path = os.path.join(models_dir, "test_data.pkl")
X_test, y_test = joblib.load(test_data_path)

models = {
    "Linear Regression": joblib.load(os.path.join(models_dir, "linear_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(models_dir, "decision_tree.pkl")),
    "Random Forest": joblib.load(os.path.join(models_dir, "random_forest.pkl"))
}

print("\nMODEL EVALUATION RESULTS\n")

results_data = {}

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
    
    # Store metrics for JSON
    results_data[name.lower().replace(" ", "_")] = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "R2": round(r2, 2)
    }

    # Generate Actual vs Predicted Chart
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('Actual Prices ($)')
    plt.ylabel('Predicted Prices ($)')
    plt.title(f'{name} - Actual vs Predicted (R² = {r2:.2f})')
    plt.grid(True, alpha=0.3)
    
    # Save chart
    chart_filename = name.lower().replace(" ", "_") + "_chart.png"
    chart_path = os.path.join(charts_dir, chart_filename)
    plt.savefig(chart_path)
    plt.close()
    print(f"Chart saved to {chart_path}")

metrics_json_path = os.path.join(models_dir, "metrics.json")
with open(metrics_json_path, 'w') as f:
    json.dump(results_data, f)
print(f"Dynamic metrics saved to {metrics_json_path}")
