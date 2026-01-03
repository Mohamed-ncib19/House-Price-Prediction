from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Base directory of the project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")

# Create models directory if not exists
os.makedirs(models_dir, exist_ok=True)

# Load and split data
X, y = load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize models
models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(random_state=42),
    "random_forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    model_path = os.path.join(models_dir, f"{name}.pkl")
    joblib.dump(model, model_path)
    print(f"{name} trained and saved to {model_path}")

# Save test split for evaluation
test_data_path = os.path.join(models_dir, "test_data.pkl")
joblib.dump((X_test, y_test), test_data_path)
print(f"Test data saved to {test_data_path}")

# Save columns for prediction consistency
columns_path = os.path.join(models_dir, "train_columns.pkl")
joblib.dump(X.columns.tolist(), columns_path)
print(f"Train columns saved to {columns_path}")
