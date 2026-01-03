# 🏙️ Skyline House Price Prediction Console

An intelligent, end-to-end Machine Learning system designed to predict residential house prices using advanced regression techniques. This project features a premium glassmorphic web dashboard for real-time model monitoring, data exploration, and interactive testing.

![Dashboard Overview](C:/Users/Ncib/.gemini/antigravity/brain/9400ca84-b586-4ad9-89d7-9d2b8baeb6e1/dashboard_overview_1767482475701.png)

---

## 🚀 Key Features

- **🧠 Intelligent Tester**: Automatically populates inputs with data-driven statistics (median/mean) from the training data for realistic testing.
- **📊 Dynamic Data Explorer**: Real-time inspection of your dataset across all stages:
    - **Raw**: The original unprocessed data.
    - **Cleaned**: Features after one-hot encoding and null handling.
    - **Predictions**: Comparison of model outputs.
- **⚡ Engine Control Center**: Granular control over the ML pipeline (Preprocess -> Train -> Predict) with live console feedback.
- **🛡️ Health Monitoring**: Real-time status badges for data integrity and model "online" status.
- **📈 Performance Metrics**: Live R² scores, MAE, and MSE tracking for multiple engines.
- **🇹🇳 Dual Currency**: Instant conversion between **USD ($)** and **Tunisian Dinar (TND)** at 1 USD = 2.91 TND.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Pandas, Scikit-Learn, Joblib, NumPy
- **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS (Glassmorphism UI)
- **Deployment**: Local Flask Server

---

## 🤖 AI Models

The system evaluates three distinct regression engines to find the best fit:

1. **Random Forest (Champion)**: Highest accuracy with an **R² of ~0.89**.
2. **Decision Tree**: Reliable but less precise (R² ~0.77).
3. **Linear Regression**: Baseline statistical model (R² ~0.44).

### Model Performance Comparison

| Model | R² Score | MAE | MSE | Best For |
|-------|----------|-----|-----|----------|
| Random Forest | 0.89 | ~$20,000 | ~800M | Production use - highest accuracy |
| Decision Tree | 0.77 | ~$28,000 | ~1.2B | Quick predictions with good accuracy |
| Linear Regression | 0.44 | ~$38,000 | ~2.5B | Baseline comparison |

---

## 📁 Project Structure

```text
house-price-prediction/
├── app.py                      # Main Flask Backend API
├── main.py                     # One-tap CLI Pipeline runner
├── requirements.txt            # Python dependencies
├── scripts/                    # ML Pipeline scripts
│   ├── preprocess.py          # Data cleaning & encoding
│   ├── train.py               # Model training (all 3 models)
│   ├── evaluate.py            # Metric generation
│   └── predict.py             # Bulk prediction engine
├── models/                     # Serialized binary models (.pkl)
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── linear_regression.pkl
│   ├── train_columns.pkl      # Feature alignment
│   ├── test_data.pkl          # Test split for evaluation
│   └── metrics.json           # Performance metrics
├── data/                       # Input datasets (.csv)
│   ├── train.csv              # Training data (1460 houses)
│   ├── test.csv               # Test data for predictions
│   └── cleaned_train.csv      # Preprocessed training data
├── outputs/                    # Model results & CSVs
│   ├── predictions_random_forest.csv
│   ├── predictions_decision_tree.csv
│   ├── predictions_linear_regression.csv
│   └── predictions_multi_currency.csv
├── templates/                  # Web interface
│   └── index.html             # Glassmorphic dashboard
└── venv/                       # Python virtual environment
```

---

## 🏁 Getting Started

### 1. Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 2. Setup Environment

```powershell
# Clone or navigate to the project directory
cd house-price-prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place your dataset files in the `data/` directory:
- `train.csv` - Training data with SalePrice
- `test.csv` - Test data for predictions

The project uses the **Kaggle House Prices** dataset by default.

### 4. Run the Full Pipeline (CLI)

Execute the entire ML pipeline in one command:

```powershell
python main.py
```

This will:
1. ✅ Preprocess the data (clean, encode, save)
2. ✅ Train all 3 models (Random Forest, Decision Tree, Linear Regression)
3. ✅ Evaluate models and save metrics
4. ✅ Generate predictions on test data

### 5. Launch the Web Dashboard

```powershell
python app.py
```

Open your browser at **`http://127.0.0.1:5000`** to access the interactive console.

---

## 🎯 Using the Web Interface

### Dashboard Overview

The web interface provides comprehensive control over the ML pipeline:

#### 1. **System Health Monitor**
- Real-time status of all models (Healthy/Offline)
- Data engine availability check
- Green/red indicators for quick status assessment

#### 2. **Performance Metrics Dashboard**
- Live R² scores with visual progress bars
- MAE and MSE values for each model
- Automatic updates after training

#### 3. **Engine Control Center**
Execute pipeline stages individually:
- **Preprocess**: Clean and encode raw data
- **Train**: Train all three models
- **Predict**: Generate predictions using selected model

#### 4. **Data Explorer**
Toggle between three views:
- **Raw**: Original unprocessed data
- **Cleaned**: Preprocessed features (one-hot encoded)
- **Results**: Prediction outputs with dual currency

#### 5. **Interactive Tester**

![Prediction Result](C:/Users/Ncib/.gemini/antigravity/brain/9400ca84-b586-4ad9-89d7-9d2b8baeb6e1/prediction_result_1767482491330.png)

Test individual predictions with custom inputs:
- Select model (Random Forest recommended)
- Input house features:
  - **OverallQual**: Overall material and finish quality (1-10)
  - **GrLivArea**: Above ground living area (sq ft)
  - **GarageCars**: Garage capacity (number of cars)
  - **TotalBsmtSF**: Total basement area (sq ft)
- Get instant predictions in USD and TND

---

## 📊 ML Pipeline Details

### 1. Data Preprocessing (`preprocess.py`)

**Objective**: Clean and prepare raw data for model training.

**Steps**:
- Load `train.csv` dataset
- Remove ID column
- Separate target variable (`SalePrice`)
- Handle missing values:
  - Numeric: Fill with column mean
  - Categorical: Fill with "Unknown"
- Apply one-hot encoding to categorical features
- Save cleaned data to `data/cleaned_train.csv`

**Output**: Feature matrix `X` and target vector `y`

### 2. Model Training (`train.py`)

**Objective**: Train multiple regression models and save them for deployment.

**Models Trained**:
1. **Linear Regression**: Simple baseline model
2. **Decision Tree Regressor**: Non-linear tree-based model
3. **Random Forest Regressor**: Ensemble of 100 decision trees

**Process**:
- Load preprocessed data
- Split into train/test sets (80/20 split, random_state=42)
- Train each model on training data
- Save trained models as `.pkl` files
- Save test split for evaluation
- Save feature columns for prediction alignment

**Outputs**:
- `models/random_forest.pkl`
- `models/decision_tree.pkl`
- `models/linear_regression.pkl`
- `models/train_columns.pkl`
- `models/test_data.pkl`

### 3. Model Evaluation (`evaluate.py`)

**Objective**: Assess model performance using standard regression metrics.

**Metrics Calculated**:
- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **MSE (Mean Squared Error)**: Squared error (penalizes large errors)
- **R² Score**: Proportion of variance explained (0-1, higher is better)

**Process**:
- Load test data split
- Load all trained models
- Generate predictions on test set
- Calculate metrics for each model
- Save results to `models/metrics.json` for web dashboard

**Output**: Console report + JSON metrics file

### 4. Prediction Generation (`predict.py`)

**Objective**: Generate price predictions for new houses.

**Process**:
- Load selected model (default: Random Forest)
- Load `test.csv` dataset
- Preprocess test data (same steps as training)
- Align features with training columns
- Generate predictions in USD
- Convert to TND (USD × 2.91)
- Save results to CSV

**Outputs**:
- `outputs/predictions_<model_name>.csv`
- `outputs/predictions_multi_currency.csv`

---

## 🌐 API Endpoints

The Flask backend exposes the following REST API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main dashboard HTML |
| `/api/health` | GET | Get model and data engine status |
| `/api/data/raw` | GET | Fetch first 10 rows of raw data |
| `/api/data/cleaned` | GET | Fetch first 10 rows of cleaned data |
| `/api/predictions?model=<name>` | GET | Fetch prediction results for specific model |
| `/api/stats` | GET | Get median/mean values for form defaults |
| `/api/metrics` | GET | Get R², MAE, MSE for all models |
| `/api/run/<stage>?model=<name>` | POST | Execute pipeline stage (preprocess/train/predict) |
| `/api/run_pipeline` | POST | Execute full pipeline (all stages) |
| `/api/predict` | POST | Get single prediction for custom inputs |

---

## 📊 Evaluation Metrics Explained

### R² Score (Coefficient of Determination)
- **Range**: 0 to 1 (higher is better)
- **Meaning**: Percentage of variance in house prices explained by the model
- **Example**: R² = 0.89 means the model explains 89% of price variation

### MAE (Mean Absolute Error)
- **Unit**: Dollars ($)
- **Meaning**: Average prediction error
- **Example**: MAE = $20,000 means predictions are off by $20k on average

### MSE (Mean Squared Error)
- **Unit**: Dollars² ($²)
- **Meaning**: Squared error (heavily penalizes large mistakes)
- **Use**: Comparing models (lower is better)

---

## 🎨 UI Design Features

The web interface showcases modern design principles:

- **Glassmorphism**: Frosted glass effect with backdrop blur
- **Dark Theme**: Professional dark background (#0b0f1a)
- **Gradient Text**: Blue-purple gradient for headers
- **Smooth Animations**: Fade-in effects and hover transitions
- **Responsive Layout**: Grid-based layout adapts to screen size
- **Custom Scrollbars**: Minimal, semi-transparent scrollbars
- **Status Indicators**: Glowing green/red dots for health status
- **Typography**: Outfit font family for modern aesthetics

---

## 🔧 Customization

### Change Currency Conversion Rate

Edit `scripts/predict.py` and `app.py`:

```python
USD_TO_TND_RATE = 2.91  # Change this value
```

### Add More Models

Edit `scripts/train.py`:

```python
models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(random_state=42)  # Add new model
}
```

### Modify Input Features

Edit the form in `templates/index.html` and update the preprocessing logic accordingly.

---

## 🐛 Troubleshooting

### Models show "Offline" status
**Solution**: Run the training pipeline first:
```powershell
python main.py
```

### "Cleaned data missing" error
**Solution**: Run preprocessing:
```powershell
python scripts/preprocess.py
```

### Predictions not found
**Solution**: Run prediction script:
```powershell
python scripts/predict.py random_forest
```

### Port 5000 already in use
**Solution**: Change port in `app.py`:
```python
app.run(debug=True, port=5001)  # Use different port
```

---

## 📦 Dependencies

```
flask
pandas
numpy
scikit-learn
joblib
```

Install all at once:
```powershell
pip install -r requirements.txt
```

---

## 🎓 Educational Value

This project demonstrates:

1. **End-to-End ML Pipeline**: From raw data to deployed predictions
2. **Model Comparison**: Evaluating multiple algorithms
3. **Web Integration**: Connecting ML models to web interfaces
4. **RESTful API Design**: Building backend endpoints
5. **Data Preprocessing**: Handling missing values and encoding
6. **Model Serialization**: Saving and loading trained models
7. **Interactive UI**: Creating user-friendly ML interfaces

---

## 📈 Future Enhancements

- [ ] Add more regression models (XGBoost, LightGBM)
- [ ] Implement feature importance visualization
- [ ] Add model retraining functionality
- [ ] Export predictions to Excel
- [ ] Add user authentication
- [ ] Deploy to cloud platform (Heroku, AWS)
- [ ] Add data upload functionality
- [ ] Implement A/B testing for models

---

## 📜 License

MIT License. Created for educational and demonstration purposes.

---

## 👨‍💻 Author

Developed as a comprehensive demonstration of machine learning engineering and full-stack development skills.

---

## 🙏 Acknowledgments

- **Dataset**: [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Framework**: Flask for Python web development
- **ML Library**: Scikit-Learn for machine learning algorithms
- **UI Framework**: Tailwind CSS for modern styling

---

## 📞 Support

For questions or issues:
1. Check the Troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are installed correctly
4. Verify dataset files are in the correct location

---

**Happy Predicting! 🏠📊**
