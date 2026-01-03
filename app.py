from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
import subprocess

# Add scripts directory to path to import preprocess
base_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(base_dir, "scripts")
sys.path.append(scripts_dir)

app = Flask(__name__)

# Directory paths
models_dir = os.path.join(base_dir, "models")
data_dir = os.path.join(base_dir, "data")

def get_model_status(name):
    path = os.path.join(models_dir, f"{name}.pkl")
    return "Healthy" if os.path.exists(path) else "Offline"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'random_forest': get_model_status('random_forest'),
        'linear_regression': get_model_status('linear_regression'),
        'decision_tree': get_model_status('decision_tree'),
        'data_engine': "Available" if os.path.exists(os.path.join(data_dir, "train.csv")) else "Missing"
    })

@app.route('/api/data/raw', methods=['GET'])
def get_raw_data():
    try:
        path = os.path.join(data_dir, "train.csv")
        df = pd.read_csv(path)
        # Robustly replace NaN with None for JSON compliance
        df = df.replace({np.nan: None})
        return jsonify(df.head(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/data/cleaned', methods=['GET'])
def get_cleaned_data():
    try:
        path = os.path.join(data_dir, "cleaned_train.csv")
        if not os.path.exists(path):
            return jsonify({'error': 'Cleaned data missing. Run Preprocessing.'}), 404
        df = pd.read_csv(path)
        # Robustly replace NaN with None for JSON compliance
        df = df.replace({np.nan: None})
        return jsonify(df.head(10).to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/run/<stage>', methods=['POST'])
def run_stage(stage):
    try:
        script = ""
        model = request.args.get('model', 'random_forest')
        
        if stage == 'preprocess': script = "preprocess.py"
        elif stage == 'train': script = "train.py"
        elif stage == 'predict': script = "predict.py"
        else: return jsonify({'success': False, 'error': 'Invalid stage'}), 400
        
        cmd = [sys.executable, os.path.join(scripts_dir, script)]
        if stage == 'predict':
            cmd.append(model)
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        return jsonify({
            'success': True,
            'output': result.stdout + result.stderr
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/run_pipeline', methods=['POST'])
def run_pipeline():
    try:
        result = subprocess.run([sys.executable, os.path.join(base_dir, "main.py")], 
                                capture_output=True, text=True)
        return jsonify({
            'success': True,
            'output': result.stdout + result.stderr
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_name = data.get('model', 'random_forest')
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': f'Model {model_name} is offline'}), 404
            
        model = joblib.load(model_path)
        train_columns = joblib.load(os.path.join(models_dir, "train_columns.pkl"))
        
        # Prepare input
        features = {k: v for k, v in data.items() if k != 'model'}
        input_data = pd.DataFrame([features])
        
        # Ensure numeric conversion for values
        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='ignore')
            
        input_data = input_data.fillna(0)
        
        # Align with training columns
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=train_columns, fill_value=0)
        
        # Predict
        prediction_usd = model.predict(input_data)[0]
        prediction_tnd = prediction_usd * 2.91
        
        return jsonify({
            'success': True,
            'prediction_usd': round(float(prediction_usd), 2),
            'prediction_tnd': round(float(prediction_tnd), 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        path = os.path.join(data_dir, "train.csv")
        if not os.path.exists(path):
            return jsonify({'error': 'Training data missing.'}), 404
        df = pd.read_csv(path)
        
        # Calculate typical values for the tester
        stats = {
            'OverallQual': int(df['OverallQual'].median()),
            'GrLivArea': int(df['GrLivArea'].mean()),
            'GarageCars': int(df['GarageCars'].mode()[0]),
            'TotalBsmtSF': int(df['TotalBsmtSF'].mean())
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/metrics', methods=['GET'])
def metrics():
    try:
        metrics_path = os.path.join(models_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Metrics not found. Run pipeline first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    try:
        model = request.args.get('model', 'random_forest')
        pred_path = os.path.join(base_dir, "outputs", f"predictions_{model}.csv")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(base_dir, "outputs", "predictions_multi_currency.csv")
            
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            return jsonify(df.head(20).to_dict(orient='records'))
        return jsonify({'error': 'Predictions not found. Run Prediction.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    import json
    app.run(debug=True, port=5000)
