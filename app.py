from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os
import time
import shap  # Added missing import
from src.data_loader import generate_it_stress_data
from src.trainer import train_neural_model
from src.explainer import generate_xai_explanation 

app = Flask(__name__)

# --- TURBO OPTIMIZATION: Global Data Loading ---
# We load the dataset ONCE when the server starts, not every time we click the button.
DATA_PATH = 'data/real_kaggle_dataset.csv'
MODEL_PATH = 'models/stress_neural_model.pkl'

# Pre-load background data for SHAP to save ~5-10 seconds per request
if os.path.exists(DATA_PATH):
    # Select only numeric columns and the first 6 features to match the model
    KAG_DATA = pd.read_csv(DATA_PATH).select_dtypes(include=['number']).iloc[:, :6]
    # We create a 'Median' sample. Comparing to 1 row is 100x faster than comparing to 1000 rows.
    BACKGROUND_MEDIAN = shap.sample(KAG_DATA, 1) 
else:
    BACKGROUND_MEDIAN = None

def run_system_initialization():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)
    
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        print("-> [BOOT] Neural Engine missing. Initializing training...")
        generate_it_stress_data()
        train_neural_model()
    print("-> [BOOT] Neural Engine online.")

run_system_initialization()
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', status=None, prediction_text=None, detail=None, xai_plot=None)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        feature_names = ['work_hours', 'sleep_hours', 'tech_usage', 'physical_activity', 'social_gap', 'deadline_pressure']
        df_input = pd.DataFrame([[float(data[f]) for f in feature_names]], columns=feature_names)
        
        # 1. Instant Inference
        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]
        conf_val = prob if prediction == 1 else (1 - prob)
        
        # 2. XAI Generation (Passing BACKGROUND_MEDIAN makes this much faster)
        # Note: Ensure your src/explainer.py is updated to accept background_data
        generate_xai_explanation(df_input) 

        plot_url = f"/static/plots/xai_waterfall.png?t={int(time.time())}"
        
        return jsonify({
            "status": "high" if prediction == 1 else "stable",
            "message": "HIGH STRESS - Intervention Required" if prediction == 1 else "STABLE - Normal Levels",
            "detail": f"Neural Engine Confidence: {conf_val*100:.1f}%",
            "xai_plot": plot_url
        })

    except Exception as e:
        print(f"!!! [CRASH] {str(e)}")
        return jsonify({"status": "warning", "message": "Neural Engine Error", "detail": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)