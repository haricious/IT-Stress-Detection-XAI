from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os
import time
from src.data_loader import generate_it_stress_data
from src.trainer import train_neural_model
from src.explainer import generate_xai_explanation 

app = Flask(__name__)

# File Paths
MODEL_PATH = 'models/stress_neural_model.pkl'

def run_system_initialization():
    """Ensures directories and models exist before the server starts."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)
    
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        print("-> [BOOT] Neural Engine missing. Initializing training...")
        generate_it_stress_data()
        train_neural_model()
        print("-> [BOOT] Success.")
    else:
        print("-> [BOOT] Neural Engine online.")

run_system_initialization()

# Load the MLP Neural Network
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    # We pass None so the HTML 'if' statement knows to be 'False'
    return render_template('index.html', xai_plot=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Standard HTML Form Fallback."""
    try:
        feature_names = ['work_hours', 'sleep_hours', 'tech_usage', 'physical_activity', 'social_gap', 'deadline_pressure']
        user_input = [float(request.form[f]) for f in feature_names]
        df_input = pd.DataFrame([user_input], columns=feature_names)
        
        prediction = model.predict(df_input)[0]
        prediction_prob = model.predict_proba(df_input)[0][1]
        
        # Generate XAI Reasoning
        generate_xai_explanation(df_input)
        plot_url = f"/static/plots/xai_waterfall.png?t={int(time.time())}"
        
        status = "high" if prediction == 1 else "stable"
        message = "HIGH STRESS - Intervention Required" if prediction == 1 else "STABLE - Normal Levels"
        detail = f"Confidence: {prediction_prob*100 if prediction==1 else (1-prediction_prob)*100:.1f}%"

        return render_template('index.html', status=status, prediction_text=message, detail=detail, xai_plot=plot_url)
    except Exception as e:
        return render_template('index.html', status="warning", prediction_text="Error", detail=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Main API for the AJAX dashboard."""
    try:
        data = request.get_json()
        feature_names = ['work_hours', 'sleep_hours', 'tech_usage', 'physical_activity', 'social_gap', 'deadline_pressure']
        user_input = [float(data[f]) for f in feature_names]
        df_input = pd.DataFrame([user_input], columns=feature_names)
        
        # Inferences
        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]
        
        # Explainable AI Logic
        generate_xai_explanation(df_input)
        plot_url = f"/static/plots/xai_waterfall.png?t={int(time.time())}"
        
        return jsonify({
            "status": "high" if prediction == 1 else "stable",
            "message": "HIGH STRESS - Intervention Required" if prediction == 1 else "STABLE - Normal Levels",
            "detail": f"Confidence: {prob*100 if prediction==1 else (1-prob)*100:.1f}%. Reasoning plot updated.",
            "xai_plot": plot_url
        })
    except Exception as e:
        return jsonify({"status": "warning", "message": "Neural Engine Error", "detail": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)