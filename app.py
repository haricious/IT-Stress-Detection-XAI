from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os
from src.data_loader import generate_it_stress_data
from src.trainer import train_stacking_model

app = Flask(__name__)

# Auto-initialize if model missing
if not os.path.exists('models/stress_stacking_model.pkl'):
    generate_it_stress_data()
    train_stacking_model()

with open('models/stress_stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature_names = ['work_hours', 'sleep_hours', 'tech_usage', 'physical_activity', 'social_gap', 'deadline_pressure']
    user_input = [float(request.form[f]) for f in feature_names]
    df_input = pd.DataFrame([user_input], columns=feature_names)
    
    prediction = model.predict(df_input)[0]
    result = "HIGH STRESS - Intervention Required" if prediction == 1 else "STABLE - Normal Levels"
    color = "text-red-500" if prediction == 1 else "text-green-500"
    
    return render_template('index.html', prediction_text=result, text_color=color)

if __name__ == "__main__":
    app.run(debug=True)