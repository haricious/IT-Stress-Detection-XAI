import shap
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

def generate_xai_explanation(input_df):
    MODEL_PATH = 'models/stress_neural_model.pkl'
    PLOT_DIR = 'static/plots'
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Load data for background reference
    train_df = pd.read_csv('data/it_stress_data.csv').drop('stress_level', axis=1)
    
    # TURBO FIX 1: Reduce background samples to 10 (Massive speed boost)
    background_data = train_df.sample(10, random_state=42)
    
    # TURBO FIX 2: Use a faster explainer setup
    explainer = shap.Explainer(model.predict, background_data)
    
    # TURBO FIX 3: Limit max_evals (Controls how many permutations it runs)
    # 100-200 is plenty for 6 features. Default is often 500+
    shap_values = explainer(input_df, max_evals=150)

    # Generate Waterfall Plot
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/xai_waterfall.png', bbox_inches='tight')
    plt.close()

    return True