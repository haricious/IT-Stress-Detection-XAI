import shap
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Prevents GUI flickering on Windows
import matplotlib.pyplot as plt
import os
from src.data_loader import generate_it_stress_data

def generate_xai_explanation(input_df):
    """
    Upgraded XAI Engine: Explains the Neural Network's decision logic.
    Generates high-quality visual reasoning for the report.
    """
    MODEL_PATH = 'models/stress_neural_model.pkl'
    PLOT_DIR = 'static/plots'
    
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    # 1. Load the MLP Neural Engine
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # 2. Load background data for the Explainer
    # SHAP needs to know what 'normal' looks like to explain 'abnormal' stress
    train_df = generate_it_stress_data().drop('stress_level', axis=1)
    
    # 3. Initialize the SHAP Explainer for Neural Networks
    # We use a small sample of background data to keep it fast
    explainer = shap.Explainer(model.predict, train_df.sample(100, random_state=42))
    
    # 4. Calculate SHAP values for the specific user input
    shap_values = explainer(input_df)

    # 5. UPGRADE: Generate the Waterfall Plot (Local Explanation)
    # This shows the "Why me?" for a single user.
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Reasoning Report: Feature Impact on Stress", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/xai_waterfall.png')
    plt.close()

    # 6. UPGRADE: Generate the Summary Plot (Global Explanation)
    # This shows general trends across the whole IT industry dataset.
    plt.figure(figsize=(10, 6))
    # Recalculate for a larger batch for the summary
    batch_shap = explainer(train_df.sample(50, random_state=42))
    shap.plots.beeswarm(batch_shap, show=False)
    plt.title("Industry-Wide Stress Drivers", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/xai_summary.png')
    plt.close()

    print("-> [XAI SUCCESS] Reasoning plots saved to static/plots/")
    return shap_values
