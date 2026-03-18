import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def get_explanation(input_data):
    with open('models/stress_stacking_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # We use the XGBoost component for SHAP as it's tree-based
    sub_model = model.named_estimators_['xgb']
    explainer = shap.TreeExplainer(sub_model)
    shap_values = explainer.shap_values(input_data)
    
    # Save a summary plot for the report
    plt.figure()
    shap.summary_plot(shap_values, input_data, show=False)
    plt.savefig('static/shap_summary.png')
    return shap_values