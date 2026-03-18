import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd
import pickle
import os

def generate_xai_explanation(input_df):
    try:
        # 1. PATH SETUP
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        PLOT_DIR = os.path.join(BASE_DIR, 'static', 'plots')
        os.makedirs(PLOT_DIR, exist_ok=True)
        
        DATA_PATH = os.path.join(BASE_DIR, 'data', 'real_kaggle_dataset.csv')
        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'stress_neural_model.pkl')
        SAVE_PATH = os.path.join(PLOT_DIR, 'xai_waterfall.png')

        # 2. LOAD MODEL
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        # 3. LOAD & ALIGN KAGGLE DATA
        # We only take the columns that exist in your input_df (the 6 form features)
        full_kaggle = pd.read_csv(DATA_PATH)
        
        # This is the magic line: it filters the CSV to match your model's 6 features
        # If the Kaggle column names are different (e.g., 'Work_Hours' vs 'work_hours'), 
        # make sure they match your form IDs!
        feature_cols = input_df.columns.tolist() 
        
        try:
            train_df = full_kaggle[feature_cols]
        except KeyError:
            # If names don't match, we'll just take the first 6 numeric columns as a fallback
            train_df = full_kaggle.select_dtypes(include=['number']).iloc[:, :6]
            train_df.columns = feature_cols # Force name alignment

        # 4. SHAP MATH
        print(f"-> [XAI] Using {len(train_df.columns)} features from Kaggle for background.")
        background = shap.sample(train_df, 10) 
        explainer = shap.Explainer(model.predict, background)
        
        # Generate values
        shap_values = explainer(input_df, max_evals=150)

        # 5. PLOT & SAVE
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        
        plt.tight_layout()
        plt.savefig(SAVE_PATH, bbox_inches='tight', dpi=100)
        plt.close('all')
        
        print(f"-> [SUCCESS] Plot saved to {SAVE_PATH}")
        return True

    except Exception as e:
        print(f"!!! [XAI ERROR] {str(e)}")
        return False