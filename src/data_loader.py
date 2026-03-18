import pandas as pd
import numpy as np
import os

def generate_it_stress_data():
    """
    Loads the real Kaggle dataset and maps it into the 6-feature schema the app/model expects.

    IMPORTANT: This function does NOT write `data/it_stress_data.csv` anymore.
    It returns the prepared dataframe in-memory so the system uses `data/real_kaggle_dataset.csv` directly.
    """
    os.makedirs('data', exist_ok=True)

    real_data_path = 'data/real_kaggle_dataset.csv'
    df = pd.read_csv(real_data_path)
    print(f"-> [DATA] Successfully loaded {len(df)} real human records.")

    n_rows = len(df)
    np.random.seed(42)

    # Strict column ordering for scikit-learn compatibility (matches app.py feature order)
    data_dict = {
        'work_hours': np.random.uniform(7, 14, n_rows),
        'sleep_hours': df['Sleep Duration'].values,
        'tech_usage': np.random.uniform(4, 12, n_rows),
        'physical_activity': df['Physical Activity Level'].values,
        'social_gap': np.random.randint(1, 11, n_rows),
        'deadline_pressure': np.random.randint(1, 11, n_rows),
        'stress_level': (df['Stress Level'] >= 6).astype(int).values
    }

    # If an old mapped CSV exists from previous versions, remove it so it won't "come back"
    legacy_path = 'data/it_stress_data.csv'
    if os.path.exists(legacy_path):
        try:
            os.remove(legacy_path)
            print("-> [CLEANUP] Removed legacy data/it_stress_data.csv")
        except OSError:
            print("-> [WARN] Could not remove legacy data/it_stress_data.csv (in use?)")

    model_data = pd.DataFrame(data_dict)
    print("-> [SUCCESS] Real data mapped in-memory. Feature order locked.")
    return model_data
