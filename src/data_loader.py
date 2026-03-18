import pandas as pd
import numpy as np

def generate_it_stress_data(n_samples=50000):
    np.random.seed(42)
    data = {
        'work_hours': np.random.uniform(4, 14, n_samples),
        'sleep_hours': np.random.uniform(3, 9, n_samples),
        'tech_usage': np.random.uniform(2, 12, n_samples),
        'physical_activity': np.random.uniform(0, 60, n_samples),
        'social_gap': np.random.randint(1, 11, n_samples),
        'deadline_pressure': np.random.randint(1, 11, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Advanced logic for ground truth
    score = (df['work_hours'] * 0.4) - (df['sleep_hours'] * 0.3) + (df['deadline_pressure'] * 0.3)
    df['stress_level'] = (score > 2.5).astype(int)
    
    df.to_csv('data/it_stress_data.csv', index=False)
    return df