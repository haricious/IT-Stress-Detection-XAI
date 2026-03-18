import pandas as pd
import pickle
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_stacking_model():
    df = pd.read_csv('data/it_stress_data.csv')
    X = df.drop('stress_level', axis=1)
    y = df['stress_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Level 0: Base Models
    estimators = [
        ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(n_estimators=100)),
        ('cat', CatBoostClassifier(iterations=100, verbose=0))
    ]
    
    # Level 1: Meta-Learner
    stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    
    print("Training Stacking Ensemble... this might take a minute.")
    stack_model.fit(X_train, y_train)
    
    # Save Model
    with open('models/stress_stacking_model.pkl', 'wb') as f:
        pickle.dump(stack_model, f)
        
    accuracy = stack_model.score(X_test, y_test)
    print(f"Model trained. Accuracy: {accuracy * 100:.2f}%")
    return stack_model