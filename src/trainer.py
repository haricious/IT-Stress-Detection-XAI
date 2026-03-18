import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from src.data_loader import generate_it_stress_data

def train_neural_model():
    if not os.path.exists('models'): os.makedirs('models')
    
    print("-> Loading Data...")
    df = generate_it_stress_data()
    X = df.drop('stress_level', axis=1)
    y = df['stress_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # The New Engine: Deep Multi-Layer Perceptron (MLP) Neural Network
    print("-> Training Deep Neural Network (MLP)...")
    mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', max_iter=500, random_state=42)
    mlp_model.fit(X_train, y_train)
    
    # Save Model
    with open('models/stress_neural_model.pkl', 'wb') as f:
        pickle.dump(mlp_model, f)
        
    accuracy = mlp_model.score(X_test, y_test)
    print(f"-> [SUCCESS] Neural Engine Trained. Accuracy: {accuracy * 100:.2f}%")
