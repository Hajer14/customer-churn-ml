# src/predict.py

import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """
    Load the trained Random Forest model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    return model

def preprocess_new_data(df, trained_columns):
    """
    Preprocess new data for prediction:
    - One-hot encode categorical features
    - Align columns with training data
    - Scale numeric features
    """
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Add missing columns from training
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only training columns order
    df = df[trained_columns]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    return X_scaled

def predict_churn(new_data_path):
    """
    Load new customer data and predict churn.
    """
    # Paths
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'random_forest_churn.pkl')

    # Load model
    model = load_model(model_path)

    # Load new customer data
    new_df = pd.read_csv(new_data_path)
    print("New customer data loaded:")
    print(new_df.head())

    # Get the columns used in training
    trained_columns = joblib.load(os.path.join(os.path.dirname(model_path), 'trained_columns.pkl'))

    # Preprocess new data
    X_new = preprocess_new_data(new_df, trained_columns)

    # Make predictions
    predictions = model.predict(X_new)
    new_df['Churn_Prediction'] = predictions

    print("\nPredictions:")
    print(new_df)
    return new_df

if __name__ == "__main__":
    # Example usage: replace with your new data CSV
    predict_churn('../data/new_customers.csv')
