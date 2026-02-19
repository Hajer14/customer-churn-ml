
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """
    Load the trained Random Forest model from disk.
    """
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    return model

def load_trained_columns(columns_path):
    """
    Load the column names that were used during training.
    These are required to align new data for prediction.
    """
    columns = joblib.load(columns_path)
    print("Trained columns loaded successfully!")
    return columns

def preprocess_new_data(df, trained_columns):
    """
    Preprocess new customer data before prediction:
    1. One-hot encode categorical features
    2. Add any missing columns that were in the training set
    3. Keep columns in the exact order of training
    4. Scale numeric features
    """
    df = df.copy()  # Avoid modifying original dataframe

    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Add missing columns with 0 values
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[trained_columns]

    # Scale numeric columns
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

def predict_churn(new_data_path):
    """
    Load new customer CSV, preprocess it, and predict churn.
    """
    # Paths to saved model and training columns
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    model_path = os.path.join(models_dir, 'random_forest_churn.pkl')
    columns_path = os.path.join(models_dir, 'trained_columns.pkl')

    # Load model and training columns
    model = load_model(model_path)
    trained_columns = load_trained_columns(columns_path)

    # Load new customer data
    new_df = pd.read_csv(new_data_path)
    print("New customer data loaded:")
    print(new_df.head())

    # Preprocess new data to match training columns
    X_new = preprocess_new_data(new_df, trained_columns)

    # Predict churn
    predictions = model.predict(X_new)
    new_df['Churn_Prediction'] = predictions

    print("\nPredictions:")
    print(new_df)
    return new_df

if __name__ == "__main__":
    # Example usage: provide path to new customer CSV
    predict_churn('../data/new_customers.csv')
