

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path='../data/raw/Customer-Churn.csv'):
    """
    Load the Customer Churn dataset CSV
    """
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """
    Preprocess the dataset:
    - Encode categorical columns with one-hot
    - Scale numeric columns
    - Split into train and test sets
    Returns:
        X_train, X_test, y_train, y_test (DataFrames for X)
    """
    df = df.copy()

    # Target
    y = df['Churn'].map({'Yes':1, 'No':0})
    X = df.drop('Churn', axis=1)

    # One-hot encode categorical features
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Scale numeric features
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Prétraitement terminé !")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test
