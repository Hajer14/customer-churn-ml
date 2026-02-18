# src/train_pipeline.py

from preprocess import load_data, preprocess_data
from model import train_model, evaluate_model, save_model
import os

def main():
    """
    Full ML pipeline:
    1. Load the dataset
    2. Preprocess the data (encoding, scaling, train/test split)
    3. Train a Random Forest model
    4. Evaluate the model on the test set
    5. Save the trained model to disk
    """
    print("=== Loading data ===")
    # Load the Customer-Churn dataset
    df = load_data()

    print("\n=== Preprocessing data ===")
    # Preprocess the data: encode, scale, and split
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\n=== Training model ===")
    # Train the Random Forest model
    model = train_model(X_train, y_train)

    print("\n=== Evaluating model ===")
    # Evaluate the model using test set
    evaluate_model(model, X_test, y_test)

    # Ensure the 'models' folder exists to save the trained model
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)  # create folder if it doesn't exist

    print("\n=== Saving model ===")
    # Save the trained model using joblib
    save_model(model, path=os.path.join(models_dir, 'random_forest_churn.pkl'))

    print("\nPipeline completed successfully!")

# If this script is run directly, execute the pipeline
if __name__ == "__main__":
    main()
