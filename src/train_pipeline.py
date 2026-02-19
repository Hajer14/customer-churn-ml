

from preprocess import load_data, preprocess_data
from model import train_model, evaluate_model, save_model
import os
import joblib

def main():
    print("=== Loading data ===")
    df = load_data()

    print("\n=== Preprocessing data ===")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # -------------------------------
    # SAVE TRAINING COLUMNS
    # -------------------------------
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save column names BEFORE converting to NumPy
    trained_columns = X_train.columns.tolist()
    joblib.dump(trained_columns, os.path.join(models_dir, 'trained_columns.pkl'))
    print("Trained column names saved!")

    print("\n=== Training model ===")
    model = train_model(X_train, y_train)

    print("\n=== Evaluating model ===")
    evaluate_model(model, X_test, y_test)

    print("\n=== Saving model ===")
    save_model(model, path=os.path.join(models_dir, 'random_forest_churn.pkl'))

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
