import joblib  # For saving and loading ML models
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Evaluation metrics

def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier on the training data.

    Parameters:
    - X_train: training features
    - y_train: training target

    Returns:
    - clf: trained model
    """
    # Create a Random Forest model with 100 trees and max depth of 10
    clf = RandomForestClassifier(
        n_estimators=100,      # number of trees in the forest
        max_depth=10,          # maximum depth of trees to avoid overfitting
        random_state=42        # ensures reproducible results
    )

    # Fit the model to the training data
    clf.fit(X_train, y_train)
    print("Model trained successfully!")
    return clf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance on the test set.

    Parameters:
    - model: trained model
    - X_test: test features
    - y_test: test target
    """
    # Predict the target for test data
    y_pred = model.predict(X_test)

    # Print overall accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Print detailed classification metrics (precision, recall, f1-score)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def save_model(model, path='models/random_forest_churn.pkl'):
    """
    Save the trained model to disk for future use.

    Parameters:
    - model: trained model
    - path: path to save the model
    """
    # Save the model using joblib
    joblib.dump(model, path)
    print(f"Model saved at {path}")
