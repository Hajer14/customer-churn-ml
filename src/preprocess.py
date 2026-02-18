import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data():
    """
    Charge le fichier Customer-Churn.csv depuis le dossier data/raw
    en utilisant un chemin relatif basé sur le script.
    """
    # Chemin du dossier du script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
    # Chemin complet vers le fichier CSV
    file_path = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'raw', 'Customer-Churn.csv'))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier n'a pas été trouvé : {file_path}")

    df = pd.read_csv(file_path)
    print("Fichier chargé avec succès !")
    print(df.head())
    return df

def preprocess_data(df):
    """
    Prétraitement du jeu de données :
    - Encode les colonnes catégorielles
    - Sépare features et target
    - Normalise les features numériques
    """
    # Exemple : encode la colonne 'Churn' si c'est la target
    if 'Churn' in df.columns:
        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df['Churn'])

    # Identifier les colonnes numériques et catégorielles
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Encoder les colonnes catégorielles
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Séparer features et target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Prétraitement terminé !")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
