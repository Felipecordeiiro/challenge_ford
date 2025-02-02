from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import numpy as np
import polars as pl
import os

# PATH DATA
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA')
SAVE_MODELS = os.getenv('SAVE_MODEL')

def save_model(model, nome_modelo):
    # Salvando o modelo
    dump(model, f'{os.path.join(SAVE_MODELS, nome_modelo)}.joblib')
    print(f"Modelo salvo em {SAVE_MODELS}")

def read_data() -> pl.DataFrame:
    all_data = []
    
    for root,dirs,files in os.walk(PROCESSED_DATA_PATH):
        for file in files:
            if file.endswith(".csv"):
                df = pl.scan_csv(os.path.join(root, file))
                all_data.append(df)

    df_final = pl.concat(all_data).collect()

    return df_final                

def main():
    df_final = read_data()
    # Separar os dados em treino e teste
    X = np.vstack(df_final['embeddings'].values)
    y = df_final['severity_label'].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar um modelo RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    save_model(clf, "modelo_teste")