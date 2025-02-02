from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import numpy as np
import polars as pl
import os
import xgboost as xgb

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

def rf_model():
    # Treinar um modelo RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    return clf

def xgb_model():
    # XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    return xgb_clf

def gradient_boosting_model():
    # Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    return gb_clf

def mlp_model():
    # MLP Classifier (Rede Neural Pronta)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
    return mlp_clf

def main():
    df_final = read_data()
    # Separar os dados em treino e teste
    X = np.vstack(df_final['embeddings'].values)
    y = df_final['severity_label'].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load models
    clf = rf_model()
    mlp_clf = mlp_model()
    gb_clf = gradient_boosting_model()
    xgb_clf = xgb_model()

    # Treinando modelos
    print("Treinando modelos:")
    print("RF Classifier")
    clf.fit(X_train, y_train)
    print("MLP Classifier")
    mlp_clf.fit(X_train, y_train)
    print("GB Classifier")
    gb_clf.fit(X_train, y_train)
    print("XGB Classifier")
    xgb_clf.fit(X_train, y_train)

    # Salvar modelos
    save_model(clf, "rf_model_test")
    save_model(mlp_clf, "mlp_model_test")
    save_model(gb_clf, "gb_model_test")
    save_model(xgb_clf, "xgb_model_test")

main()