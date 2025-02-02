from sklearn.metrics import classification_report

def predict_model(clf, X_test, y_test):
    # Avaliação do modelo
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
