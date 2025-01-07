import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score
import joblib
from sklearn.utils import resample

def mlp_binary_classification(df: pd.DataFrame , target_column: str, test_size = 0.2, random_state = 42):
        X = df.drop(target_column, axis = 1)
        y = df[target_column]
        y = y.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        mlp_pipeline = Pipeline([
            ('scaler',  StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=random_state)),
        ])
        mlp_pipeline.fit(X_train,y_train)
        y_pred = mlp_pipeline.predict(X_test)
        print('Informe de clasificación')
        print(classification_report(y_test, y_pred))
        print('Matriz de confusión')
        print(confusion_matrix(y_test, y_pred))
        return mlp_pipeline
    

def xgboost_bayesopt_classifier(df, target_column, test_size=0.2, random_state=42, n_iter=50):
    # Verificar si la columna objetivo existe en el DataFrame
    if target_column not in df.columns:
        raise ValueError(f"La columna '{target_column}' no existe en el DataFrame.")

    # Separar características (X) y variable objetivo (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Asegurarse de que la variable objetivo sea numérica y binaria
    unique_values = y.unique()
    if len(unique_values) != 2:
        raise ValueError(f"La variable objetivo debe ser binaria. Valores encontrados: {unique_values}")
    y = y.map({unique_values[0]: 0, unique_values[1]: 1})

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Crear un pipeline con escalado de características y XGBoost
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state))
    ])

    # Definir el espacio de búsqueda para los hiperparámetros
    search_spaces = {
        'xgb__learning_rate': Real(0.01, 1.0, 'log-uniform'),
        'xgb__max_depth': Integer(3, 10),
        'xgb__min_child_weight': Real(0.5, 10, 'log-uniform'),
        'xgb__subsample': Real(0.5, 1.0, 'uniform'),
        'xgb__colsample_bytree': Real(0.5, 1.0, 'uniform'),
        'xgb__gamma': Real(1e-9, 0.5, 'log-uniform'),
        'xgb__n_estimators': Integer(100, 1000)
    }

    # Configurar BayesSearchCV
    bayes_search = BayesSearchCV(
        xgb_pipeline,
        search_spaces,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )

    # Realizar la búsqueda de hiperparámetros
    bayes_search.fit(X_train, y_train)

    # Imprimir los mejores hiperparámetros encontrados
    print("Mejores hiperparámetros encontrados:")
    print(bayes_search.best_params_)

    # Evaluar el modelo en el conjunto de prueba
    y_pred = bayes_search.predict(X_test)

    # Imprimir el informe de clasificación y la matriz de confusión
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    return bayes_search

def train_broader_hyperparams(df: pd.DataFrame, target: str, features: list, 
                              test_size: float=0.2, random_state: int=42, suffix: str=""):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models_params = {
        'svc': (
            Pipeline([('scaler', StandardScaler()), ('svc', SVC())]),
            {
                'svc__C': [0.1, 1, 10, 100],
                'svc__kernel': ['linear', 'rbf', 'poly'],
                'svc__gamma': ['scale', 'auto']
            }
        ),
        'xgb': (
            Pipeline([('scaler', StandardScaler()),
                      ('xgb', XGBClassifier(
                                            eval_metric='logloss', 
                                            random_state=random_state))]),
            {
                'xgb__learning_rate': [0.01, 0.1, 0.2],
                'xgb__n_estimators': [100, 300, 500],
                'xgb__max_depth': [3, 6, 10]
            }
        ),
        'rf': (
            Pipeline([('scaler', StandardScaler()),
                      ('rf', RandomForestClassifier(random_state=random_state))]),
            {
                'rf__n_estimators': [100, 200, 300],
                'rf__max_depth': [5, 10, 20]
            }
        ),
    }

    for name, (pipe, params) in models_params.items():
        search = GridSearchCV(pipe, params, cv=5, n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions, average='binary')
        
        # Calculate 95% confidence intervals using bootstrapping
        n_iterations = 1000
        accuracies = []
        recalls = []
        for _ in range(n_iterations):
            X_resample, y_resample = resample(X_test, y_test)  # remove random_state to vary each iteration
            y_resample_pred = best_model.predict(X_resample)
            accuracies.append(accuracy_score(y_resample, y_resample_pred))
            recalls.append(recall_score(y_resample, y_resample_pred, average='binary'))
        
        acc_lower, acc_upper = np.percentile(accuracies, [0.5, 99.5])
        rec_lower, rec_upper = np.percentile(recalls, [0.5, 99.5])
        
        acc_margin = (acc_upper - acc_lower) / 2
        rec_margin = (rec_upper - rec_lower) / 2
        print(
            f"{name.upper()}: Accuracy={accuracy:.3f} ± {acc_margin:.3f} (95% CI: {acc_lower:.3f}-{acc_upper:.3f}), "
            f"Recall={recall:.3f} ± {rec_margin:.3f} (95% CI: {rec_lower:.3f}-{rec_upper:.3f})"
        )
        
        joblib.dump(best_model, f"models/{name}_pipeline_{suffix}.pkl")
