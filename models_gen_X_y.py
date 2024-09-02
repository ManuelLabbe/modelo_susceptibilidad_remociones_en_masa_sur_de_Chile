import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay    
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint

def xgboost_random_search(X, y, param_grid, test_size=0.2, n_iter=50, random_state=42):
    """
    Realiza una búsqueda aleatoria de hiperparámetros para XGBoost y evalúa el mejor modelo.

    Parámetros:
    X : array-like de forma (n_muestras, n_características)
        Las muestras de entrenamiento.
    y : array-like de forma (n_muestras,)
        Los valores objetivo.
    param_grid : dict
        Diccionario con parámetros como claves y listas de valores de parámetros para buscar.
    test_size : float, opcional (por defecto=0.2)
        Proporción del conjunto de datos a incluir en la división de prueba.
    n_iter : int, opcional (por defecto=50)
        Número de combinaciones de parámetros a probar.
    random_state : int, opcional (por defecto=42)
        Controla la aleatoriedad del experimento.

    Retorna:
    dict : Un diccionario conteniendo el mejor modelo, la precisión y los mejores parámetros.
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f'Número de muestras de entrenamiento: {X_train.shape[0]}, Número de muestras de prueba: {X_test.shape[0]}')

    # Inicializar el modelo XGBoost
    model = XGBClassifier(eval_metric='logloss', random_state=random_state)

    # Configurar y ejecutar la búsqueda aleatoria
    random_search = RandomizedSearchCV(
        model, 
        param_distributions=param_grid, 
        n_iter=n_iter, 
        scoring='accuracy', 
        cv=5, 
        verbose=1, 
        random_state=random_state
    )
    random_search.fit(X_train, y_train)

    # Obtener el mejor modelo
    best_model = random_search.best_estimator_

    # Hacer predicciones y calcular la precisión
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f'Precisión: {acc*100:.2f}%')
    plot_confusion_matrix(best_model, X_test, y_test)
    # Retornar los resultados
    return {
        'best_model': best_model,
        'accuracy': acc,
        'best_params': random_search.best_params_
    }

"""    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 1.0, 10.0]
    }"""
    

def mlp_binary_classification(X, y, test_size=0.2, random_state=42):
    """
    Realiza una clasificación binaria utilizando un Perceptrón Multicapa (MLP).

    :param X: np.ndarray, características de entrada
    :param y: np.ndarray, etiquetas de clase (deben ser 0 y 1)
    :param test_size: float, proporción del conjunto de prueba
    :param random_state: int, semilla aleatoria para reproducibilidad
    :return: Pipeline, el modelo MLP entrenado
    """
    # Asegurar que y sea de tipo int
    y = y.astype(int)

    # Verificar que y contiene solo 0 y 1
    unique_values = np.unique(y)
    if not np.array_equal(unique_values, [0, 1]):
        raise ValueError(f"y debe contener solo 0 y 1. Valores encontrados: {unique_values}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=random_state)),
    ])

    mlp_pipeline.fit(X_train, y_train)
    y_pred = mlp_pipeline.predict(X_test)

    print('Informe de clasificación')
    print(classification_report(y_test, y_pred))
    print('Matriz de confusión')
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(mlp_pipeline, X_test, y_test)
    return mlp_pipeline

def xgboost_bayesopt_classifier(X, y, test_size=0.2, random_state=42, n_iter=50):
    """
    Realiza una clasificación binaria utilizando XGBoost con optimización Bayesiana de hiperparámetros.

    :param X: np.ndarray, características de entrada
    :param y: np.ndarray, etiquetas de clase (deben ser 0 y 1)
    :param test_size: float, proporción del conjunto de prueba
    :param random_state: int, semilla aleatoria para reproducibilidad
    :param n_iter: int, número de iteraciones para la búsqueda de hiperparámetros
    :return: BayesSearchCV, el modelo XGBoost optimizado
    """
    # Verificar que y contiene solo 0 y 1
    unique_values = np.unique(y)
    if not np.array_equal(unique_values, [0, 1]):
        raise ValueError(f"y debe contener solo 0 y 1. Valores encontrados: {unique_values}")

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Crear un pipeline con escalado de características y XGBoost
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state))
    ])

    # Definir el espacio de búsqueda para los hiperparámetros
    search_spaces = {
        'xgb__learning_rate': sp_uniform(0.01, 0.99),
        'xgb__max_depth': sp_randint(3, 11),
        'xgb__min_child_weight': sp_uniform(0.5, 9.5),
        'xgb__subsample': sp_uniform(0.5, 0.5),
        'xgb__colsample_bytree': sp_uniform(0.5, 0.5),
        'xgb__gamma': sp_uniform(0, 0.5),
        'xgb__n_estimators': sp_randint(100, 901)
    }

    # Configurar RandomizedSearchCV (usado en lugar de BayesSearchCV para compatibilidad)
    random_search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=search_spaces,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )

    # Realizar la búsqueda de hiperparámetros
    random_search.fit(X_train, y_train)

    # Imprimir los mejores hiperparámetros encontrados
    print("Mejores hiperparámetros encontrados:")
    print(random_search.best_params_)

    # Evaluar el modelo en el conjunto de prueba
    y_pred = random_search.predict(X_test)

    # Imprimir el informe de clasificación y la matriz de confusión
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(random_search, X_test, y_test)
    return random_search

def plot_confusion_matrix(model, X_test, y_test):
    """
    Genera y muestra una matriz de confusión simple para un modelo de clasificación.

    Parámetros:
    model : objeto con método predict
        El modelo de clasificación entrenado.
    X_test : array-like de forma (n_muestras, n_características)
        Datos de prueba para generar predicciones.
    y_test : array-like de forma (n_muestras,)
        Etiquetas verdaderas de los datos de prueba.
    """
    # Generar predicciones
    predictions = model.predict(X_test)

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, predictions)

    # Crear y mostrar la visualización
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Matriz de Confusión')
    plt.show()
    
    import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def svm_with_hyperparameter_tuning(X, y, test_size=0.2, random_state=42):
    """
    Crea un modelo SVM con ajuste de hiperparámetros.

    Parámetros:
    X : array-like de forma (n_muestras, n_características)
        Los datos de entrada.
    y : array-like de forma (n_muestras,)
        Las etiquetas objetivo.
    test_size : float, opcional (por defecto=0.2)
        La proporción del conjunto de datos a incluir en la división de prueba.
    random_state : int, opcional (por defecto=42)
        Controla la aleatoriedad de la división de los datos.

    Retorna:
    best_model : objeto GridSearchCV
        El mejor modelo SVM encontrado después del ajuste de hiperparámetros.
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Crear un pipeline con escalado y SVM
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=random_state))
    ])

    # Definir la cuadrícula de parámetros para la búsqueda
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.1, 1],
        'svm__kernel': ['rbf', 'poly', 'sigmoid']
    }

    # Realizar la búsqueda de cuadrícula con validación cruzada
    grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_

    # Evaluar el modelo en el conjunto de prueba
    y_pred = best_model.predict(X_test)

    # Imprimir resultados
    print("Mejores parámetros encontrados:")
    print(grid_search.best_params_)
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    return best_model