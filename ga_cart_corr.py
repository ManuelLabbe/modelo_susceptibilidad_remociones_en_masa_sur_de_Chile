import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def eliminar_caracteristicas(df, umbral):
    """
    Elimina características de un DataFrame basado en la correlación.
    
    Args:
    - df (pd.DataFrame): DataFrame con las características.
    - umbral (float): Umbral de correlación para eliminar características.
    
    Returns:
    - pd.DataFrame: DataFrame con las características restantes.
    - int: Número de características eliminadas.
    """
    # Calcular la matriz de correlación
    correlaciones = df.corr().abs()
    
    # Crear una máscara para evitar considerar la diagonal principal
    mask = np.triu(np.ones_like(correlaciones, dtype=bool), k=1)
    
    # Identificar pares de características con alta correlación
    to_drop = set()
    for i in range(len(correlaciones.columns)):
        for j in range(i):
            if correlaciones.iloc[i, j] > umbral:
                colname = correlaciones.columns[i]
                to_drop.add(colname)
    
    # Eliminar las características
    df_reducido = df.drop(columns=to_drop)
    
    # Número de características eliminadas
    num_eliminadas = len(to_drop)
    
    return df_reducido, num_eliminadas

def prueba_umbral(df, umbrales):
    """
    Prueba diferentes umbrales y guarda el número de características eliminadas.
    
    Args:
    - df (pd.DataFrame): DataFrame con las características.
    - umbrales (list): Lista de umbrales para probar.
    
    Returns:
    - pd.DataFrame: DataFrame con los resultados de la prueba.
    """
    resultados = []
    
    for umbral in umbrales:
        df_reducido, num_eliminadas = eliminar_caracteristicas(df, umbral)
        resultados.append({'Umbral': umbral, 'Caracteristicas eliminadas': num_eliminadas})
    
    return pd.DataFrame(resultados)

def cart_feature_selection(df, target_column, n_features=5):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cart = RandomForestClassifier(random_state=42)
    #cart = XGBClassifier(random_state=42)
    cart.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': cart.feature_importances_
    }).sort_values('importance', ascending=False)
    
    selected_features = feature_importance['feature'][:n_features].tolist()
    
    return selected_features

