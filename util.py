import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def comparar_columnas(df1, df2):
    # Obtener los nombres de las columnas en cada DataFrame
    columnas_df1 = set(df1.columns)
    columnas_df2 = set(df2.columns)
    
    # Encontrar columnas que están en df1 pero no en df2
    solo_en_df1 = columnas_df1 - columnas_df2
    
    # Encontrar columnas que están en df2 pero no en df1
    solo_en_df2 = columnas_df2 - columnas_df1
    
    # Imprimir los resultados
    if solo_en_df1:
        print("Columnas en df1 pero no en df2:")
        for columna in solo_en_df1:
            print(columna)
    else:
        print("Todas las columnas de df1 están en df2.")
    
    if solo_en_df2:
        print("Columnas en df2 pero no en df1:")
        for columna in solo_en_df2:
            print(columna)
    else:
        print("Todas las columnas de df2 están en df1.")
        
        
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
    #df_reducido, num_eliminadas = eliminar_caracteristicas(final, 0.99)
    return pd.DataFrame(resultados)
