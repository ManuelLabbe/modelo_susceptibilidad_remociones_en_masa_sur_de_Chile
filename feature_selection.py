import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from deap import creator, base, tools, algorithms
import random
from functools import partial
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import multiprocessing

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz
import pandas as pd
import re

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz
import pandas as pd
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import pandas as pd
import graphviz
import re
import numpy as np
from scipy import stats

def cart_feature_selection(df, target_column, n_features=5, n_cv_folds=5):
    """
    Realiza selección de características usando CART con ajuste de hiperparámetros
    y cálculo de intervalos de confianza para el error.
    
    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna objetivo
        n_features: Número de características a seleccionar
        n_cv_folds: Número de folds para validación cruzada
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crear el modelo base
    base_tree = SVC(random_state=42)
    
    # Realizar búsqueda de grid con validación cruzada
    grid_search = GridSearchCV(
        estimator=base_tree,
        #param_grid=param_grid,
        cv=n_cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Ajustar el modelo
    grid_search.fit(X_train, y_train)
    
    # Obtener el mejor modelo
    best_tree = grid_search.best_estimator_
    
    # Obtener importancias de características
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_tree.feature_importances_
    }).sort_values('importance', ascending=False)
    
    selected_features = feature_importance['feature'][:n_features].tolist()
    
    # Exportar el árbol a formato DOT
    dot_data = export_graphviz(
        best_tree,
        max_depth=3,
        out_file=None,
        feature_names=X.columns,
        class_names=None,
        filled=False,
        impurity=True,
        proportion=False,
        rounded=True,
        precision=2,
        label='none'
    )
    
    # Función para reemplazar las etiquetas de los nodos
    def replace_labels(dot_data):
        pattern = re.compile(r'label="([^"]+)"')
        def repl(match):
            label_content = match.group(1)
            lines = label_content.split('\\n')
            new_label_lines = []
            for line in lines:
                if '<=' in line:
                    feature_name = line.split('<=')[0].strip()
                    if feature_name == 'valor_humedad_suelo1':
                        feature_name = 'VMoist'
                    feature_name = re.sub(r'0-5cm', 'a', feature_name)
                    feature_name = re.sub(r'5-15cm', 'b', feature_name)
                    feature_name = re.sub(r'15-30cm', 'c', feature_name)
                    feature_name = re.sub(r'30-60cm', 'd', feature_name)
                    feature_name = re.sub(r'60-100cm', 'e', feature_name)
                    feature_name = re.sub(r'100-200cm', 'f', feature_name)
                    new_label_lines.append(feature_name)
                elif 'gini' in line:
                    new_label_lines.append(line)
            new_label = '\\n'.join(new_label_lines)
            return f'label="{new_label}"'
        return pattern.sub(repl, dot_data)

    # Modificar las etiquetas en el DOT data
    dot_data = replace_labels(dot_data)

    # Visualizar el árbol
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree", format='png', cleanup=True)
    
    # Calcular predicciones y métricas
    y_pred = best_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error = 1 - accuracy
    
    # Calcular margen de error (95%)
    n = len(y_test)
    confidence = 0.95
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Método de Wilson para el margen de error
    denominador = 1 + (z**2 / n)
    centro = (error + z**2 / (2*n)) / denominador
    margen = z * np.sqrt((error * (1 - error) + z**2 / (4*n)) / n) / denominador
    
    # Imprimir resultados
    print("\nResultados del modelo:")
    print(f"Mejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"- {param}: {value}")
    print(f"\nPrecisión del modelo: {accuracy:.4f}")
    print(f"Error del modelo: {error:.4f} ± {margen:.4f}")
    
    return {
        'selected_features': selected_features,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'error': error,
        'error_margin': margen,
        'model': best_tree
    }
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluar_cromosoma(individual, X, y, n_caracteristicas):
    # Decodificar el cromosoma
    selected_features = X.columns[np.array(individual, dtype=bool)]

    # Verificar restricciones
    if len(selected_features) == 0:
        return 1,  # Penalización máxima

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features],
        y,
        test_size=0.2,
        random_state=42
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Configurar el modelo Random Forest
    clf = SVC(
        random_state=42
    )

    # Entrenar el modelo
    clf.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = clf.predict(X_test)

    # Calcular precisión
    accuracy = accuracy_score(y_test, y_pred)

    # Calcular rendimiento
    rendimiento = 1 - accuracy

    # Penalización por número de características
    penalizacion = len(selected_features) / 100

    return rendimiento + penalizacion,

def seleccion_caracteristicas_genetico(df, target_column, n_generations=50, population_size=50, n_processes=None, 
                                       outputpath = 'path', mutFlipBit=0.9, mut_prob=0.9, cruce_prob=0.9):
    if n_processes is None:
        n_processes = multiprocessing.cpu_count() - 1
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    n_caracteristicas = len(X.columns)
    
    outputpath = f"{outputpath}/{outputpath.split('/')[-1]}.csv"

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_caracteristicas)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pool = multiprocessing.Pool(processes=n_processes)
    toolbox.register("map", pool.map)
    
    eval_partial = partial(evaluar_cromosoma, X=X, y=y, n_caracteristicas=n_caracteristicas)
    toolbox.register("evaluate", eval_partial)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutFlipBit)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    logbook = tools.Logbook()
    hof = tools.HallOfFame(1)

    # DataFrame para guardar generaciones
    generations_df = pd.DataFrame(columns=X.columns)
    
    # Evaluar población inicial
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    hof.update(population)

    try:
        for gen in range(n_generations):
            # Seleccionar elite
            elite = tools.selBest(population, k=4)
            elite = list(map(toolbox.clone, elite))
            
            # Selección para el resto
            offspring = toolbox.select(population, len(population) - 4)
            offspring = list(map(toolbox.clone, offspring))
            
            # Cruzamiento con probabilidad 0.9
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cruce_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutación con probabilidad que aumenta
            mutpb = mut_prob + (gen / n_generations) * 0.1
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluar individuos sin fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Combinar elite con offspring
            population[:] = elite + offspring
            
            # Actualizar hall of fame
            hof.update(population)
            
            # Guardar población en DataFrame
            gen_matrix = np.array([ind for ind in population])
            gen_df = pd.DataFrame(gen_matrix, columns=X.columns)
            generations_df = pd.concat([generations_df, gen_df], ignore_index=True)
            
            # Guardar CSV
            generations_df.to_csv(outputpath, index=False)
            print(f'Guardando en {outputpath}')
            # Registrar y mostrar estadísticas
            record = stats.compile(population)
            logbook.record(gen=gen, **record)
            print(f"Generación {gen}: {record}")

    finally:
        pool.close()
        pool.join()

    return population, logbook, hof

def plot_convergence(logbook, title="Convergencia del Algoritmo Genético"):
    """
    Función para graficar la convergencia del algoritmo genético.
    
    Args:
        logbook: Registro de estadísticas del algoritmo genético
        title: Título del gráfico
    """
    gen = logbook.select("gen")
    avg = logbook.select("avg")
    min_vals = logbook.select("min")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar promedio y mínimo
    ax.plot(gen, avg, 'r-', label='Promedio')
    ax.plot(gen, min_vals, 'b-', label='Mínimo')
    
    # Configurar el gráfico
    ax.set_xlabel('Generación')
    ax.set_ylabel('Fitness (Error)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()
    
    return fig