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

    # Configurar el modelo Random Forest
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
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