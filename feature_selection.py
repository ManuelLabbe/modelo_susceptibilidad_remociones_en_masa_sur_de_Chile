import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from deap import creator, base, tools, algorithms
from functools import partial
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
    

def evaluar_cromosoma(individual, X, y, n_caracteristicas):
    # Decodificar el cromosoma
    selected_features = X.columns[np.array(individual, dtype=bool)]
    
    # Verificar restricciones
    if len(selected_features) == 0:
        return 1,  # Penalización máxima si no se selecciona ninguna característica
    
    # Entrenar el modelo
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X[selected_features], y, cv=5, scoring='accuracy')
    
    # Obtener medida de rendimiento
    rendimiento = 1 - np.mean(scores)
    
    return rendimiento,

def seleccion_caracteristicas_genetico(df, target_column, n_generations=50, population_size=50, n_processes=None):
    if n_processes is None:
        n_processes = multiprocessing.cpu_count() - 1  # Usar todos los núcleos menos uno
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    n_caracteristicas = len(X.columns)

    # Configuración del problema
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Configuración del toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_caracteristicas)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Configurar la evaluación paralela
    pool = multiprocessing.Pool(processes=n_processes)
    toolbox.register("map", pool.map)
    
    # Registrar las operaciones genéticas
    eval_partial = partial(evaluar_cromosoma, X=X, y=y, n_caracteristicas=n_caracteristicas)
    toolbox.register("evaluate", eval_partial)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Crear población inicial
    population = toolbox.population(n=population_size)
    
    # Configurar estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    logbook = tools.Logbook()
    hof = tools.HallOfFame(1)

    try:
        # Algoritmo evolutivo
        for gen in range(n_generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.9, mutpb=0.1 + (gen / n_generations) * 0.1)
            
            # Evaluación paralela de la población
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            population = toolbox.select(offspring, k=len(population))
            hof.update(population)
            
            # Registrar estadísticas
            record = stats.compile(population)
            logbook.record(gen=gen, **record)
            print(f"Generación {gen}: {record}")

    finally:
        pool.close()
        pool.join()

    # Obtener mejor resultado
    best_individual = hof[0]
    selected_features = X.columns[np.array(best_individual, dtype=bool)].tolist()

    return selected_features, logbook, hof