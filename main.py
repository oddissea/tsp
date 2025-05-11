# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       main.py
# Descripción:   Programa principal para la ejecución de un algoritmo genético
#                aplicado al problema del vendedor viajero (TSP). Realiza la
#                configuración de parámetros, ejecución del algoritmo y análisis
#                de resultados, incluyendo métricas de convergencia y visualización.
# Versión:       0.1
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------

# main.py

from src.genetic_algorithm import TSPGeneticAlgorithm
from utils.tsp_instance import TSPInstance
from utils.random_seed import set_seed
from utils.io import load_parameters, save_results
from utils.analyze import analyze_results

from itertools import product
import time
import numpy as np

def main(config, num_repeats):
    params = load_parameters(config)

    tsp_instance = TSPInstance(params['instance_tsp_path'])

    distances = tsp_instance.distance_matrix

    # Extraer los valores de los parámetros
    generations = params["generations"]
    population_sizes = params["population_size"]
    tournament_size = params["tournament_size"]
    crossover_rates = params["crossover_rates"]
    mutation_rates = params["mutation_rate"]
    elite_sizes = params["elite_size"]
    replace_strategies = params["replace_strategy"]
    patience = None if params["patience"] == "None" else params["patience"]
    # num_repeats = 5  # Número de repeticiones para cada combinación de parámetros


    # Resultado para todas las combinaciones
    experiment_results = []

    # Iterar por cada combinación de population_size y elite_size
    for population_size, elite_size in zip(population_sizes, elite_sizes):
        print(f"\nEvaluando population_size: {population_size}, elite_size: {elite_size}")

        # Generar todas las combinaciones de los otros parámetros
        param_combinations = product(crossover_rates, mutation_rates, replace_strategies)

        # Iterar por las combinaciones de parámetros secundarios
        for crossover_rate, mutation_rate, replace_strategy in param_combinations:
            print(
                f" - Probando crossover_rate: {crossover_rate}, mutation_rate: {mutation_rate}, replace_strategy: {replace_strategy}")

            # Repetir cada combinación con diferentes semillas
            for repeat in range(num_repeats):
                current_seed = repeat  # Usar el índice de repetición como semilla
                print(f"   > Repetición {repeat + 1}/{num_repeats}, semilla: {current_seed}")

                # Configurar la semilla para reproducibilidad
                set_seed(current_seed)

                # Crear y resolver el problema
                tsp = TSPGeneticAlgorithm(
                    distances=distances,
                    population_size=population_size,
                    generations=generations,
                    tournament_size=tournament_size,
                    mutation_rate=mutation_rate,
                    crossover_rate=crossover_rate,
                    elite_size=elite_size,
                    replace_strategy=replace_strategy,
                    patience=patience
                )
                # Medir tiempo de ejecución
                start_time = time.time()

                # Ejecutar el algoritmo
                result = tsp.solve()

                # Calcular tiempo de ejecución
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Analizar convergencia
                convergence_metrics = analyze_convergence(result["evolution"])

                # Guardar los resultados
                experiment_results.append({
                    "population_size": population_size,
                    "elite_size": elite_size,
                    "crossover_rate": crossover_rate,
                    "mutation_rate": mutation_rate,
                    "replace_strategy": replace_strategy,
                    "seed": current_seed,
                    "best_distance": result["distance"],
                    "best_route": result["route"],
                    "evolution": result["evolution"],
                    "generations_to_convergence": result["generations_completed"],
                    "execution_time": elapsed_time,
                    "generations_to_best": convergence_metrics["generations_to_best"],
                    "generations_to_stable": convergence_metrics["generations_to_stable"],
                    "improvement_rate": convergence_metrics["improvement_rate"]
                })

                #plot_evolution(result['evolution'])

    # Análisis final
    return experiment_results


def analyze_convergence(evolution_data, threshold=0.01):
    """
    Analiza la velocidad de convergencia.

    :param evolution_data: Lista de valores de fitness por generación
    :param threshold: Umbral de mejora relativa para considerar convergencia
    :return: Dict con métricas de convergencia
    """
    best_gen = int(np.argmin(evolution_data))  # Conversión a int
    initial_fitness = evolution_data[0]
    best_fitness = evolution_data[best_gen]

    # Encontrar generación de estabilización
    stable_gen = None
    for i in range(1, len(evolution_data)):
        rel_improvement = abs((evolution_data[i - 1] - evolution_data[i]) / max(evolution_data[i - 1], 1e-9))
        if rel_improvement < threshold:
            stable_gen = int(i)  # Conversión a int
            break

    return {
        "generations_to_best": best_gen + 1,  # Generación es index + 1
        "generations_to_stable": stable_gen + 1 if stable_gen is not None else None,
        "improvement_rate": float((initial_fitness - best_fitness) / (best_gen + 1))  # Conversión a float
    }


if __name__ == "__main__":
    output = "output_1083"
    csv_path = f"{output}/experiment_results.csv"
    json_path = f"{output}/experiment_details.json"
    config_file = f"config/config_xit1083.json"
    label = "xqf131"
    repeats = 10

    res = main(config=config_file, num_repeats=repeats)

    save_results(results=res, csv_path=csv_path, json_path=json_path)

    g_params = load_parameters(config_file)
    optimal_distance = g_params["optimal_fitness"]

    analyze_results(csv_path,
                    json_path,
                    top_evs=5,
                    output_path=f"{output}",
                    optimal_value=optimal_distance,
                    label=label)