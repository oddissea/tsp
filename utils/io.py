# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       io.py
# Descripción:   Módulo de entrada y salida para la gestión de parámetros,
#                resultados de experimentos y su almacenamiento en formatos CSV y JSON.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------

# utils/io.py

import json
import pandas as pd

def load_parameters(config_tsp_path="config.json"):
    """
    Carga los parámetros del algoritmo desde un archivo JSON.

    Parameters:
    :param config_tsp_path: (str) Ruta al archivo de configuración.

    Returns:
    :returns dict: Diccionario con los parámetros cargados.
    """
    with open(config_tsp_path, 'r') as file:
        parameters = json.load(file)
    return parameters

def save_results(results,
                 csv_path="output_100/experiment_results.csv",
                 json_path="output_100/experiment_details.json"):
    results_for_csv = [
        {
            "population_size": res["population_size"],
            "elite_size": res["elite_size"],
            "crossover_rate": res["crossover_rate"],
            "mutation_rate": res["mutation_rate"],
            "replace_strategy": res["replace_strategy"],
            "seed": res["seed"],
            "best_distance": res["best_distance"],
            "execution_time": res["execution_time"],
            "generations_to_convergence": res["generations_to_convergence"],
            "generations_to_best": res["generations_to_best"],
            "generations_to_stable": res["generations_to_stable"],
            "improvement_rate": res["improvement_rate"]
        }
        for res in results
    ]

    # Crear un DataFrame y guardar en CSV
    df_results = pd.DataFrame(results_for_csv)
    df_results.to_csv(csv_path, index=False)
    print("Resultados principales guardados en 'experiment_results.csv'.")

    # 2. Exportar detalles adicionales a JSON
    # Guardamos toda la estructura original (incluyendo "evolution") en un archivo JSON
    with open(json_path, "w") as file:
        json.dump(results, file, indent=4)
    print("Detalles adicionales guardados en 'experiment_details.json'.")

