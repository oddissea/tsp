# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       analyze.py
# Descripción:   Módulo de análisis y visualización para los resultados de experimentos
#                con algoritmos genéticos aplicados al Problema del Viajante (TSP).
#                Incluye funciones para graficar, calcular estadísticas, generar
#                informes y analizar convergencia.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional


# === FUNCIONES DE VISUALIZACIÓN === #

def plot_evolution(evolution: List[float], title="Evolución de las Distancias",
                   output_path=None):
    """
    Grafica la evolución de las distancias por generación.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(evolution)
    plt.title(title)
    plt.xlabel('Generación')
    plt.ylabel('Distancia')
    plt.grid()
    if output_path:
        plt.savefig(output_path)
    plt.show()


def intersection_points(avg, thresh, X):
    """
    Detecta puntos de intersección entre una curva promedio y un umbral dado usando interpolación.
    """
    points = []
    for i in range(1, len(avg)):
        if avg[i - 1] <= thresh < avg[i] or avg[i - 1] > thresh >= avg[i]:
            # Interpolación en el cruce
            x1, x2 = X.iloc[i - 1], X.iloc[i]
            y1, y2 = avg.iloc[i - 1], avg.iloc[i]
            slope = (y2 - y1) / (x2 - x1)
            x_intersection = x1 + (thresh - y1) / slope
            points.append(x_intersection)

    return points


def calculate_statistics(df, variable_param, impact):
    """
    Calcula estadísticas agrupadas por un parámetro variable.
    """
    stats = df.groupby(variable_param).agg(
        avg_value=(impact, "mean"),
        std_value=(impact, "std"),
        min_value=(impact, "min"),
        max_value=(impact, "max")
    ).reset_index()
    return stats


def calc_w(points, avg, thresh, X):
    """
    Calcula la medida W sumando las anchuras de los intervalos válidos.
    """
    valid_intervals = []

    all_points = [min(X)] + points + [max(X)]

    for i in range(len(all_points) - 1):
        start = all_points[i]
        end = all_points[i + 1]

        middle = (start + end) / 2
        middle_fitness = np.interp(middle, X, avg)

        # Validar el intervalo usando middle_fitness
        if middle_fitness < thresh:
            valid_intervals.append((start, end))

    # Calcular W como la suma de las anchuras de los intervalos válidos
    return sum(end - start for start, end in valid_intervals)


def plot_error_bar(df, fixed_params, variable_param, impact="best_distance",
                   y_label="Distancia Promedio", title_prefix="Impacto",
                   theta=0.5, output_path=None):
    """
    Representa barras de error para un parámetro variable mientras mantiene otros fijos.
    """
    # Filtrar resultados por parámetros fijos
    filtered_results = df.copy()
    for param, value in fixed_params.items():
        filtered_results = filtered_results[filtered_results[param] == value]

    # Calcular estadísticas
    stats = calculate_statistics(filtered_results, variable_param, impact)

    # Extraer valores para representar
    x_values = stats[variable_param]
    avg_values = stats["avg_value"]
    std_values = stats["std_value"]

    # Calcular H basado en las medias y el threshold basado en theta
    H = avg_values.max() - avg_values.min()
    threshold = avg_values.min() + theta * H

    # Detectar los puntos de cruce entre avg_values y threshold usando interpolación
    inter_points = intersection_points(avg_values, threshold, x_values)

    # Calcular W
    W = calc_w(inter_points, avg_values, threshold, x_values)

    # Crear texto adicional para los parámetros fijos, excluyendo elite_size
    other_params_text = ", ".join(f"{k}={v}" for k, v in fixed_params.items() if k != "elite_size")

    # Graficar con barras de error
    plt.errorbar(x_values, avg_values, yerr=std_values, fmt='o-', capsize=5, label="")
    plt.axhline(y=threshold, color='green', linestyle='--',
                label=f"Threshold (θ={theta}, H={H:.2f}, W={W:.2f})")

    # Graficar puntos de intersección
    for x_intersection in inter_points:
        plt.scatter(x_intersection, threshold, color='red', zorder=5, label=f"Cruce en x={x_intersection:.2f}")

    # Cambiar el tamaño de los ticks
    plt.tick_params(axis='both', which='major', labelsize=10)
    # Ajustar los ticks del eje X
    plt.xticks(ticks=x_values, labels=[f"{val:.2f}" for val in x_values])

    # Título compacto
    plt.title(f"{title_prefix} ({variable_param})", fontsize=12)

    # Añadir información detallada en la leyenda
    plt.xlabel(variable_param, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(title=f"{other_params_text}", fontsize=10, title_fontsize=10)
    plt.grid()
    if output_path:
        plt.savefig(output_path)
    plt.show()



def plot_param_impact(data, impact="best_distance", impact_label="Distancia Promedio", param="crossover_rate",
                      param_label="Crossover Rate", output_path=None):
    """
    Representa el impacto de un parámetro en un valor agregado como promedio o tiempo.
    """
    impact_data = data.groupby(param)[impact].mean()
    plt.figure(figsize=(12, 8))
    impact_data.plot(kind="bar")
    plt.title(f"Impacto de {param_label} en {impact_label}", fontsize=12)
    # Cambiar el tamaño de los ticks
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel(param_label, fontsize=12)
    plt.ylabel(impact_label, fontsize=12)
    plt.grid()
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_average_evolution_per_param(data: List[dict], param="crossover_rate", title_prefix="Evolución Promedio", output_dir=None):
    """
    Calcula y representa la evolución promedio de las distancias por cada valor único de un parámetro,
    incluyendo VAMM ± DE en la leyenda.
    """
    # Extraer valores únicos del parámetro
    unique_values = set(res[param] for res in data)

    for value in unique_values:
        # Filtrar evoluciones para el valor específico del parámetro
        evolutions = [res["evolution"] for res in data if res[param] == value]

        if evolutions:
            # Encontrar la longitud máxima de las evoluciones
            max_length = max(len(evo) for evo in evolutions)

            # Calcular la media y desviación estándar dinámicamente
            means = []
            stds = []

            for gen in range(max_length):
                # Extraer valores de la generación actual para todas las evoluciones que la alcanzaron
                generation_values = [evo[gen] for evo in evolutions if len(evo) > gen]
                means.append(np.mean(generation_values))
                stds.append(np.std(generation_values))

            # Calcular VAMM y desviación estándar
            final_distances = [evo[-1] for evo in evolutions]
            vamm = np.mean(final_distances)
            de = np.std(final_distances)

            # Crear leyenda con VAMM ± DE
            legend_text = f"{param}={value}\nVAMM: {vamm:.2f} ± {de:.2f}"

            # Crear título dinámico
            title = f"{title_prefix} ({param}={value})"

            # Graficar evolución promedio con desviación estándar
            plt.figure(figsize=(12, 8))
            plt.plot(means, label=legend_text)
            plt.fill_between(range(len(means)),
                             np.array(means) - np.array(stds),
                             np.array(means) + np.array(stds),
                             alpha=0.2, label="Desviación estándar")

            # Cambiar el tamaño de los ticks
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.title(title, fontsize=12)
            plt.xlabel("Generaciones", fontsize=12)
            plt.ylabel("Distancia", fontsize=12)
            plt.legend(loc="upper right", fontsize=10)
            plt.grid()

            # Guardar o mostrar la gráfica
            if output_dir:
                output_path = f"{output_dir}{param}_{value}.png"
                plt.savefig(output_path)
                print(f"Gráfica guardada en: {output_path}")

            plt.show()

def plot_top_evolutions(data: Optional[List[dict]], top_n=3,
                        title_prefix="Mejores Evoluciones Basadas en VAMM",
                        output_path=None):
    """
    Representa las mejores evoluciones basadas en el VAMM, mostrando VAMM ± DE y los parámetros relevantes en la leyenda.
    """
    if data:
        # Calcular VAMM y tiempos de ejecución para cada configuración
        vamm_data = []
        for res in data:
            # Filtrar solo las claves relevantes
            params = {k: res[k] for k in res if k in ["population_size", "crossover_rate",
                                                      "seed"]}
            final_distances = res["evolution"][-1:]  # Últimos valores como proxy del final
            vamm = np.mean(final_distances)  # Promedio de las distancias finales
            de = np.std(final_distances)  # Desviación estándar de las distancias finales

            vamm_data.append({
                "params": params,
                "vamm": vamm,
                "de": de,
                "execution_time": res.get("execution_time", None),  # Tiempo si está disponible
                "evolution": res["evolution"]
            })

        # Ordenar por VAMM (menor es mejor)
        sorted_vamm_data = sorted(vamm_data, key=lambda x: x["vamm"])

        # Seleccionar las top_n configuraciones
        top_vamm_data = sorted_vamm_data[:top_n]

        # Graficar las evoluciones seleccionadas
        plt.figure(figsize=(12, 8))
        for i, entry in enumerate(top_vamm_data):
            params = entry["params"]
            evolution = entry["evolution"]
            vamm = entry["vamm"]
            execution_time = entry["execution_time"]

            # Crear texto de la leyenda
            params_text = ", ".join(f"{k}={v}" for k, v in params.items())
            legend_text = (
                f"{params_text}\n"
                f"VAMM={vamm:.2f}, Time={execution_time:.2f}s"
            )

            # Graficar la evolución
            plt.plot(evolution, label=legend_text)

        # Configuración de la gráfica
        plt.title(title_prefix, fontsize=12)
        plt.xlabel("Generaciones", fontsize=12)
        plt.ylabel("Distancia", fontsize=12)
        # Cambiar el tamaño de los ticks
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid()
        if output_path:
            plt.savefig(f"{output_path}top_evolutions.png")
        plt.show()


def plot_box(data, column="best_distance", by="crossover_rate",
             column_label="Métrica", by_label="Parámetro", output_path=None):
    """
    Representa un boxplot de una métrica en función de un parámetro.

    :param data: DataFrame con los datos.
    :param column: Nombre de la columna de la métrica a analizar (por defecto "best_distance").
    :param by: Nombre del parámetro por el cual se agruparán los datos (por defecto "crossover_rate").
    :param column_label: Etiqueta para el eje Y.
    :param by_label: Etiqueta para el eje X.
    :param output_path: Ruta para guardar la gráfica (opcional).
    """
    plt.figure(figsize=(12, 8))
    data.boxplot(column=column, by=by, grid=False)

    # Títulos y etiquetas
    plt.title(f"Distribución de {column_label} por {by_label}", fontsize=12)
    plt.suptitle("")  # Eliminar el título automático generado por pandas
    plt.xlabel(by_label, fontsize=12)
    plt.ylabel(column_label, fontsize=12)

    # Cambiar el tamaño de los ticks
    plt.tick_params(axis='both', which='major', labelsize=8)

    if output_path:
        plt.savefig(output_path)
        print(f"Gráfica guardada en: {output_path}")

    plt.show()




def plot_convergence_by_param(df_results, param="crossover_rate", y="generations_to_stable", output_path=None):
    """
    Grafica el tiempo promedio de estabilización en función de un parámetro.

    :param df_results: DataFrame con los datos.
    :param param: Nombre del parámetro para el eje X (por defecto "crossover_rate").
    :param y: Nombre de la métrica a analizar (por defecto "generations_to_stable").
    :param output_path: Ruta para guardar la gráfica (opcional).
    """
    convergence_data = df_results.groupby(param)[y].mean()
    plt.figure(figsize=(12, 8))
    convergence_data.plot(kind="bar")

    # Títulos y etiquetas con tamaño ajustado
    plt.title(f"Tiempo de Estabilización Promedio por {param}", fontsize=12)
    plt.xlabel(param, fontsize=12)
    plt.ylabel("Generaciones a Estabilización", fontsize=12)

    # Ajustar ticks del eje X e Y
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Añadir leyenda con tamaño ajustado
    plt.legend([f"Estabilización ({param})"], fontsize=10, loc="upper right", frameon=False)

    # Mostrar o guardar la gráfica
    if output_path:
        plt.savefig(output_path)
        print(f"Gráfica guardada en: {output_path}")
    plt.show()



# === FUNCIONES DE CARGA Y ANÁLISIS === #

def load_results(results_csv, details_json):
    """
    Carga los resultados del CSV y JSON.
    """
    df_results = pd.read_csv(results_csv)
    experiment_details = None
    try:
        with open(details_json, "r") as file:
            experiment_details = json.load(file)
    except FileNotFoundError:
        pass
    return df_results, experiment_details


def calculate_stats(df, group_by, metrics):
    """
    Calcula estadísticas agregadas para un DataFrame.
    """
    return df.groupby(group_by).agg(metrics).reset_index()


def print_stats(stats, name):
    """
    Imprime estadísticas con un encabezado.
    """
    print(f"\n=== Estadísticas de {name} ===")
    print(stats)


def extract_unique_combinations(df, params):
    """
    Extrae combinaciones únicas de valores para los parámetros especificados.
    """
    return df[params].drop_duplicates().to_dict('records')

# === Cálculo de métricas === #

def calculate_vamm(df):
    """
    Calcula el Valor de Adaptación Medio del Mejor Individuo (VAMM).
    """
    vamm = df.groupby("crossover_rate")["best_distance"].mean()
    std_dev = df.groupby("crossover_rate")["best_distance"].std()
    return pd.DataFrame({"VAMM": vamm, "std_dev": std_dev}).reset_index()

def calculate_relative_distance(df, optimal_value):
    """
    Calcula la distancia relativa al óptimo.
    """
    df["relative_distance"] = ((df["best_distance"] - optimal_value) / optimal_value) * 100
    return df.groupby("crossover_rate")["relative_distance"].mean().reset_index()

def calculate_convergence(df):
    """
    Calcula las generaciones necesarias hasta convergencia.
    """
    convergence_generations = df.groupby("crossover_rate")["generations_to_convergence"].mean()
    return pd.DataFrame({"convergence_generations": convergence_generations}).reset_index()

def generate_summary_table(df, optimal_value):
    """
    Genera una tabla resumen con VAMM, distancia relativa y generaciones hasta convergencia.
    """
    vamm_stats = calculate_vamm(df)
    relative_distances = calculate_relative_distance(df, optimal_value)
    convergence_data = calculate_convergence(df)

    # Combinar los datos en una única tabla
    summary = vamm_stats.merge(relative_distances, on="crossover_rate")
    summary = summary.merge(convergence_data, on="crossover_rate")

    # Ajustar columnas para formato deseado
    summary.rename(columns={
        "crossover_rate": "Crossover Rate",
        "VAMM": "VAMM",
        "std_dev": "Std Dev",
        "relative_distance": "Relative Distance (%)",
        "convergence_generations": "Generations to Convergence"
    }, inplace=True)

    return summary


def analyze_convergence_results(df_results,
                                group_by=None,
                                output_path="output_10150"):
    """
    Analiza métricas de convergencia por grupos de parámetros y guarda los resultados.

    :param df_results: DataFrame con los resultados del experimento.
    :param group_by: Lista de columnas por las que agrupar para el análisis.
    :param output_path: Ruta de salida para guardar los resultados.
    :return: DataFrame con las métricas de convergencia.
    """
    if group_by is None:
        group_by = ["crossover_rate", "mutation_rate"]
    os.makedirs(output_path, exist_ok=True)

    # Calcular métricas de convergencia
    convergence_metrics = df_results.groupby(group_by).agg(
        avg_generations_to_best=("generations_to_best", "mean"),
        avg_generations_to_stable=("generations_to_stable", "mean"),
        avg_improvement_rate=("improvement_rate", "mean")
    ).reset_index()

    # Imprimir las métricas de convergencia
    print("\n=== Métricas de Convergencia ===")
    print(convergence_metrics)

    # Guardar las métricas en CSV
    csv_path = os.path.join(output_path, "convergence_metrics.csv")
    convergence_metrics.to_csv(csv_path, index=False)
    print(f"Métricas de convergencia guardadas en: {csv_path}")

    # Exportar las métricas a LaTeX
    latex_path = os.path.join(output_path, "convergence_metrics.tex")
    with open(latex_path, "w") as f:
        f.write(convergence_metrics.to_latex(index=False, float_format="%.2f"))
    print(f"Métricas de convergencia exportadas a LaTeX: {latex_path}")

    return convergence_metrics


def data_report(results_csv="experiment_results.csv",
                details_json="experiment_details.json",
                optimal_value=40,
                output_path="output_10150"):
    """
    Genera un informe detallado con estadísticas adicionales como VAMM, distancia relativa y generaciones
    hasta convergencia.
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_path, exist_ok=True)

    # Cargar datos
    df_results, experiment_details = load_results(results_csv, details_json)

    # Calcular estadísticas adicionales
    summary_table = generate_summary_table(df_results, optimal_value)

    # Guardar la tabla en CSV
    summary_csv_path = os.path.join(output_path, "summary_results.csv")
    summary_table.to_csv(summary_csv_path, index=False)
    print(f"Tabla resumen guardada en: {summary_csv_path}")

    # Exportar tabla a LaTeX
    summary_latex_path = os.path.join(output_path, "summary_results.tex")
    with open(summary_latex_path, "w") as f:
        f.write(summary_table.to_latex(index=False, float_format="%.2f"))
    print(f"Tabla resumen exportada a LaTeX: {summary_latex_path}")

    return summary_table



# === FUNCIÓN GRÁFICAS === #

def impact_graphs(data, output_path):
    # Gráficas de impacto
    params_to_plot = {
        "crossover_rate": "Crossover Rate",
        #"mutation_rate": "Mutation Rate",
        #"replace_strategy": "Replace Strategy",
        #"population_size": "Population Size"
    }
    for param, param_label in params_to_plot.items():
        plot_param_impact(data, impact="best_distance", impact_label="Distancia Promedio",
                          param=param, param_label=param_label, output_path=f"{output_path}impact_{param}_distance.png")
        plot_param_impact(data, impact="execution_time", impact_label="Tiempo Promedio (s)",
                          param=param, param_label=param_label, output_path=f"{output_path}impact_{param}_time.png")


def err_graphs(data, output_path):
    variable_param = "crossover_rate"  # Solo el parámetro crossover_rate
    other_param = "mutation_rate"  # Complemento

    # Extraer combinaciones únicas de los demás parámetros
    error_bar_params = extract_unique_combinations(data, ["population_size", "elite_size", other_param])

    # Iterar y graficar solo para crossover_rate
    for i, fixed_params in enumerate(error_bar_params):
        plot_error_bar(
            df=data,
            fixed_params=fixed_params,
            variable_param=variable_param,
            title_prefix=f"Impacto de {variable_param.capitalize().replace('_', ' ')}",
            theta=0.3,
            output_path=f"{output_path}robustez_crossover_{i}.png"
        )

# === FUNCIÓN PRINCIPAL === #

def analyze_results(results_csv="experiment_results.csv",
                    details_json="experiment_details.json",
                    top_evs=5,
                    output_path=None,
                    optimal_value=40, label="xqf131"):

    # Configuración global
    sns.set_theme(style="whitegrid", context="talk", palette="deep")

    # Cargar datos
    df_results, experiment_details = load_results(results_csv, details_json)

    # Calcular estadísticas de distancias y tiempos
    distance_metrics = {
        "best_distance": ["mean", "std", "min", "max"]
    }
    time_metrics = {
        "execution_time": ["mean", "std", "min", "max"]
    }
    group_by = ["population_size", "elite_size", "crossover_rate", "mutation_rate", "replace_strategy"]

    dis_stats = calculate_stats(df_results, group_by, distance_metrics)
    time_stats = calculate_stats(df_results, group_by, time_metrics)

    # Calcular la medida H y agregarla a dis_stats
    dis_stats[("best_distance", "H")] = dis_stats[("best_distance", "max")] - dis_stats[("best_distance", "min")]

    # Mostrar estadísticas
    print_stats(dis_stats, "Distancia Promedio")
    print_stats(time_stats, "Tiempo de Ejecución")

    # Gráficas de impacto
    impact_graphs(df_results, f"{output_path}/graphs/")

    # Gráficas de error
    err_graphs(df_results, f"{output_path}/graphs/")

    # Representa las gráficas con el promedio de las evoluciones
    # average_evo_plots(df_results, experiment_details, ev_avg_best, output_path)
    plot_average_evolution_per_param(
        data=experiment_details,
        param="crossover_rate",
        title_prefix=f"Evolución Promedio para la instancia {label}",
        output_dir=f"{output_path}/graphs/"  # Directorio donde guardar las gráficas
    )

    # Mejores distancias individuales
    plot_top_evolutions(experiment_details, top_n=top_evs, title_prefix="Mejores Evoluciones Basadas en VAMM")

    # Generar informe adicional con estadísticas detalladas
    if output_path:
        data_report(
            results_csv=results_csv,
            details_json=details_json,
            optimal_value=optimal_value,
            output_path=f"{output_path}/report"
        )

    analyze_convergence_results(
        df_results=df_results,
        group_by=["crossover_rate", "mutation_rate"],
        output_path=f"{output_path}/report"
    )

    plot_box(
        data=df_results,
        column="best_distance",
        by="crossover_rate",
        column_label="Distancia",
        by_label=f"$p_c$ ({label})",
        output_path=f"{output_path}/graphs/boxplot_best_distance_by_crossover_rate.png"
    )

    plot_box(
        data=df_results,
        column="execution_time",
        by="crossover_rate",
        column_label="Tiempo de Ejecución (s)",
        by_label=f"$p_c$ ({label})",
        output_path=f"{output_path}/graphs/boxplot_execution_time_by_crossover_rate.png"
    )

    plot_convergence_by_param(df_results, param="crossover_rate", y="generations_to_stable",
                              output_path=f"{output_path}/graphs/convergence_by_crossover_rate.png")

