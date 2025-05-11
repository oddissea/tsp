# ------------------------------------------------------------------------------
# Proyecto TSP - Algoritmo Genético para el Problema del Viajante
# Autor: Fernando H. Nasser-Eddine López
# Fecha: 14/12/2024
# Versión: 1.0
# ------------------------------------------------------------------------------

## Descripción del Proyecto
Este proyecto implementa un algoritmo genético para resolver el Problema del Viajante (TSP, por sus siglas en inglés).
El TSP consiste en encontrar la ruta más corta que visita una lista de ciudades exactamente una vez y regresa a la ciudad inicial.

El programa permite configurar parámetros como tamaño de población, probabilidad de cruce, probabilidad de mutación,
tamaño del torneo de selección y criterios de convergencia. Además, incluye funcionalidades para analizar los resultados
y generar gráficos (aunque en algunas solo respecto a la probabilidad de cruce).

## Estructura del Proyecto

proyecto_tsp /
│
├── src /                           # Módulos principales
│   ├── genetic_algorithm.py        # Implementación del algoritmo genético
│   ├── selection.py                # Operador de selección (torneo)
│   ├── crossover.py                # Operador de cruce (PMX)
│   ├── mutation.py                 # Operador de mutación (swap mutation)
│   └── fitness.py                  # Cálculo de fitness (distancia total de la ruta)
│
├── utils /                         # Módulos auxiliares
│   ├── io.py                       # Carga y almacenamiento de datos
│   ├── tsp_instance.py             # Gestión de instancias TSP y matriz de distancias
│   └── random_seed.py              # Configuración de semillas para reproducibilidad
│   └── analyze.py                  # Análisis de los datos
│
├── config /                        # Archivos de configuración
│   └── config_xit1083.json         # Archivos JSON con los parámetros del experimento instancia de 1083
│   └── config_xqf131.json          # Archivos JSON con los parámetros del experimento instancia de 131
│
├── data /                          # Datos de entrada
│   └── xit1083.tsp                 # Archivo TSP con 1083 ciudades
│   └── xqf131.tsp                  # Archivo TSP con 131 ciudades
│
├── output_131 /                    # Resultados y análisis instancia 131
│   ├── experiment_results.csv      # Resultados principales en formato CSV
│   ├── experiment_details.json     # Detalles completos del experimento en JSON
│   ├── graphs /                    # Gráficas generadas
│   └── report /                    # Informes y tablas resumen
|
├── output_1083 /                   # Resultados y análisis instancia 1083
│   ├── experiment_results.csv      # Resultados principales en formato CSV
│   ├── experiment_details.json     # Detalles completos del experimento en JSON
│   ├── graphs /                    # Gráficas generadas
│   └── report /                    # Informes y tablas resumen
│
└── main.py                         # Script principal para ejecutar el proyecto
└── README.txt                      # Script principal para ejecutar el proyecto
