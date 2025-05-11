# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       fitness.py
# Descripción:   Cálculo del fitness para rutas en el Problema del Viajante (TSP).
#                Evalúa la distancia total de una ruta basada en la matriz de
#                distancias proporcionada.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------

from typing import List


class RouteFitness:
    def __init__(self, distances: List[List[float]]):
        """
        Inicializa la clase con la matriz de distancias.

        :param distances: Matriz de distancias entre las ciudades.
        """
        self.distances = distances

    def calculate_route_distance(self, route: List[int]) -> float:
        """
        Calcula la distancia total de una ruta.

        :param route: Lista que representa el orden de las ciudades en la ruta.
        :return: Distancia total recorrida en la ruta.
        """
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += self.distances[from_city][to_city]
        return total_distance
