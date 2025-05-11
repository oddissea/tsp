# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       mutation.py
# Descripción:   Implementación de la mutación por intercambio (Swap Mutation)
#                para algoritmos genéticos. Realiza un intercambio entre dos
#                posiciones de una ruta con una probabilidad dada.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------
# src/mutation.py

import random
from typing import List


class SwapMutation:
    def __init__(self, mutation_rate: float):
        """
        Inicializa la mutación por intercambio.

        :param mutation_rate: (float) Probabilidad de aplicar la mutación.
        """
        self.mutation_rate = mutation_rate

    def mutate(self, route: List[int]) -> List[int]:
        """
        Aplica la mutación por intercambio (swap mutation) con una probabilidad dada.

        :param route: (List[int]) Ruta actual representada como una lista de enteros.
        :returns: (List[int]) La ruta mutada después de aplicar el intercambio.
        """
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)  # Seleccionar dos índices diferentes al azar
            route[i], route[j] = route[j], route[i]  # Intercambiar posiciones
        return route
