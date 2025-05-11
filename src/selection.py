# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       selection.py
# Descripción:   Implementación del operador de selección por torneo para un
#                algoritmo genético. Se seleccionan individuos en función de
#                su valor de fitness.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------
# src/selection.py

import random
from typing import List, Callable


class TournamentSelection:
    """
    Implementación del operador de selección por torneo para un algoritmo genético.

    En la selección por torneo, se elige un subconjunto de individuos al azar de la población
    y se selecciona el individuo con mejor fitness (menor valor) como ganador del torneo.
    Este proceso se repite hasta obtener el número deseado de individuos seleccionados.

    Attributes:
        tournament_size (int): Tamaño del torneo, es decir, el número de individuos
                               que competirán en cada torneo.
        population_size (int): Tamaño total de la población.
        elite_size (int): Número de individuos élite que se preservan directamente
                          en la siguiente generación sin competir en el torneo.
    """

    def __init__(self, tournament_size: int, population_size: int, elite_size: int):
        """
        Inicializa los parámetros de selección por torneo.

        :param tournament_size: Tamaño del torneo (número de individuos en cada torneo).
        :param population_size: Tamaño total de la población.
        :param elite_size: Número de individuos élite que no participan en la selección.
        """
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.elite_size = elite_size

    def select(self, population: List[List[int]], fitness_func: Callable) -> List[List[int]]:
        """
        Realiza la selección por torneo sobre la población.

        :param population: Lista que contiene la población actual. Cada individuo es
                           representado como una lista de enteros.
        :param fitness_func: Función de fitness que recibe un individuo como entrada
                             y devuelve su valor de fitness (a minimizar).

        :return: List[List[int]] - Lista de individuos seleccionados que formarán parte
                                   de la siguiente generación. El tamaño de la lista
                                   será igual a 'population_size - elite_size'.
        """
        selected = []  # Lista para almacenar los individuos seleccionados
        for _ in range(self.population_size - self.elite_size):
            # Seleccionar aleatoriamente un subconjunto de individuos
            tournament = random.sample(population, self.tournament_size)
            # Determinar el ganador del torneo (el individuo con el mejor fitness)
            winner = min(tournament, key=fitness_func)
            selected.append(winner)  # Añadir el ganador a la lista de seleccionados
        return selected
