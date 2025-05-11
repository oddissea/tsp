# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       genetic_algorithm.py
# Descripción:   Implementación del algoritmo genético para resolver el
#                Problema del Viajante (TSP). Incluye métodos de selección,
#                cruce, mutación y estrategias de supervivencia con elitismo.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------
# src/genetic_algorithm.py

import random
from typing import List, Dict, Any
from tqdm.auto import tqdm

from src.selection import TournamentSelection
from src.crossover import PMXCrossover
from src.mutation import SwapMutation
from src.survival import GenerationalElitism
from src.fitness import RouteFitness


class TSPGeneticAlgorithm:
    def __init__(self,
                 distances: List[List[float]],
                 generations: int = 500,
                 population_size: int = 100,
                 tournament_size: int = 2,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.01,
                 elite_size: int = 10,
                 replace_strategy: str = "worst",
                 patience: int = 10):
        """
        Inicializa el algoritmo genético para TSP.

        :param distances: Matriz de distancias entre ciudades
        :param population_size: Tamaño de la población
        :param generations: Número de generaciones
        :param mutation_rate: Probabilidad de mutación
        :param elite_size: Número de soluciones élite a preservar
        :param tournament_size: Tamaño del torneo para selección por torneo
        :param crossover_rate: Probabilidad de aplicar cruce.
        :param replace_strategy: Estrategia de reemplazo en elitismo ('worst' o 'rnd').
        :param patience: Número de generaciones sin mejora antes de detener el algoritmo.
        """
        self.distances = distances
        self.num_cities = len(distances)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.replace_strategy = replace_strategy
        self.patience = patience

        # Inicializar estrategias
        self.selection = TournamentSelection(
            tournament_size=tournament_size,
            population_size=population_size,
            elite_size=elite_size
        )
        self.crossover = PMXCrossover()
        self.mutation = SwapMutation(mutation_rate)
        self.survival = GenerationalElitism()
        self.fitness = RouteFitness(distances)

        # Población inicial
        self.population = self._create_initial_population()

    def _create_initial_population(self) -> List[List[int]]:
        """
        Crea la población inicial de rutas aleatorias.
        """
        population = []
        for _ in range(self.population_size):
            route = list(range(self.num_cities))
            random.shuffle(route)
            population.append(route)
        return population

    def solve(self) -> Dict[str, Any]:
        """
        Ejecuta el algoritmo genético y devuelve la mejor ruta.
        """
        best_distances = []
        best_overall_distance = float('inf')
        no_improvement_count = 0

        with tqdm(total=self.generations, desc="Progreso", unit="gen") as pbar:
            for generation in range(self.generations):
                # Mejor de la generación actual
                best_current = min(self.population, key=self.fitness.calculate_route_distance)
                best_current_distance = self.fitness.calculate_route_distance(best_current)

                # Actualizar tracking de mejora
                if best_current_distance < best_overall_distance:
                    best_overall_distance = best_current_distance
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Criterio de parada
                if self.patience is not None and no_improvement_count >= self.patience:
                    print(f"Convergencia alcanzada en la generación {generation}.")
                    break

                # Selección de padres
                offspring = self.selection.select(
                    self.population,
                    fitness_func=self.fitness.calculate_route_distance
                )

                # Generar población completa nueva
                next_generation = []
                while len(next_generation) < self.population_size:
                    parent1, parent2 = random.sample(offspring, 2)

                    if random.random() < self.crossover_rate:
                        child1, child2 = self.crossover.crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()

                    child1 = self.mutation.mutate(child1)
                    child2 = self.mutation.mutate(child2)

                    if len(next_generation) < self.population_size:
                        next_generation.append(child1)
                    if len(next_generation) < self.population_size:
                        next_generation.append(child2)

                # Aplicar estrategia de supervivencia
                self.population = self.survival.select_survivors(
                    self.population,
                    next_generation,
                    self.fitness.calculate_route_distance
                )

                # Actualizar estadísticas
                best_route = min(self.population, key=self.fitness.calculate_route_distance)
                best_distance = self.fitness.calculate_route_distance(best_route)
                best_distances.append(best_distance)

                pbar.set_postfix({
                    "Mejor distancia": f"{best_distance:.2f}",
                    "Gen. sin mejora": no_improvement_count
                })
                pbar.update(1)

        best_route = min(self.population, key=self.fitness.calculate_route_distance)
        return {
            'route': best_route,
            'distance': self.fitness.calculate_route_distance(best_route),
            'evolution': best_distances,
            'generations_completed': generation + 1
        }