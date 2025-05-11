# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       survival.py
# Descripción:   Implementación de la estrategia de supervivencia generacional
#                con elitismo para un algoritmo genético. Preserva el mejor
#                individuo actual reemplazando al peor o a uno aleatorio
#                en la nueva generación si es necesario.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------
# src/survival.py
import random
from typing import List, Callable


class GenerationalElitism:
    def __init__(self, replace_strategy: str = "worst"):
        """
        Inicializa la estrategia de supervivencia generacional con elitismo.

        :param replace_strategy: Estrategia de reemplazo ('worst' o 'rnd')
        """
        if replace_strategy not in ["worst", "rnd"]:
            raise ValueError("Estrategia de reemplazo debe ser 'worst' o 'rnd'")
        self.replace_strategy = replace_strategy

    def select_survivors(self,
                         current_population: List[List[int]],
                         next_generation: List[List[int]],
                         fitness_func: Callable) -> List[List[int]]:
        """
        Aplica la estrategia de supervivencia generacional con elitismo.

        :param current_population: Población actual
        :param next_generation: Nueva generación completa
        :param fitness_func: Función de fitness (menor es mejor)
        :return: Nueva generación después de aplicar elitismo
        """
        # Encontrar el mejor de la generación actual
        best_current = min(current_population, key=fitness_func)
        best_current_fitness = fitness_func(best_current)

        # Encontrar el mejor de la nueva generación
        best_new = min(next_generation, key=fitness_func)
        best_new_fitness = fitness_func(best_new)

        # Si el mejor actual es mejor que cualquiera de la nueva generación
        if best_current_fitness < best_new_fitness:
            if self.replace_strategy == "worst":
                # Reemplazar al peor
                worst_idx = max(
                    range(len(next_generation)),
                    key=lambda i: fitness_func(next_generation[i])
                )
            else:  # "rnd"
                # Reemplazar a uno aleatorio
                worst_idx = random.randrange(len(next_generation))

            next_generation[worst_idx] = best_current

        return next_generation