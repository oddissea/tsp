# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       crossover.py
# Descripción:   Implementación del operador de cruce parcialmente mapeado (PMX)
#                para generar dos hijos a partir de dos padres en algoritmos genéticos.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------

# src/crossover.py
import random
from typing import List, Tuple


class PMXCrossover:
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Operador de cruce parcialmente mapeado (PMX) que genera dos hijos.

        :param parent1: Primer padre
        :param parent2: Segundo padre
        :return: Tupla con los dos hijos (h1, h2)
        """
        n = len(parent1)
        # Seleccionar dos puntos de cruce al azar
        start, end = sorted(random.sample(range(n), 2))

        # Inicializar los hijos
        child1 = [None] * n
        child2 = [None] * n

        # 1. Copiar el segmento seleccionado
        child1[start:end + 1] = parent2[start:end + 1]
        child2[start:end + 1] = parent1[start:end + 1]

        # 2. Rellenar el primer hijo
        for i in range(n):
            if i < start or i > end:
                value = parent1[i]
                # Si hay conflicto, seguir el mapeo
                if value in parent2[start:end + 1]:
                    current = value
                    while current in parent2[start:end + 1]:
                        idx = parent2[start:end + 1].index(current)
                        current = parent1[start + idx]
                    value = current
                child1[i] = value

        # 3. Rellenar el segundo hijo
        for i in range(n):
            if i < start or i > end:
                value = parent2[i]
                # Si hay conflicto, seguir el mapeo
                if value in parent1[start:end + 1]:
                    current = value
                    while current in parent1[start:end + 1]:
                        idx = parent1[start:end + 1].index(current)
                        current = parent2[start + idx]
                    value = current
                child2[i] = value

        return child1, child2

if __name__ == "__main__":
    cruce = PMXCrossover()
    print(cruce.crossover([1, 2, 3, 4, 5, 6, 7, 8], [6, 5, 3, 1, 7, 4, 8, 2]))

