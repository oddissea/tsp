# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       crossover.py
# Descripción:   Implementación del operador de cruce parcialmente mapeado (PMX)
#                para generar dos hijos a partir de dos padres en algoritmos genéticos.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------

from typing import List, Tuple
import random

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

        # 2. Rellenar los hijos utilizando una función auxiliar
        self._fill_remaining_genes(child1, parent1, parent2, start, end)
        self._fill_remaining_genes(child2, parent2, parent1, start, end)

        return child1, child2

    @staticmethod
    def _fill_remaining_genes(child: List[int], source: List[int], mapping_source: List[int], start: int, end: int):
        """
        Rellena los genes faltantes de un hijo, resolviendo conflictos utilizando el mapeo PMX.

        :param child: Hijo que se está completando.
        :param source: Padre fuente para los genes faltantes.
        :param mapping_source: Padre de donde se extrajo el segmento cruzado.
        :param start: Índice de inicio del segmento cruzado.
        :param end: Índice de fin del segmento cruzado.
        """
        n = len(child)
        for i in range(n):
            if i < start or i > end:
                value = source[i]
                # Resolver conflictos siguiendo el mapeo
                while value in mapping_source[start:end + 1]:
                    idx = mapping_source[start:end + 1].index(value)
                    value = source[start + idx]
                child[i] = value


if __name__ == "__main__":
    cruce = PMXCrossover()
    print(cruce.crossover([1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1]))

