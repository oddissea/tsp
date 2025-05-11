# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       random_seed.py
# Descripción:   Configuración de semillas globales para garantizar la reproducibilidad
#                en experimentos utilizando generadores aleatorios de Python y NumPy.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------

# utils/random_seed.py
import random
import numpy as np

def set_seed(seed: int):
    """
    Configura la semilla global para reproducibilidad.
    :param seed: Semilla para los generadores aleatorios de Python y NumPy.
    """
    random.seed(seed)
    np.random.seed(seed)
