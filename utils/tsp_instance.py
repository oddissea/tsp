# ------------------------------------------------------------------------------
# Nombre:        Fernando H. Nasser-Eddine López
# Archivo:       tsp_instance.py
# Descripción:   Carga y representación gráfica de instancias del problema TSP
#                (Travelling Salesman Problem), cálculo de la matriz de distancias
#                y rutas posibles para algoritmos de optimización.
# Versión:       1.0
# Fecha:         14/12/2024
# ------------------------------------------------------------------------------

# utils/tsp_instance.py

import numpy as np
import matplotlib.pyplot as plt

class TSPInstance:
    def __init__(self, filepath):
        self.filepath = filepath
        self.cities = self._load_cities()
        self.num_cities = len(self.cities)
        self.distance_matrix = self._calculate_distance_matrix()

    def _load_cities(self):
        """Read the city coordinates from the TSP file."""
        cities = []
        with open(self.filepath, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3 and parts[0].isdigit():  # Detect coordinate lines
                    # Adjust the index to 0-based
                    # index = int(parts[0]) - 1
                    x, y = float(parts[1]), float(parts[2])
                    cities.append((x, y))
        return cities

    def _calculate_distance_matrix(self):
        """
        Generate a distance matrix based on the city coordinates using vectorized operations.
        The distances are computed as integers for simplicity and symmetry.

        Returns:
        :return numpy.ndarray: A symmetric distance matrix of size num_cities x num_cities.
        """
        # Convert city coordinates to a NumPy array
        coords = np.array(self.cities)

        # Calculate the differences between all pairs of coordinates
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

        # Compute the Euclidean distances between all pairs of cities
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Round the distances and return as integers
        # return np.round(distances).astype(int)
        return distances

    def get_distance(self, i, j):
        """Retrieve the precomputed distance between cities i and j."""
        return self.distance_matrix[i][j]

    def plot_cities(self):
        # Separar las coordenadas en listas X e Y
        x_vals, y_vals = zip(*self.cities)

        # Graficar las ciudades
        plt.figure(figsize=(8, 8))
        plt.scatter(x_vals, y_vals, c="blue", label="Ciudades")
        for i, (x, y) in enumerate(self.cities, start=0):
            plt.text(x, y, f"{i}", fontsize=8, ha="right", va="bottom")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.title("Representación de ciudades (TSP)")
        plt.legend()
        plt.grid(True)
        # plt.savefig("data/ciudades.png")
        plt.show()  # Esto aún intentará mostrarlo en pantalla

    def plot_route(self, route):
        # Graficar todas las ciudades en azul
        x_vals, y_vals = zip(*self.cities)
        plt.scatter(x_vals, y_vals, c="blue", label="Ciudades")
        for i, (x, y) in enumerate(self.cities, start=0):
            plt.text(x, y, f"{i}", fontsize=8, ha="right", va="bottom")

        # Extraer las coordenadas de las ciudades en el orden del recorrido
        if not route:
            # Si la ruta está vacía, solo mostrar las ciudades
            plt.xlabel("Coordenada X")
            plt.ylabel("Coordenada Y")
            plt.title("Ruta TSP")
            plt.legend()
            plt.grid(True)
            plt.show()
            return

        route_coords = [self.cities[i] for i in route]
        route_x, route_y = zip(*route_coords)

        # Dibujar la ruta en rojo
        plt.plot(route_x, route_y, marker='o', linestyle='-', color='red', label="Ruta")

        # Opcional: volver a la primera ciudad para cerrar el ciclo
        plt.plot([route_x[-1], route_x[0]], [route_y[-1], route_y[0]], 'r-')

        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.title("Ruta TSP")
        plt.legend()
        plt.grid(True)
        plt.show()
