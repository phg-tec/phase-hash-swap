# src/utils/discretization.py
import numpy as np
from sklearn.cluster import KMeans


def learn_kmeans_centers(x: np.ndarray,
                         y: np.ndarray,
                         K: int = 5,
                         seed: int = 0) -> np.ndarray:
    """
    Aprende K centros representativos a partir de los componentes de x e y.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    data = np.concatenate([x, y]).reshape(-1, 1)

    km = KMeans(n_clusters=K, n_init=1000, random_state=seed, max_iter=2000)
    km.fit(data)
    return np.sort(km.cluster_centers_.flatten())


def quantize_to_centers(v: np.ndarray, centers: np.ndarray):
    """
    Asigna cada componente de v al centro más cercano.
    Devuelve:
      - valores cuantizados (mismos shape que v)
      - índices de centro (enteros en [0, K-1])
    """
    v = np.asarray(v, float)
    centers = np.asarray(centers, float)

    d = np.abs(v[:, None] - centers[None, :])
    idx = np.argmin(d, axis=1)
    return centers[idx], idx


def kmeans_to_centers(x, K=2):
    """
    Ejecuta KMeans sobre x y devuelve:
      - valores cuantizados (mismo shape que x)
      - índices de centro (enteros en [0, K-1])
      - centros ordenados (vector de tamaño K)
    """
    x = np.asarray(x, float)
    z = x.reshape(-1, 1)

    km = KMeans(n_clusters=K, n_init=100, random_state=123).fit(z)

    labels = km.labels_                    # índices de centros
    centers = km.cluster_centers_.flatten() # valores de cada centro

    quantized = centers[labels]            # asignar cada x_i a su centro

    return quantized, labels
