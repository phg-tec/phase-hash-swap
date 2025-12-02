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

    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
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


def idx_to_phases(idx: np.ndarray, K: int) -> np.ndarray:
    """
    Convierte índices de nivel (0..K-1) en fases uniformes en [0, 2π).
    """
    idx = np.asarray(idx, int)
    return 2.0 * np.pi * idx / float(K)
