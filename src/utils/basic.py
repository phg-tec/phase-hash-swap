# src/utils/basic.py
import numpy as np
import math


def make_pair_with_cosine(dim: int, rho: float, seed: int = 123):
    """
    Genera dos vectores x,y de dimensión 'dim' con cos(x,y) ≈ rho.
    """
    rng = np.random.default_rng(seed)

    x = rng.standard_normal(dim)
    x /= np.linalg.norm(x)

    z = rng.standard_normal(dim)
    z -= x * np.dot(x, z)
    z /= np.linalg.norm(z)

    y = rho * x + math.sqrt(max(0.0, 1.0 - rho * rho)) * z
    return x, y


def cos_sim(u: np.ndarray, v: np.ndarray) -> float:
    """
    Coseno clásico entre dos vectores reales.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def p0_from_counts(counts: dict) -> float:
    """
    Extrae p0 = P(medir '0') a partir de un diccionario de cuentas de Qiskit.
    """
    shots = sum(counts.values())
    if shots <= 0:
        return 0.0
    return counts.get("0", 0) / shots


def corr_abs_from_p0(p0: float) -> float:
    """
    |⟨ψ|φ⟩| a partir de p0 del SWAP test:  p0 = (1 + |⟨ψ|φ⟩|^2)/2.
    """
    return math.sqrt(max(0.0, 2.0 * p0 - 1.0))


def corr_to_cos(corr: float) -> float:
    """
    Mapea una correlación en [0,1] al coseno estimado usando la fórmula
    de SimHash:  θ = (π/2)(1 - corr),  cos(θ).
    """
    c = max(-1.0, min(1.0, float(corr)))
    theta = (math.pi / 2.0) * (1.0 - c)
    return math.cos(theta)
