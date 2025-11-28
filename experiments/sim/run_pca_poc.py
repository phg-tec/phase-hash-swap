#!/usr/bin/env python
"""
PoC: AE-SWAP vs deterministic PCA-sign hash SWAP

- Genera pares (x,y) con coseno ~ rho.
- AE-SWAP: amplitud encoding.
- PCA-hash SWAP: proyección PCA + signo + fase + SWAP.

Requiere: qiskit, qiskit-aer, numpy.
"""

import math
import time
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import DiagonalGate


# ----------------------------------------------------------
# Utilidades básicas
# ----------------------------------------------------------
def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def make_pair_with_cosine(dim, rho, rng):
    """
    Genera x,y en R^dim con cos(x,y) ≈ rho.
    y = rho*x + sqrt(1-rho^2)*z, z ortogonal a x.
    """
    x = rng.standard_normal(dim)
    x /= np.linalg.norm(x)

    z = rng.standard_normal(dim)
    z -= x * np.dot(x, z)
    z /= np.linalg.norm(z)

    y = rho * x + math.sqrt(max(0.0, 1 - rho**2)) * z
    return x, y


def p0_from_counts(counts):
    shots = sum(counts.values())
    return counts.get("0", 0) / max(1, shots)


def corr_abs_from_p0(p0):
    # relación estándar del SWAP test
    return math.sqrt(max(0.0, 2 * p0 - 1.0))


def corr_to_cos(corr_signed):
    # Estimador SimHash-style: cos(phi) ≈ cos(pi/2 * (1 - r))
    c = max(-1.0, min(1.0, float(corr_signed)))
    theta = (math.pi / 2.0) * (1.0 - c)
    return math.cos(theta)


# ----------------------------------------------------------
# PCA determinista (sin sklearn)
# ----------------------------------------------------------
def compute_pca_components(X, m):
    """
    PCA clásica vía SVD.
    X: (N, d)
    Devuelve matriz U de tamaño (m, d) con los m primeros componentes.
    """
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0, keepdims=True)
    # Xc = U*S*Vt  → filas de Vt son las direcciones principales
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    U = Vt[:m, :]  # (m, d)
    return U


def pca_sign_hash(x, U):
    """
    Hash determinista PCA + signo:
        b = sign(U @ x) en {+1,-1}^m
    """
    x = np.asarray(x, float)
    z = U @ x
    b = np.where(z >= 0, 1.0, -1.0)
    return b


# ----------------------------------------------------------
# Preparación de estados cuánticos
# ----------------------------------------------------------
def build_amp_state(vec):
    """
    Amplitude encoding usando Initialize.
    Rellena con ceros hasta dimensión potencia de 2.
    """
    from qiskit.circuit.library import Initialize

    vec = np.asarray(vec, float)
    amp = normalize(vec)

    n = int(math.ceil(math.log2(len(amp))))
    if len(amp) < 2 ** n:
        pad = np.zeros(2 ** n, dtype=complex)
        pad[: len(amp)] = amp.astype(complex)
        amp = pad

    q = QuantumRegister(n, "data")
    qc = QuantumCircuit(q, name="amp_state")
    qc.append(Initialize(amp), q[:])
    return qc


def build_phase_state(bits_pm1):
    """
    Codificación de fase con ±1:
    - Partimos de superposición uniforme H^{⊗n}
    - Aplicamos fase +1/-1 en la base computacional (DiagonalGate)
    """
    bits_pm1 = np.asarray(bits_pm1, float)
    m = len(bits_pm1)
    n = int(math.log2(m))
    if 2 ** n != m:
        raise ValueError(f"bits_pm1 length {m} is not power of 2")

    q = QuantumRegister(n, "data")
    qc = QuantumCircuit(q, name="phase_state")
    qc.h(q)

    phases = [1.0 if b > 0 else -1.0 for b in bits_pm1]
    qc.append(DiagonalGate(phases), q[:])
    return qc


# ----------------------------------------------------------
# SWAP test
# ----------------------------------------------------------
def build_swap_test(prepA, prepB):
    """
    Construye SWAP test estándar para dos circuitos de preparación.
    """
    n = prepA.num_qubits
    anc = QuantumRegister(1, "anc")
    qa = QuantumRegister(n, "a")
    qb = QuantumRegister(n, "b")
    c = ClassicalRegister(1, "m")

    qc = QuantumCircuit(anc, qa, qb, c, name="swap_test")
    qc.compose(prepA, qa, inplace=True)
    qc.compose(prepB, qb, inplace=True)

    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], qa[i], qb[i])
    qc.h(anc[0])
    qc.measure(anc[0], c[0])
    return qc


# ----------------------------------------------------------
# Runners: AE-SWAP y PCA-hash SWAP
# ----------------------------------------------------------
def run_swap_amp(x, y, shots=2048, seed=None, opt_level=3):
    """
    SWAP test con encoding de amplitud.

    Devuelve:
      cos_hat : estimación del coseno (con signo)
    """
    sim = AerSimulator(seed_simulator=seed)

    x = normalize(x)
    y = normalize(y)

    prep_x = build_amp_state(x)
    prep_y = build_amp_state(y)
    qc = build_swap_test(prep_x, prep_y)
    
    
    tqc = transpile(qc, sim, optimization_level=opt_level, seed_transpiler=seed)
    
    start = time.time()
    res = sim.run(tqc, shots=shots).result()
    ex_time = time.time() - start
    p0 = p0_from_counts(res.get_counts(tqc))
    overlap_abs = corr_abs_from_p0(p0)  # |<x|y>|

    # signo conocido clásicamente vía producto escalar (solo para PoC)
    sign_xy = np.sign(float(np.dot(x, y))) or 1.0
    cos_hat = sign_xy * overlap_abs
    return cos_hat, ex_time


def run_swap_pca_hash(x, y, U, shots=2048, seed=None, opt_level=3):
    """
    SWAP test con codificación PCA-sign + fase.

    U: matriz PCA (m x d), con m potencia de 2.
    Devuelve:
      cos_hat : estimación del coseno (vía correlación binaria + SWAP).
    """
    sim = AerSimulator(seed_simulator=seed)

    # embedding determinista en ±1
    b = pca_sign_hash(x, U)
    c = pca_sign_hash(y, U)

    # signo aproximado de la correlación a partir del hash binario
    sign_proxy = np.sign(np.mean(b * c)) or 1.0

    prep_b = build_phase_state(b)
    prep_c = build_phase_state(c)
    qc = build_swap_test(prep_b, prep_c)
    
    tqc = transpile(qc, sim, optimization_level=opt_level, seed_transpiler=seed)
    start = time.time()
    res = sim.run(tqc, shots=shots).result()
    ex_time = time.time() - start
    p0 = p0_from_counts(res.get_counts(tqc))
    corr_abs = corr_abs_from_p0(p0)

    corr_signed = corr_abs * sign_proxy
    cos_hat = corr_to_cos(corr_signed)
    return cos_hat, ex_time


# ----------------------------------------------------------
# Experimento de prueba de concepto
# ----------------------------------------------------------
def poc_experiment():
    rng = np.random.default_rng(123)

    dim = 2046              # dimensión original
    m = 1024                # m bits (m = potencia de 2 para phase-state)
    N_pca = 1024           # muestras para ajustar PCA
    shots = 2048
    rhos = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9]
    pairs_per_rho = 5     # par(es) por cada rho (sube esto si quieres más robustez)

    print("=== Construyendo PCA para PCA-sign hash (determinista) ===")
    # dataset aleatorio para PCA (aquí podría ser tu dataset real)
    X = rng.standard_normal((N_pca, dim))
    U = compute_pca_components(X, m)  # (m, d)

    print(f"Dimensión original d = {dim}")
    print(f"Dimensión hash m = {m} → n_qubits = log2(m) = {int(math.log2(m))}")
    print("")

    results = []

    for rho in rhos:
        ae_errs = []
        pca_errs = []

        for _ in range(pairs_per_rho):
            x, y = make_pair_with_cosine(dim, rho, rng)

            # AE-SWAP
            seed_int = int(rng.integers(1 << 30))
            cos_hat_ae, ae_time = run_swap_amp(x, y, shots=shots, seed=seed_int)
            ae_errs.append(abs(cos_hat_ae - rho))

            # PCA-hash SWAP
            seed_int = int(rng.integers(1 << 30))
            cos_hat_pca, pca_time = run_swap_pca_hash(x, y, U, shots=shots, seed=seed_int)
            pca_errs.append(abs(cos_hat_pca - rho))

        mae_ae = float(np.mean(ae_errs))
        mae_pca = float(np.mean(pca_errs))
        results.append((rho, mae_ae, mae_pca))

        print(f"rho = {rho:+.2f} | MAE AE-SWAP = {mae_ae:.4f} | MAE PCA-hash SWAP = {mae_pca:.4f}")
        print(f"Time AE-SWAP = {ae_time:.4f} | Time PCA-hash SWAP = {pca_time:.4f}")

    print("\n=== Resumen global (MAE promedio) ===")
    mae_ae_global = float(np.mean([r[1] for r in results]))
    mae_pca_global = float(np.mean([r[2] for r in results]))
    print(f"AE-SWAP       MAE medio = {mae_ae_global:.4f}")
    print(f"PCA-hash SWAP MAE medio = {mae_pca_global:.4f}")


if __name__ == "__main__":
    poc_experiment()
