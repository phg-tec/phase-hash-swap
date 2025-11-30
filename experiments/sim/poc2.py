#!/usr/bin/env python3
"""
Comparación AE-SWAP vs PES-PCA (PCA+sign SWAP).

- Genera pares (x,y) en R^dim con coseno objetivo.
- AE-SWAP: amplitude encoding + SWAP test.
- PES-PCA: hash determinista PCA+sign + fases ±1 + SWAP test.
- Escribe resultados en un CSV y los muestra por pantalla.

Requisitos:
    - qiskit, qiskit-aer
    - pandas
    - opcional: scikit-learn (para PCA); si no está, usa numpy.linalg.eigh
"""

import math
import time
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import DiagonalGate

# Opcional: usar PCA de sklearn si está disponible
try:
    from sklearn.decomposition import PCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# =========================================================
# Utilidades básicas
# =========================================================

def make_pair_with_cosine(dim, rho, seed=123):
    """
    Genera x,y en R^dim con coseno aprox. rho.
    Construcción: y = rho*x + sqrt(1-rho^2)*z, z ortogonal a x.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(dim)
    x /= np.linalg.norm(x)

    z = rng.standard_normal(dim)
    z -= x * np.dot(x, z)
    z /= np.linalg.norm(z)

    y = rho * x + math.sqrt(max(0.0, 1 - rho**2)) * z
    return x, y


# =========================================================
# Preparación de estados cuánticos
# =========================================================
def build_amp_state(vec):
    """
    Amplitude encoding usando Initialize.
    Rellena con ceros hasta 2^n si hace falta.
    """
    from qiskit.circuit.library import Initialize

    vec = np.asarray(vec, dtype=float)
    n = int(math.ceil(math.log2(len(vec))))
    qc = QuantumCircuit(n, name="amp_state")

    amp = vec

    if len(amp) < 2**n:
        pad = np.zeros(2**n, dtype=complex)
        pad[:len(amp)] = amp.astype(complex)
        amp = pad

    qc.append(Initialize(amp), qc.qubits)
    return qc


def build_phase_state(bits_pm1):
    """
    Phase-hash encoding:
    - Empezar en superposición uniforme
    - Aplicar DiagonalGate con fases ±1
    bits_pm1 debe tener longitud potencia de 2.
    """
    bits_pm1 = np.asarray(bits_pm1, float)
    m = len(bits_pm1)
    n = int(math.log2(m))
    if 2**n != m:
        raise ValueError(f"bits_pm1 length {m} is not power of 2")

    q = QuantumRegister(n, "data")
    qc = QuantumCircuit(q, name="phase_state")
    qc.h(q)

    phases = [1.0 if b > 0 else -1.0 for b in bits_pm1]
    qc.append(DiagonalGate(phases), q[:])
    return qc


# =========================================================
# SWAP test
# =========================================================
def build_swap_test(prepA, prepB):
    n = prepA.num_qubits
    anc = QuantumRegister(1, "anc")
    qa  = QuantumRegister(n, "a")
    qb  = QuantumRegister(n, "b")
    c   = ClassicalRegister(1, "m")

    qc = QuantumCircuit(anc, qa, qb, c)
    qc.compose(prepA, qa, inplace=True)
    qc.compose(prepB, qb, inplace=True)

    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], qa[i], qb[i])
    qc.h(anc[0])
    qc.measure(anc[0], c[0])
    return qc


# =========================================================
# Medidas / conversiones
# =========================================================
def p0_from_counts(counts):
    shots = sum(counts.values())
    return counts.get("0", 0) / max(1, shots)


def corr_abs_from_p0(p0):
    # |⟨ψ|φ⟩| = sqrt(2 p0 - 1)
    return math.sqrt(max(0.0, 2*p0 - 1.0))


def corr_to_cos(corr_signed):
    """
    Misma conversión que usas en tu código:
    corr_signed ∈ [-1,1] → estimación de coseno.
    """
    c = max(-1.0, min(1.0, float(corr_signed)))
    theta = (math.pi/2) * (1 - c)
    return math.cos(theta)


def count_twoq(tcirc):
    """
    Conteo heurístico de puertas de 2 qubits.
    """
    ops = tcirc.count_ops()
    total = int(sum(ops.get(g, 0) for g in
                    ["cx", "cz", "swap", "ecr", "rxx", "ryy", "rzx", "rzz"]))
    total += 3*int(ops.get("cswap", 0)) + 6*int(ops.get("ccx", 0))
    return total


# =========================================================
# AE-SWAP
# =========================================================
def run_swap_amp(x, y, shots=2048, seed=None, opt_level=3, measure_cost=False):
    """
    Devuelve:
      - cos_hat (estimación del cos(x,y) con signo) + tiempo
      - si measure_cost=True: también depth y nº de puertas de 2q.
    """
    sim = AerSimulator(seed_simulator=seed)

    prep_x = build_amp_state(x)
    prep_y = build_amp_state(y)
    qc = build_swap_test(prep_x, prep_y)

    tqc = transpile(qc, sim, optimization_level=opt_level,
                    seed_transpiler=seed)

    # Ejecutamos circuito
    t0 = time.time()
    res = sim.run(tqc, shots=shots).result()
    elapsed = time.time() - t0

    # SWAP test → |cos|
    p0 = p0_from_counts(res.get_counts(tqc))
    print ("Counts AE: ", res.get_counts(tqc).get("0", 0))
    overlap_abs = corr_abs_from_p0(p0)   # = |⟨ψ_x | ψ_y⟩| = |cos|

    # Recuperamos el signo usando el producto escalar real directo
    sign = np.sign(np.dot(x, y)) or 1.0  # evita 0 en casos borde

    cos_hat = sign * overlap_abs

    if not measure_cost:
        return cos_hat, elapsed

    return cos_hat, elapsed, tqc.depth(), count_twoq(tqc)



# =========================================================
# PCA + sign hashing (determinista)
# =========================================================
_PCA_HASH_CACHE = {}  # clave: (dim, m, n_train, seed) → (W, mu)


def get_pca_hash_operator(dim, m, n_train=4096, seed=123):
    """
    Construye (y cachea) el operador de hash PCA+sign para
    dimensión 'dim' y nº bits 'm'.

    Aquí entrenamos PCA sobre vectores gaussianos normalizados.
    Para series temporales reales, querrás sustituir esto por
    tus ventanas reales.
    """
    if m > dim:
        raise ValueError(f"PCA hash: m={m} no puede ser mayor que dim={dim}")

    key = (dim, m, n_train, seed)
    if key in _PCA_HASH_CACHE:
        return _PCA_HASH_CACHE[key]

    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_train, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    if _HAS_SKLEARN:
        pca = PCA(n_components=m, svd_solver="auto", random_state=seed)
        X_centered = X - X.mean(axis=0, keepdims=True)
        pca.fit(X_centered)
        W = pca.components_          # (m, dim)
        mu = pca.mean_               # (dim,)
    else:
        # Fallback: PCA vía eigen-descomposición
        mu = X.mean(axis=0)
        Xc = X - mu
        S = (Xc.T @ Xc) / (Xc.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(S)
        idx = np.argsort(eigvals)[::-1][:m]
        W = eigvecs[:, idx].T  # (m, dim)

    _PCA_HASH_CACHE[key] = (W, mu)
    return W, mu


def pca_sign_hash(v, W, mu):
    """
    Hash PCA+sign:
      z = W @ (v - mu)
      bits = sign(z) ∈ {-1, +1}^m
    """
    v = np.asarray(v, float)
    z = W @ (v - mu)
    bits_pm1 = np.where(z >= 0, 1.0, -1.0)
    return bits_pm1


# =========================================================
# PES-PCA (PCA+sign SWAP)
# =========================================================
def run_swap_pca_sign(x, y, m, shots=2048, seed=123,
                      opt_level=3, measure_cost=False,
                      pca_train_size=4096):
    """
    SWAP test con hash determinista PCA+sign.

    - Construye un operador PCA (W, mu) para (dim, m) una vez.
    - Hashea x,y -> bits ±1 con sign(W @ (v - mu)).
    - Esos bits se codifican como fases ±1 y se usa el SWAP test.
    - E repeticiones solo promedian el ruido de medición (los hashes
      no cambian con E).

    Devuelve:
      cos_hat, elapsed  (o plus avg depth,twoq si measure_cost)
    """
    sim = AerSimulator()
    dim = len(x)

    W, mu = get_pca_hash_operator(dim, m, n_train=pca_train_size, seed=seed)

    corr_list = []
    depths, twoqs = [], []

    t0 = time.time()
    b = pca_sign_hash(x, W, mu)
    c = pca_sign_hash(y, W, mu)

    # Por seguridad, signo de la correlación clásica de bits
    sign_proxy = np.sign(np.mean(b * c)) or 1.0

    prep_b = build_phase_state(b)
    prep_c = build_phase_state(c)
    qc = build_swap_test(prep_b, prep_c)

    seed_t = int(seed)
    seed_s = int(seed)

    tqc = transpile(qc, sim, optimization_level=opt_level,
                    seed_transpiler=seed_t)
    res = sim.run(tqc, shots=shots, seed_simulator=seed_s).result()

    p0 = p0_from_counts(res.get_counts(tqc))
    print ("Counts PES: ", res.get_counts(tqc).get("0", 0))
    corr_abs = corr_abs_from_p0(p0)
    corr_signed = corr_abs * sign_proxy

    corr_list.append(corr_signed)

    if measure_cost:
        depths.append(tqc.depth())
        twoqs.append(count_twoq(tqc))

    elapsed = time.time() - t0
    corr_hat = float(np.mean(corr_list))
    cos_hat = corr_to_cos(corr_hat)

    if not measure_cost:
        return cos_hat, elapsed

    return cos_hat, elapsed, float(np.mean(depths)), float(np.mean(twoqs))


# =========================================================
# MAIN: comparación AE-SWAP vs PES-PCA
# =========================================================
def main():
    # -----------------------------------------------------
    # Configuración del experimento (ajusta a tu gusto)
    # -----------------------------------------------------
    dims        = [512]                 # dimensiones a probar
    cos_targets = [-0.9, 0.9]
    shots_list  = [4096]
    m_list      = [256]                 # nº bits (debe ser potencia de 2)
    reps        = 1

    measure_cost   = True
    opt_level      = 3
    base_seed0     = 0
    pca_train_size = 4096

    out_csv = "results_ae_vs_pes_pca.csv"

    rows = []

    for rep in range(reps):
        for dim in dims:
            base_seed = base_seed0 + 1000*rep + dim
            for true_cos in cos_targets:
                x, y = make_pair_with_cosine(dim, true_cos, seed=base_seed)

                for shots in shots_list:

                    # ==========================
                    # AE-SWAP
                    # ==========================
                    if measure_cost:
                        overlap_amp, t_amp, d_amp, q_amp = run_swap_amp(
                            x, y, shots=shots, seed=base_seed,
                            opt_level=opt_level, measure_cost=True
                        )
                    else:
                        overlap_amp, t_amp = run_swap_amp(
                            x, y, shots=shots, seed=base_seed,
                            opt_level=opt_level, measure_cost=False
                        )
                        d_amp, q_amp = "-", "-"

                    cos_amp = corr_to_cos(overlap_amp)
                    rows.append([
                        "AE-SWAP", rep, dim, true_cos, "-", "-",
                        shots, cos_amp, abs(cos_amp - true_cos),
                        t_amp, d_amp, q_amp
                    ])

                    # ==========================
                    # PES-PCA (PCA+sign SWAP)
                    # ==========================
                    for m in m_list:
                        if measure_cost:
                            cos_pca, t_pca, d_pca, q_pca = run_swap_pca_sign(
                                x, y, m=m, shots=shots,
                                seed=base_seed + 10_000 + m,
                                opt_level=opt_level,
                                measure_cost=True,
                                pca_train_size=pca_train_size,
                            )
                        else:
                            cos_pca, t_pca = run_swap_pca_sign(
                                x, y, m=m, shots=shots,
                                seed=base_seed + 10_000 + m,
                                opt_level=opt_level,
                                measure_cost=False,
                                pca_train_size=pca_train_size,
                            )
                            d_pca, q_pca = "-", "-"

                        rows.append([
                            "PES-PCA (SWAP)", rep, dim, true_cos, m,
                            shots, cos_pca, abs(cos_pca - true_cos),
                            t_pca, d_pca, q_pca
                        ])

    df = pd.DataFrame(rows, columns=[
        "Method", "Rep", "Dim", "TrueCos", "m", "Shots",
        "EstCos", "AbsErr", "Time_s", "Depth", "TwoQ"
    ])

    #df.to_csv(out_csv, index=False)
    #print(f"\n✅ Resultados guardados en: {out_csv} ({len(df)} filas)\n")
    #print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
