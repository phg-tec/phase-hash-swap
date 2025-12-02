#!/usr/bin/env python
"""
PES-multilevel SWAP: AE-SWAP vs PES global vs PES-MULTISWAP (fases)
-------------------------------------------------------------------

Incluye:

  - AE-SWAP (amplitude encoding).
  - PES data-driven global (una sola DiagonalGate con fases por nivel).
  - PES-MULTISWAP estratificado por grupos (p,q) usando fases continuas
    φ = α * valor_discretizado (k-medias sobre x,y).

Se comparan los MAE de cada método respecto al coseno "real" continuo.
"""

import numpy as np
import math
from math import log2
from collections import defaultdict

from sklearn.cluster import KMeans

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


# ============================================================
# Utilidades básicas
# ============================================================

def make_pair_with_cosine(dim, rho, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(dim)
    x /= np.linalg.norm(x)

    z = rng.standard_normal(dim)
    z -= x * np.dot(x, z)
    z /= np.linalg.norm(z)

    y = rho * x + math.sqrt(max(0, 1 - rho * rho)) * z
    return x, y


def cos_sim(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def p0_from_counts(counts):
    shots = sum(counts.values())
    return counts.get("0", 0) / max(1, shots)


def corr_abs_from_p0(p0):
    return math.sqrt(max(0.0, 2 * p0 - 1.0))


def corr_to_cos(corr):
    c = max(-1, min(1, float(corr)))
    theta = (math.pi / 2) * (1 - c)
    return math.cos(theta)


# ============================================================
# AE-SWAP
# ============================================================

def build_amp_state(vec):
    vec = np.asarray(vec, float)
    n = int(np.ceil(np.log2(len(vec))))
    pad_len = 2**n
    amp = np.zeros(pad_len, complex)
    amp[:len(vec)] = vec / np.linalg.norm(vec)

    from qiskit.circuit.library import Initialize
    qr = QuantumRegister(n, "amp")
    qc = QuantumCircuit(qr)
    qc.append(Initialize(amp), qr)
    return qc


def build_swap_test(prepA, prepB):
    n = prepA.num_qubits
    anc = QuantumRegister(1, "anc")
    qa = QuantumRegister(n, "a")
    qb = QuantumRegister(n, "b")
    c = ClassicalRegister(1, "m")

    qc = QuantumCircuit(anc, qa, qb, c)
    qc.compose(prepA, qa, inplace=True)
    qc.compose(prepB, qb, inplace=True)

    qc.h(anc)
    for i in range(n):
        qc.cswap(anc[0], qa[i], qb[i])
    qc.h(anc)
    qc.measure(anc[0], c[0])
    return qc


def run_ae_swap(x, y, shots=2048, seed=123, opt_level=3):
    sim = AerSimulator(seed_simulator=seed)

    px = build_amp_state(x)
    py = build_amp_state(y)
    qc = build_swap_test(px, py)

    tqc = transpile(qc, sim, optimization_level=opt_level, seed_transpiler=seed)
    res = sim.run(tqc, shots=shots).result()

    p0 = p0_from_counts(res.get_counts(tqc))
    return corr_to_cos(corr_abs_from_p0(p0))


# ============================================================
# K-means discretization DIRECTLY ON x,y
# ============================================================

def learn_kmeans_centers(x, y, K=5, seed=0):
    """Aprende K valores representativos sobre todos los componentes."""
    data = np.concatenate([x, y]).reshape(-1, 1)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    km.fit(data)
    return np.sort(km.cluster_centers_.flatten())


def quantize_to_centers(v, centers):
    """Asigna cada componente al centro más cercano, devuelve valores e índices."""
    v = np.asarray(v)
    centers = np.asarray(centers)
    d = np.abs(v[:, None] - centers[None, :])
    idx = np.argmin(d, axis=1)
    return centers[idx], idx


def idx_to_phases(idx, K):
    """Convierte el índice del nivel a fase uniforme [0,2π)."""
    return 2 * np.pi * idx / K


# ============================================================
# PES global (una sola DiagonalGate con fases por nivel)
# ============================================================

def build_phase_state(phases):
    phases = np.asarray(phases, float)
    m = len(phases)
    n = int(np.log2(m))
    if 2**n != m:
        # pad hasta potencia de 2
        new_m = 1
        while new_m < m:
            new_m *= 2
        pad = np.zeros(new_m)
        pad[:m] = phases
        phases = pad
        m = new_m
        n = int(np.log2(m))

    q = QuantumRegister(n, "data")
    qc = QuantumCircuit(q)
    qc.h(q)
    diag = np.exp(1j * phases)
    qc.append(DiagonalGate(diag), q)
    return qc


def run_pes_datadriven_global(x, y, centers, shots=2048, seed=123, opt_level=3):
    """
    Versión PES "global": una sola DiagonalGate con fases por nivel (idx).
    """
    sim = AerSimulator(seed_simulator=seed)

    qx, idx_x = quantize_to_centers(x, centers)
    qy, idx_y = quantize_to_centers(y, centers)

    cos_disc = cos_sim(qx, qy)   # coseno clásico tras discretización

    K = len(centers)
    phix = idx_to_phases(idx_x, K)
    phiy = idx_to_phases(idx_y, K)

    px = build_phase_state(phix)
    py = build_phase_state(phiy)
    qc = build_swap_test(px, py)

    tqc = transpile(qc, sim, optimization_level=opt_level, seed_transpiler=seed)
    res = sim.run(tqc, shots=shots).result()

    cos_hat = corr_to_cos(corr_abs_from_p0(p0_from_counts(res.get_counts(tqc))))
    cos_real = cos_sim(x, y)

    mae_disc = abs(cos_real - cos_disc)
    mae_pes = abs(cos_real - cos_hat)

    return cos_real, cos_disc, cos_hat, mae_disc, mae_pes


# ============================================================
# NUEVO: PES-MULTISWAP (fases continuas por grupos)
# ============================================================

def build_partition_groups_from_indices(idx_x, idx_y, centers):
    """
    Particiona las coordenadas en grupos etiquetados por (k,l),
    donde k,l son índices de centros (0..K-1).

    Devuelve:
      groups: dict (k,l) -> lista índices i
    """
    idx_x = np.asarray(idx_x, int)
    idx_y = np.asarray(idx_y, int)
    K = len(centers)

    groups = defaultdict(list)
    for i in range(len(idx_x)):
        k = idx_x[i]
        l = idx_y[i]
        # Usamos (min,max) o (k,l)? Aquí mantenemos (k,l) tal cual,
        # pues la fórmula de agregación no requiere simetrizar.
        groups[(k, l)].append(i)

    return groups


def classical_cos_from_groups_quantized(qx, qy, groups):
    """
    Versión clásica estratificada, usando los vectores discretizados qx, qy:

      cos(qx,qy)
      = [ Σ_g cos_g * sqrt(Na_g * Nb_g) ] / sqrt( Σ Na_g * Σ Nb_g )
    """
    qx = np.asarray(qx, float)
    qy = np.asarray(qy, float)

    Na_g = {}
    Nb_g = {}
    cos_g = {}

    for key, idxs in groups.items():
        idxs = np.asarray(idxs)
        sub_a = qx[idxs]
        sub_b = qy[idxs]

        na = float(sub_a @ sub_a)
        nb = float(sub_b @ sub_b)
        Na_g[key] = na
        Nb_g[key] = nb

        if na > 0 and nb > 0:
            cos_g[key] = float(sub_a @ sub_b) / math.sqrt(na * nb)
        else:
            cos_g[key] = 0.0

    Na = sum(Na_g.values())
    Nb = sum(Nb_g.values())
    num = sum(cos_g[g] * math.sqrt(Na_g[g] * Nb_g[g]) for g in groups)

    cos_hat = num / math.sqrt(Na * Nb)
    return cos_hat, cos_g, Na_g, Nb_g


# --- utilidades de fase continua (como en tu script nuevo) ---

def next_power_of_two(m: int) -> int:
    if m <= 1:
        return 1
    return 1 << (m - 1).bit_length()


def pad_values_to_power_of_two(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    L = len(values)
    m = next_power_of_two(L)
    if m == L:
        return values
    pad_len = m - L
    pad = np.zeros(pad_len)
    return np.concatenate([values, pad])


def build_phase_state_from_values(values: np.ndarray, alpha: float):
    """
    Construye el estado:

      |ψ> = (1/√m) Σ_i e^{i * α * values[i]} |i>
    """
    values = np.asarray(values, float)
    m = len(values)
    n = int(log2(m))
    if 2**n != m:
        raise ValueError("build_phase_state_from_values: longitud no potencia de 2")

    phases = alpha * values
    diag_vals = [np.exp(1j * phi) for phi in phases]

    qr = QuantumRegister(n, "data")
    qc = QuantumCircuit(qr, name="phase_state")

    qc.h(qr)
    diag_gate = DiagonalGate(diag_vals)
    qc.append(diag_gate, qr)

    return qc


def build_phase_swap_circuit(sub_a, sub_b, alpha: float):
    """
    SWAP test entre dos estados de fase codificados a partir
    de sub_a, sub_b (valores discretizados del grupo).
    """
    sub_a = np.asarray(sub_a, float)
    sub_b = np.asarray(sub_b, float)
    if len(sub_a) != len(sub_b):
        raise ValueError("build_phase_swap_circuit: longitudes distintas")

    vals_a = pad_values_to_power_of_two(sub_a)
    vals_b = pad_values_to_power_of_two(sub_b)

    m = len(vals_a)
    n = int(log2(m))

    anc = QuantumRegister(1, "anc")
    qa = QuantumRegister(n, "qa")
    qb = QuantumRegister(n, "qb")
    c = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, qa, qb, c)

    phase_a = build_phase_state_from_values(vals_a, alpha)
    phase_b = build_phase_state_from_values(vals_b, alpha)

    qc.compose(phase_a, qa, inplace=True)
    qc.compose(phase_b, qb, inplace=True)

    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], qa[i], qb[i])
    qc.h(anc[0])

    qc.measure(anc[0], c[0])
    return qc


def run_phase_swap(sub_a, sub_b, alpha: float, shots=2048, backend=None):
    if backend is None:
        backend = AerSimulator()

    qc = build_phase_swap_circuit(sub_a, sub_b, alpha)
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    n0 = counts.get("0", 0)
    p0_hat = n0 / shots
    return p0_hat


def estimate_cosine_group_quantum_phase(sub_a, sub_b, alpha: float,
                                        shots=2048, backend=None):
    """
    Estimador cuántico tipo PES para un grupo (k,l) usando
    codificación de fase continua φ_i = α * valor_discretizado_i.
    """
    sub_a = np.asarray(sub_a, float)
    sub_b = np.asarray(sub_b, float)

    # Grupo trivial de tamaño 1: magnitud 1, signo clásico
    if len(sub_a) == 1:
        sign_classic = np.sign(sub_a[0] * sub_b[0])
        if sign_classic == 0:
            sign_classic = 1.0
        return float(sign_classic)

    na = float(sub_a @ sub_a)
    nb = float(sub_b @ sub_b)
    if na == 0 or nb == 0:
        return 0.0

    p0_hat = run_phase_swap(sub_a, sub_b, alpha, shots=shots, backend=backend)

    val = max(0.0, 2.0 * p0_hat - 1.0)
    overlap_mag = math.sqrt(val)

    sign_classic = np.sign(sub_a @ sub_b)
    if sign_classic == 0:
        sign_classic = 1.0

    return float(sign_classic * overlap_mag)


def run_pes_multiswap_phase(x, y, centers, alpha=np.pi,
                            shots=2048, seed=123, opt_level=3, verbose=False):
    """
    PES-MULTISWAP estratificado:

      1) Discretizar x,y a centros -> qx,qy + índices idx_x, idx_y.
      2) Particionar por grupos (k,l) según (idx_x[i], idx_y[i]).
      3) Calcular cos_g de cada grupo usando SWAP de fase (fases continuas).
      4) Agregar con la identidad estratificada.
    """
    sim = AerSimulator(seed_simulator=seed)

    qx, idx_x = quantize_to_centers(x, centers)
    qy, idx_y = quantize_to_centers(y, centers)

    # Partición en grupos (k,l)
    groups = build_partition_groups_from_indices(idx_x, idx_y, centers)

    # Coseno estratificado clásico (para check)
    cos_classic, cos_g_classic, Na_g, Nb_g = classical_cos_from_groups_quantized(
        qx, qy, groups
    )

    # Coseno estratificado cuántico
    cos_g_quantum = {}
    Na_g_q = {}
    Nb_g_q = {}

    for key, idxs in groups.items():
        idxs = np.asarray(idxs)
        sub_a = qx[idxs]
        sub_b = qy[idxs]

        na = float(sub_a @ sub_a)
        nb = float(sub_b @ sub_b)
        Na_g_q[key] = na
        Nb_g_q[key] = nb

        if na == 0 or nb == 0:
            cos_g_quantum[key] = 0.0
        else:
            cos_g_quantum[key] = estimate_cosine_group_quantum_phase(
                sub_a, sub_b, alpha=alpha, shots=shots, backend=sim
            )

    Na = sum(Na_g_q.values())
    Nb = sum(Nb_g_q.values())
    num = sum(cos_g_quantum[g] * math.sqrt(Na_g_q[g] * Nb_g_q[g]) for g in groups)
    cos_hat = num / math.sqrt(Na * Nb)

    cos_real = cos_sim(x, y)

    if verbose:
        print("\n=== DETALLE GRUPOS (k,l) ===")
        for key in sorted(groups.keys()):
            print(
                f"Grupo {key}: n={len(groups[key]):3d}, "
                f"cl={cos_g_classic[key]: .4f}, "
                f"cu={cos_g_quantum[key]: .4f}"
            )

    mae_ms = abs(cos_real - cos_hat)
    return cos_real, cos_hat, mae_ms, cos_classic


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fusion AE-SWAP vs PES global vs PES-MULTISWAP (fases continuas)."
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=np.pi/4)
    parser.add_argument("--seed0", type=int, default=123)
    args = parser.parse_args()

    dim = args.dim
    K = args.K
    shots = args.shots
    alpha = args.alpha

    rhos = [-0.9, -0.5, 0.0, 0.5, 0.9]

    print("\n=== AE-SWAP vs PES-DATA-DRIVEN vs PES-MULTISWAP (FUSIÓN) ===\n")
    print("rho | cos_real | AE | PES-DD | PES-MS | MAE_AE | MAE_DD | MAE_MS")

    for i, rho in enumerate(rhos):
        seed = args.seed0 + i

        x, y = make_pair_with_cosine(dim, rho, seed)
        centers = learn_kmeans_centers(x, y, K=K, seed=seed)

        # AE-SWAP
        cos_ae = run_ae_swap(x, y, shots=shots, seed=seed)

        # PES global
        cos_real, cos_disc, cos_pes_global, mae_disc, mae_pes_global = \
            run_pes_datadriven_global(x, y, centers, shots=shots, seed=seed)

        # PES-MULTISWAP (nuevo)
        cos_real_ms, cos_pes_ms, mae_ms, cos_classic_strat = \
            run_pes_multiswap_phase(x, y, centers,
                                    alpha=alpha, shots=shots, seed=seed,
                                    verbose=False)

        mae_ae = abs(cos_real - cos_ae)

        print(f"{rho:+.2f} | {cos_real:+.3f} | "
              f"{cos_ae:+.3f} | {cos_pes_global:+.3f} | {cos_pes_ms:+.3f} | "
              f"{mae_ae:.3f} | {mae_pes_global:.3f} | {mae_ms:.3f}")


if __name__ == "__main__":
    main()
