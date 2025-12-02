# src/quantum/pes_multiswap.py
import numpy as np
import math
import time

from math import log2
from collections import defaultdict

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

from src.utils.basic import cos_sim
from src.utils.discretization import quantize_to_centers


# ============================================================
# Partición en grupos (k,l)
# ============================================================

def build_partition_groups_from_indices(idx_x, idx_y, centers):
    """
    Particiona los índices de coordenadas en grupos etiquetados por (k,l),
    donde k,l son índices de centros (0..K-1).

    Devuelve:
      groups: dict (k,l) -> lista de índices i.
    """
    idx_x = np.asarray(idx_x, int)
    idx_y = np.asarray(idx_y, int)

    groups = defaultdict(list)
    for i in range(len(idx_x)):
        k = idx_x[i]
        l = idx_y[i]
        groups[(k, l)].append(i)

    return groups


def classical_cos_from_groups_quantized(qx, qy, groups):
    """
    Versión clásica estratificada, usando los vectores discretizados qx, qy:

      cos(qx,qy)
      = [ Σ_g cos_g * sqrt(Na_g * Nb_g) ] / sqrt( Σ Na_g * Σ Nb_g )

    donde:
      Na_g = ||qx_g||^2, Nb_g = ||qy_g||^2
      cos_g = cos(qx_g, qy_g) si ambos tienen norma > 0, 0 en otro caso.
    """
    qx = np.asarray(qx, float)
    qy = np.asarray(qy, float)

    Na_g = {}
    Nb_g = {}
    cos_g = {}

    for key, idxs in groups.items():
        idxs = np.asarray(idxs, int)
        sub_a = qx[idxs]
        sub_b = qy[idxs]

        na = float(sub_a @ sub_a)
        nb = float(sub_b @ sub_b)
        Na_g[key] = na
        Nb_g[key] = nb

        if na > 0.0 and nb > 0.0:
            cos_g[key] = float(sub_a @ sub_b) / math.sqrt(na * nb)
        else:
            cos_g[key] = 0.0

    Na = sum(Na_g.values())
    Nb = sum(Nb_g.values())
    if Na == 0.0 or Nb == 0.0:
        return 0.0, cos_g, Na_g, Nb_g

    num = sum(cos_g[g] * math.sqrt(Na_g[g] * Nb_g[g]) for g in groups)
    cos_hat = num / math.sqrt(Na * Nb)
    return cos_hat, cos_g, Na_g, Nb_g


# ============================================================
# Utilidades de fase continua
# ============================================================

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


def build_phase_state_from_values(values: np.ndarray, alpha: float) -> QuantumCircuit:
    """
    Construye el estado:

      |ψ⟩ = (1/√m) Σ_i e^{i * α * values[i]} |i>

    sobre log2(m) qubits.
    """
    values = np.asarray(values, float)
    m = len(values)
    n = int(log2(m))
    if 2 ** n != m:
        raise ValueError("build_phase_state_from_values: longitud no potencia de 2")

    phases = alpha * values
    diag_vals = [np.exp(1j * phi) for phi in phases]

    qr = QuantumRegister(n, "data")
    qc = QuantumCircuit(qr, name="phase_state")

    qc.h(qr)
    diag_gate = DiagonalGate(diag_vals)
    qc.append(diag_gate, qr)

    return qc


def build_phase_swap_circuit(sub_a, sub_b, alpha: float) -> QuantumCircuit:
    """
    SWAP test entre dos estados de fase codificados a partir de
    sub_a, sub_b (valores discretizados del grupo).
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

    qc = QuantumCircuit(anc, qa, qb, c, name="pes_multiswap_group")

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


def run_phase_swap(sub_a,
                   sub_b,
                   alpha: float,
                   shots: int = 2048,
                   backend=None) -> float:
    """
    Ejecuta el SWAP de fase para un grupo y devuelve p0_hat.
    """
    if backend is None:
        backend = AerSimulator()

    qc = build_phase_swap_circuit(sub_a, sub_b, alpha)
    tqc = transpile(qc, backend)  # optimization_level por defecto
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts(tqc)

    n0 = counts.get("0", 0)
    return n0 / float(shots)


def estimate_cosine_group_quantum_phase(sub_a,
                                        sub_b,
                                        alpha: float,
                                        shots: int = 2048,
                                        backend=None) -> float:
    """
    Estimador cuántico tipo PES para un grupo (k,l) usando
    codificación de fase continua φ_i = α * valor_discretizado_i.
    """
    sub_a = np.asarray(sub_a, float)
    sub_b = np.asarray(sub_b, float)

    # Grupo trivial de tamaño 1: usamos signo clásico (magnitud 1)
    if len(sub_a) == 1:
        sign_classic = np.sign(sub_a[0] * sub_b[0])
        if sign_classic == 0.0:
            sign_classic = 1.0
        return float(sign_classic)

    na = float(sub_a @ sub_a)
    nb = float(sub_b @ sub_b)
    if na == 0.0 or nb == 0.0:
        return 0.0

    p0_hat = run_phase_swap(sub_a, sub_b, alpha, shots=shots, backend=backend)
    val = max(0.0, 2.0 * p0_hat - 1.0)
    overlap_mag = math.sqrt(val)

    sign_classic = np.sign(sub_a @ sub_b)
    if sign_classic == 0.0:
        sign_classic = 1.0

    return float(sign_classic * overlap_mag)


# ============================================================
# Función principal PES-MULTISWAP
# ============================================================

def run_pes_multiswap_phase(x,
                            y,
                            centers,
                            alpha: float = math.pi,
                            shots: int = 2048,
                            seed: int = 123,
                            verbose: bool = False):
    """
    Devuelve:
      cos_real, cos_hat, mae_ms, cos_classic,
      t_preproc, t_quantum_pes
    """
    sim = AerSimulator(seed_simulator=seed)

    # -----------------------------
    #  (1) Preprocesado clásico
    # -----------------------------
    t0_pre = time.perf_counter()

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    centers = np.asarray(centers, float)

    # Discretización
    qx, idx_x = quantize_to_centers(x, centers)
    qy, idx_y = quantize_to_centers(y, centers)

    # Partición en grupos
    groups = build_partition_groups_from_indices(idx_x, idx_y, centers)

    # Coseno clásico estratificado
    cos_classic, cos_g_classic, Na_g, Nb_g = classical_cos_from_groups_quantized(
        qx, qy, groups
    )

    t1_pre = time.perf_counter()
    t_preproc = t1_pre - t0_pre

    # -----------------------------
    # (2) Tiempo cuántico por grupos
    # -----------------------------
    t_quantum_pes = 0.0

    cos_g_quantum = {}
    Na_g_q = {}
    Nb_g_q = {}

    for key, idxs in groups.items():
        idxs = np.asarray(idxs, int)
        sub_a = qx[idxs]
        sub_b = qy[idxs]

        na = float(sub_a @ sub_a)
        nb = float(sub_b @ sub_b)
        Na_g_q[key] = na
        Nb_g_q[key] = nb

        if na == 0.0 or nb == 0.0:
            cos_g_quantum[key] = 0.0
            continue

        # Tiempo cuántico del grupo
        tg0 = time.perf_counter()
        val = estimate_cosine_group_quantum_phase(
            sub_a, sub_b, alpha=alpha, shots=shots, backend=sim
        )
        tg1 = time.perf_counter()

        t_quantum_pes += (tg1 - tg0)
        cos_g_quantum[key] = val

    # -----------------------------
    # (3) Agregación final
    # -----------------------------
    Na = sum(Na_g_q.values())
    Nb = sum(Nb_g_q.values())
    if Na == 0.0 or Nb == 0.0:
        cos_hat = 0.0
    else:
        num = sum(cos_g_quantum[g] * math.sqrt(Na_g_q[g] * Nb_g_q[g])
                  for g in groups)
        cos_hat = num / math.sqrt(Na * Nb)

    cos_real = cos_sim(x, y)
    mae_ms = abs(cos_real - cos_hat)

    if verbose:
        print("\n=== DETALLE GRUPOS (k,l) ===")
        for key in sorted(groups.keys()):
            print(
                f"Grupo {key}: n={len(groups[key]):3d}, "
                f"cl={cos_g_classic[key]: .4f}, "
                f"cu={cos_g_quantum[key]: .4f}"
            )

    return (cos_real, cos_hat, mae_ms, cos_classic,
            t_preproc, t_quantum_pes)
