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
from src.utils.discretization import quantize_to_centers, kmeans_to_centers


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
    K = max(idx_x.max(), idx_y.max()) + 1

    # Primero construimos los grupos básicos crudos
    raw_groups = defaultdict(list)
    for t in range(len(idx_x)):
        i = idx_x[t]
        j = idx_y[t]
        raw_groups[(i, j)].append(t)

    # Estructura para grupos fusionados
    fused = defaultdict(list)

    # 1) Procesar primero los off-diagonales
    offdiag_pending = []  # diagonales pendientes de reasignación

    for (i, j), elements in raw_groups.items():
        if i != j:
            # grupo canónico
            key = (min(i, j), max(i, j))
            fused[key].extend(elements)
        else:
            # diagonal (i,i) → hay que reasignarlo luego
            offdiag_pending.append((i, elements))

    # 2) Reasignar los grupos diagonales (i,i)
    for (i, elems) in offdiag_pending:
        # buscar todos los grupos donde aparece i
        candidates = [(key, fused[key]) for key in fused.keys()
                                          if i in key]

        if len(candidates) == 0:
            # no existe ningún grupo relacionado con i
            # creamos un grupo ficticio con algún vecino ficticio
            # pero debe cumplir a<b
            fake_key = (i, i+1)
            fused[fake_key] = []
            candidates = [(fake_key, fused[fake_key])]

        # elegir el grupo con menor tamaño actual
        best_key = min(candidates, key=lambda kv: len(kv[1]))[0]

        # añadir los elementos de la diagonal
        fused[best_key].extend(elems)

    return fused


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
            # 🔧 AQUÍ ESTABA EL BUG: antes poníamos sub_a @ sub_a
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
# Utilidades de fase / encoding complejo
# ============================================================

def next_power_of_two(m: int) -> int:
    if m <= 1:
        return 1
    return 1 << (m - 1).bit_length()


def pad_values_to_power_of_two(values: np.ndarray) -> np.ndarray:
    """
    Rellena el vector de valores (reales o complejos) hasta la siguiente
    potencia de 2. Los elementos nuevos se ponen a 1 (fase identidad).
    """
    values = np.asarray(values)
    L = len(values)
    m = next_power_of_two(L)
    if m == L:
        return values
    pad_len = m - L
    pad = np.ones(pad_len, dtype=values.dtype)  # fase 1+0j para los extras
    return np.concatenate([values, pad])


def real_to_unit_complex(c: float) -> complex:
    """
    Mapea un valor real c a un complejo de módulo 1:

        c ↦ c ± i * sqrt(1 - c^2)

    con el signo imaginario elegido para preservar el signo de c.

    Si |c| > 1, se satura a [-1,1] antes de construir el complejo.
    """
    c = float(c)
    if c > 1.0:
        c = 1.0
    if c < -1.0:
        c = -1.0
    imag = math.sqrt(max(0.0, 1.0 - c * c))
    if c >= 0.0:
        return c + 1j * imag
    else:
        return c - 1j * imag


def values_to_unit_complex(values: np.ndarray) -> np.ndarray:
    """
    Aplica real_to_unit_complex componente a componente.
    """
    values = np.asarray(values, float)
    return np.array([real_to_unit_complex(v) for v in values], dtype=complex)

"""
def build_phase_state_from_values(diag_vals: np.ndarray) -> QuantumCircuit:

    diag_vals = np.asarray(diag_vals, complex)
    m = len(diag_vals)
    n = int(log2(m))
    if 2 ** n != m:
        raise ValueError("build_phase_state_from_values: longitud no potencia de 2")

    # Extraemos la parte real
    real_parts = np.real(diag_vals)

    # Codificamos como e^{i * theta}
    encoded_phases = np.exp(1j * real_parts)

    # Construimos el circuito
    qr = QuantumRegister(n, "data")
    qc = QuantumCircuit(qr, name="phase_state")

    # Estado uniforme
    qc.h(qr)

    # Puerta diagonal con las fases codificadas
    diag_gate = DiagonalGate(encoded_phases.tolist())
    qc.append(diag_gate, qr)

    return qc

"""

def build_phase_state_from_values(diag_vals: np.ndarray) -> QuantumCircuit:
    """
    Construye el estado:

      |ψ⟩ = (1/√m) Σ_i diag_vals[i] |i>

    donde diag_vals[i] son complejos de módulo 1 (fases en el círculo unitario),
    sobre log2(m) qubits.
    """
    diag_vals = np.asarray(diag_vals, complex)
    m = len(diag_vals)
    n = int(log2(m))
    if 2 ** n != m:
        raise ValueError("build_phase_state_from_values: longitud no potencia de 2")

    qr = QuantumRegister(n, "data")
    qc = QuantumCircuit(qr, name="phase_state")

    qc.h(qr)
    diag_gate = DiagonalGate(diag_vals.tolist())
    qc.append(diag_gate, qr)

    return qc


def build_phase_swap_circuit(sub_a, sub_b, alpha: float = None) -> QuantumCircuit:
    """
    SWAP test entre dos estados de fase codificados a partir de sub_a, sub_b,
    donde cada componente real se mapea a un complejo de módulo 1 mediante
    real_to_unit_complex.

    El parámetro alpha se mantiene por compatibilidad, pero NO se usa.
    """
    sub_a = np.asarray(sub_a, float)
    sub_b = np.asarray(sub_b, float)
    if len(sub_a) != len(sub_b):
        raise ValueError("build_phase_swap_circuit: longitudes distintas")

    # 1) Mapeo real -> complejo de módulo 1
    diag_a = values_to_unit_complex(sub_a)
    diag_b = values_to_unit_complex(sub_b)

    # 2) Relleno a potencia de 2 con fase identidad (1+0j)
    vals_a = pad_values_to_power_of_two(diag_a)
    vals_b = pad_values_to_power_of_two(diag_b)

    m = len(vals_a)
    n = int(log2(m))

    anc = QuantumRegister(1, "anc")
    qa = QuantumRegister(n, "qa")
    qb = QuantumRegister(n, "qb")
    c = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, qa, qb, c, name="pes_multiswap_group")

    phase_a = build_phase_state_from_values(vals_a)
    phase_b = build_phase_state_from_values(vals_b)

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
                   alpha: float = None,
                   shots: int = 2048,
                   backend=None) -> float:
    """
    Ejecuta el SWAP de fase para un grupo y devuelve p0_hat.

    alpha se ignora (encoding puramente geométrico en el círculo unitario).
    """
    if backend is None:
        backend = AerSimulator()

    qc = build_phase_swap_circuit(sub_a, sub_b, alpha=alpha)
    tqc = transpile(qc, backend)  # optimization_level por defecto
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts(tqc)

    n0 = counts.get("0", 0)
    return n0 / float(shots)


def estimate_cosine_group_quantum_phase(sub_a,
                                        sub_b,
                                        alpha: float = None,
                                        shots: int = 2048,
                                        backend=None) -> float:
    """
    Estimador cuántico tipo PES para un grupo (k,l) usando el encoding
    complejo en el círculo unitario aplicado a los valores discretizados
    del grupo.

    IMPORTANTE:
      - El módulo del solapamiento |⟨ψ_a|ψ_b⟩| lo da el SWAP (vía p0).
      - El signo lo seguimos metiendo de forma clásica, vía sub_a @ sub_b.
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

    p0_hat = run_phase_swap(sub_a, sub_b, alpha=alpha, shots=shots, backend=backend)
    print(p0_hat)
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

    NOTA: el parámetro alpha se mantiene en la interfaz por compatibilidad,
    pero el encoding actual NO depende de él (no hay α*valor).
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
    #qx, idx_x = kmeans_to_centers(x, len(centers))
    #qy, idx_y = kmeans_to_centers(y, len(centers))
    # Partición en grupos
    groups = build_partition_groups_from_indices(idx_x, idx_y, centers)

    # Coseno clásico estratificado (sobre valores-centro)
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
    # (3) Agregación final estratificada
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
    print(f"Numero de grupos = {len(groups)}")
    if verbose:
        print("\n=== DETALLE GRUPOS (k,l) ===")
        for key in sorted(groups.keys()):
            print(
                f"Grupo {key}: n={len(groups[key]):3d}, "
                f"Valor {groups[key]}"
                f"cl={cos_g_classic[key]: .4f}, "
                f"cu={cos_g_quantum[key]: .4f}"
            )

    return (cos_real, cos_hat, mae_ms, cos_classic,
            t_preproc, t_quantum_pes)
