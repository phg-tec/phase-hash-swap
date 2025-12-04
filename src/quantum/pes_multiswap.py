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
from sklearn.cluster import KMeans


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


def build_groups_kmeans_2d(qx, qy, n_groups: int, random_state: int = 123):
    """
    Agrupa las coordenadas por K-Means en el plano (qx, qy).

    - qx, qy: vectores discretizados (mismas longitudes)
    - n_groups: número de clusters (grupos) deseados
    - random_state: semilla para reproducibilidad

    Devuelve:
      groups: dict g -> lista de índices i pertenecientes al cluster g
    """
    qx = np.asarray(qx, float)
    qy = np.asarray(qy, float)
    assert len(qx) == len(qy)
    d = len(qx)

    # No tiene sentido tener más grupos que dimensiones
    n_groups = min(n_groups, d)
    if n_groups <= 0:
        # caso degenerado, todo en un único grupo
        return {0: list(range(d))}

    Z = np.stack([qx, qy], axis=1)  # shape (d, 2)

    km = KMeans(n_clusters=n_groups, n_init=10, random_state=random_state)
    labels = km.fit_predict(Z)
    centers_2d = km.cluster_centers_

    groups = defaultdict(list)
    for idx, g in enumerate(labels):
        groups[int(g)].append(idx)

    # Opcional: fusionar clusters de tamaño 1 con el centro más cercano
    for g, idxs in list(groups.items()):
        if len(idxs) == 1 and len(groups) > 1:
            idx = idxs[0]
            v = Z[idx]
            dists = np.linalg.norm(centers_2d - v, axis=1)
            dists[g] = np.inf  # no se reasigna al mismo cluster
            g2 = int(np.argmin(dists))
            groups[g2].append(idx)
            del groups[g]

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
        return c - 1j * imag
    else:
        return c + 1j * imag


def values_to_unit_complex(values: np.ndarray) -> np.ndarray:
    """
    Aplica real_to_unit_complex componente a componente.
    """
    values = np.asarray(values, float)
    return np.array([real_to_unit_complex(v) for v in values], dtype=complex)


def build_phase_state_from_values(diag_vals: np.ndarray) -> QuantumCircuit:

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


    na = float(sub_a @ sub_a)
    nb = float(sub_b @ sub_b)
    if na == 0.0 or nb == 0.0:
        return 0.0

    p0_hat = run_phase_swap(sub_a, sub_b, alpha=alpha, shots=shots, backend=backend)
    val = max(0.0, 2.0 * p0_hat - 1.0)
    overlap_mag = math.sqrt(val)

    sign_classic = np.sign(sub_a @ sub_b)
    if sign_classic == 0.0:
        sign_classic = 1.0

    return float(sign_classic * overlap_mag), p0_hat

def merge_singleton_groups(groups):
    """
    Fusión mínima y correcta para PES-MULTISWAP:
    Solo fusiona grupos de tamaño 1 con OTROS GRUPOS QUE COMPARTEN
    EXACTAMENTE EL MISMO PAR (k,l).

    Si no existe ningún otro grupo (k,l), se fusiona con el grupo más pequeño existente,
    pero esto solo ocurre en casos degenerados y muy raros.
    """

    # Copia limpia del diccionario
    groups = {key: list(idxs) for key, idxs in groups.items()}

    while True:
        # Detectar singletons
        singletons = [key for key, idxs in groups.items() if len(idxs) == 1]
        if not singletons:
            break

        key = singletons[0]
        k, l = key
        lone_val = groups[key][0]

        # Buscar OTROS grupos exactamente con la misma clave (k,l)
        candidates = [other for other in groups.keys()
                      if other != key and other == key]

        # En la práctica lo anterior puede ser vacío, porque los grupos (k,l)
        # están definidos de forma única.
        # Entonces, buscamos grupos que representen el MISMO par (k,l)
        # en casos donde se haya fusionado previamente (por robustez):
        candidates = [other for other in groups.keys()
                      if other != key and other[0] == k and other[1] == l]

        # Si no existe ninguno, caso degenerado:
        # fusionar con el grupo más pequeño disponible.
        if not candidates:
            candidates = [other for other in groups.keys() if other != key]

        if not candidates:
            # solo quedaba 1 grupo, no se puede hacer nada
            break

        # Elegir el más pequeño para mantener equilibrio
        target = min(candidates, key=lambda g: len(groups[g]))

        # mover valor
        groups[target].append(lone_val)
        del groups[key]

    return groups



# ============================================================
# Función principal PES-MULTISWAP
# ============================================================

def run_pes_multiswap_phase(
        x, y, centers,
        alpha: float = math.pi,
        shots: int = 2048,
        seed: int = 123,
        verbose: bool = False):

    sim = AerSimulator(seed_simulator=seed)

    # -----------------------------
    # (1) Preprocesado clásico
    # -----------------------------
    t0_pre = time.perf_counter()

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    centers = np.asarray(centers, float)


    maxabs = max(np.max(np.abs(x)), np.max(np.abs(y)))
    if maxabs == 0:
        maxabs = 1.0
    x = x / maxabs
    y = y / maxabs

    # Discretización para elegir grupos (pero no para el SWAP)
    qx, idx_x = quantize_to_centers(x, centers)
    qy, idx_y = quantize_to_centers(y, centers)

    groups = build_partition_groups_from_indices(idx_x, idx_y, centers)
    groups = merge_singleton_groups(groups)
    n_single = sum(1 for g in groups.values() if len(g) == 1)
    print(f"Grupos tamaño 1 tras Singleton = {n_single}")

    # Clásico estratificado usando los valores reales
    cos_classic, cos_g_classic, Na_g, Nb_g = classical_cos_from_groups_quantized(
        x, y, groups
    )

    t_preproc = time.perf_counter() - t0_pre

    # -----------------------------
    # (2) Evaluación cuántica por grupos
    # -----------------------------
    t_quantum_pes = 0.0

    cos_g_quantum = {}
    Na_g_q = {}
    Nb_g_q = {}
    p0_g_list = []

    for key, idxs in groups.items():
        idxs = np.asarray(idxs, int)

        # Usar valores reales
        sub_a = x[idxs]
        sub_b = y[idxs]

        na = float(sub_a @ sub_a)
        nb = float(sub_b @ sub_b)


        Na_g_q[key] = na
        Nb_g_q[key] = nb

        if na == 0.0 or nb == 0.0:
            cos_g_quantum[key] = 0.0
            p0_g_list.append(0.5)
            continue

        t0 = time.perf_counter()
        cos_val, p0_hat = estimate_cosine_group_quantum_phase(
            sub_a, sub_b,
            alpha=alpha,
            shots=shots,
            backend=sim
        )
        t_quantum_pes += time.perf_counter() - t0

        cos_g_quantum[key] = cos_val
        p0_g_list.append(p0_hat)

    # -----------------------------
    # (3) Agregación del coseno global
    # -----------------------------
    Na = sum(Na_g_q.values())
    Nb = sum(Nb_g_q.values())

    if Na == 0.0 or Nb == 0.0:
        cos_quantum_global = 0.0
    else:
        numer = sum(
            cos_g_quantum[g] * math.sqrt(Na_g_q[g] * Nb_g_q[g])
            for g in groups
        )
        cos_quantum_global = numer / math.sqrt(Na * Nb)

    # -----------------------------
    # (4) Cálculo del p0_global desde M (único correcto)
    # -----------------------------
    if Na == 0.0 or Nb == 0.0:
        p0_global = 0.5
    else:
        numer_M = sum(
            math.sqrt(max(0.0, 2.0 * p0g - 1.0)) * math.sqrt(Na_g_q[key] * Nb_g_q[key])
            for (key, p0g) in zip(groups.keys(), p0_g_list)
        )
        M = numer_M / math.sqrt(Na * Nb)
        p0_global = (1.0 + M * M) / 2.0

    # -----------------------------
    # (5) Comparación por grupos: p0_real_g vs p0_quant_g
    # -----------------------------
    p0_real_g = {}
    p0_quant_g = {}

    for (g, idxs), p0g in zip(groups.items(), p0_g_list):
        sub_a = x[idxs]
        sub_b = y[idxs]

        na = float(sub_a @ sub_a)
        nb = float(sub_b @ sub_b)

        if na == 0.0 or nb == 0.0:
            p0_real_g[g] = 0.5
        else:
            cosg = float(sub_a @ sub_b) / math.sqrt(na * nb)
            p0_real_g[g] = (1.0 + cosg * cosg) / 2.0

        p0_quant_g[g] = p0g

    cos_quantum_global = math.sqrt(max(0, 2*p0_global-1))
    # -----------------------------
    # (6) Resultados finales
    # -----------------------------
    cos_real_global = cos_sim(x, y)
    cos_mae = abs(cos_real_global - cos_quantum_global)

    return {
        "cos_real": cos_real_global,
        "cos_quantum": cos_quantum_global,
        "cos_mae": cos_mae,
        "p0_real_g": p0_real_g,
        "p0_quant_g": p0_quant_g,
        "p0_global": p0_global,
        "t_pre": t_preproc,
        "t_pes": t_quantum_pes,
        "groups": groups
    }
