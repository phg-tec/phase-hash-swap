# ============================================================
# AE-SWAP vs PES data-driven vs PES multi-level (residual global)
# SIN HASHING — K-means aplicado directamente sobre x,y
# ============================================================

import numpy as np
import math
from sklearn.cluster import KMeans
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import DiagonalGate


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
    theta = (math.pi/2) * (1 - c)
    return math.cos(theta)


# ============================================================
# AE-SWAP
# ============================================================

def build_amp_state(vec):
    vec = np.asarray(vec, float)
    n = int(np.ceil(np.log2(len(vec))))
    pad_len = 2**n
    amp = np.zeros(pad_len, complex)
    amp[:len(vec)] = vec/np.linalg.norm(vec)

    from qiskit.circuit.library import Initialize
    qr = QuantumRegister(n, "amp")
    qc = QuantumCircuit(qr)
    qc.append(Initialize(amp), qr)
    return qc


def build_swap_test(prepA, prepB):
    n = prepA.num_qubits
    anc = QuantumRegister(1, "anc")
    qa  = QuantumRegister(n, "a")
    qb  = QuantumRegister(n, "b")
    c   = ClassicalRegister(1, "m")

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

    tqc = transpile(qc, sim, optimization_level=opt_level,
                    seed_transpiler=seed)
    res = sim.run(tqc, shots=shots).result()

    p0 = p0_from_counts(res.get_counts(tqc))
    return corr_to_cos(corr_abs_from_p0(p0))


# ============================================================
# K-means discretization DIRECTAMENTE SOBRE x,y
# ============================================================

def learn_kmeans_centers(x, y, K=5, seed=0):
    """Aprende K valores representativos sobre todos los componentes de x e y."""
    data = np.concatenate([x, y]).reshape(-1, 1)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    km.fit(data)
    return np.sort(km.cluster_centers_.flatten())


def quantize_to_centers(v, centers):
    """Asigna cada componente al centro K-means más cercano."""
    v = np.asarray(v)
    centers = np.asarray(centers)
    d = np.abs(v[:, None] - centers[None, :])
    idx = np.argmin(d, axis=1)
    return centers[idx], idx


def idx_to_phases(idx, K):
    """Convierte el índice del nivel a fase uniforme [0,2π)."""
    return 2 * np.pi * idx / K


# ============================================================
# PES normal (una sola DiagonalGate, solo niveles)
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


def run_pes_datadriven(x, y, centers, shots=2048, seed=123, opt_level=3):
    """
    PES multinivel “naive”:
      - discretiza x,y a centros K-means
      - asigna una fase uniforme φ_k a cada nivel
      - un único SWAP test
    """
    sim = AerSimulator(seed_simulator=seed)

    qx, idx_x = quantize_to_centers(x, centers)
    qy, idx_y = quantize_to_centers(y, centers)

    cos_disc = cos_sim(qx, qy)

    K = len(centers)
    phix = idx_to_phases(idx_x, K)
    phiy = idx_to_phases(idx_y, K)

    px = build_phase_state(phix)
    py = build_phase_state(phiy)
    qc = build_swap_test(px, py)

    tqc = transpile(qc, sim, optimization_level=opt_level,
                    seed_transpiler=seed)
    res = sim.run(tqc, shots=shots).result()

    cos_hat = corr_to_cos(corr_abs_from_p0(p0_from_counts(res.get_counts(tqc))))
    cos_real = cos_sim(x, y)

    return cos_real, cos_disc, cos_hat, abs(cos_real - cos_disc), abs(cos_real - cos_hat)


# ============================================================
# PES multi-level con residual global (φ_k + α·residual)
# ============================================================

def run_pes_multilevel_residual_global(x, y, K=4,
                                       shots=4096, seed=123, opt_level=3):
    """
    PES multinivel “bueno”:

    - Aprende K centros por K-means sobre (x,y)
    - Cada componente i:
         se le asigna un centro c_k(i)
         residual r_i = x_i - c_k(i), s_i = y_i - c_k(i) para y
    - Fase total:
         φ_x(i) = φ_center(k(i)) + α * r_i
         φ_y(i) = φ_center(k(i)) + α * s_i

      donde:
        φ_center(k) se obtiene de los centros normalizados a [-π/2, π/2]
        α es un factor GLOBAL que normaliza los residuos a [-π/2, π/2]

    - Se construyen dos estados de fase y se ejecuta un único SWAP test.
    """
    sim = AerSimulator(seed_simulator=seed)

    # 1) Centros globales
    centers = learn_kmeans_centers(x, y, K=K, seed=seed)

    # 2) Discretización a centros (valores + índices)
    qx, idx_x = quantize_to_centers(x, centers)
    qy, idx_y = quantize_to_centers(y, centers)

    # 3) Fases de los centros φ_k (mapeamos centros a [-π/2, π/2])
    c = centers.copy()
    c_mean = np.mean(c)
    c_shift = c - c_mean
    max_abs_c = np.max(np.abs(c_shift))
    if max_abs_c < 1e-12:
        max_abs_c = 1.0
    c_norm = c_shift / max_abs_c       # en [-1,1]
    phi_centers = (np.pi / 2.0) * c_norm   # en [-π/2, π/2]

    # 4) Residuales reales
    centers_x = centers[idx_x]   # mismo centro para x e y por índice
    centers_y = centers[idx_y]

    residuals_x = x - centers_x
    residuals_y = y - centers_y

    max_res = max(np.max(np.abs(residuals_x)),
                  np.max(np.abs(residuals_y)))
    if max_res < 1e-12:
        alpha = 0.0
    else:
        # Escalamos los residuos para que ocupen también ~[-π/2, π/2]
        alpha = (np.pi / 2.0) / max_res

    # 5) Fase total = fase del centro + fase del residual
    phases_x = phi_centers[idx_x] + alpha * residuals_x
    phases_y = phi_centers[idx_y] + alpha * residuals_y

    # 6) Construimos estados y SWAP test
    px = build_phase_state(phases_x)
    py = build_phase_state(phases_y)
    qc = build_swap_test(px, py)

    tqc = transpile(qc, sim, optimization_level=opt_level,
                    seed_transpiler=seed)
    res = sim.run(tqc, shots=shots).result()

    p0 = p0_from_counts(res.get_counts(tqc))
    cos_hat = corr_to_cos(corr_abs_from_p0(p0))

    cos_real = cos_sim(x, y)
    mae = abs(cos_real - cos_hat)

    return cos_real, cos_hat, mae


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":
    dim   = 256
    K     = 4
    shots = 4096

    rhos = [-0.9, -0.5, 0, 0.5, 0.9]

    print("\n=== AE-SWAP vs PES-DATA-DRIVEN vs PES-MULTI (RESIDUAL GLOBAL) ===\n")
    print("rho | cos_real | AE | PES-DD | PES-RES | MAE_AE | MAE_DD | MAE_RES")

    for i, rho in enumerate(rhos):
        seed = 1000 + i

        x, y = make_pair_with_cosine(dim, rho, seed)
        centers = learn_kmeans_centers(x, y, K=K, seed=seed)

        # AE-SWAP
        cos_ae = run_ae_swap(x, y, shots=shots, seed=seed)

        # PES multinivel naive
        cos_real, cos_disc, cos_pes, mae_disc, mae_pes = run_pes_datadriven(
            x, y, centers, shots=shots, seed=seed
        )

        # PES multinivel con residual global
        _, cos_res, mae_res = run_pes_multilevel_residual_global(
            x, y, K=K, shots=shots, seed=seed
        )

        mae_ae = abs(cos_real - cos_ae)

        print(f"{rho:+.2f} | {cos_real:+.3f} | {cos_ae:+.3f} | "
              f"{cos_pes:+.3f} | {cos_res:+.3f} | "
              f"{mae_ae:.3f} | {mae_pes:.3f} | {mae_res:.3f}")
#!/usr/bin/env python
"""
PES-MULTISWAP (versión fases continuas)
---------------------------------------

Prueba de concepto:

  - Generar dos vectores con un coseno dado.
  - Discretizarlos con k-medias en k centros.
  - Particionarlos en grupos (p,q) SIN solapamiento.
  - Calcular el coseno global:
      * modo clásico estratificado (identidad exacta).
      * modo "cuántico" estratificado: PES-MULTISWAP por grupos,
        codificando las FASES como φ_i = α * valor_discreto_i.

  - Comparar:
      cos_continuo, cos_discretizado, cos_clásico_estratificado,
      cos_cuántico_PES-MULTISWAP.
"""

import numpy as np
from math import log2
from collections import defaultdict
from sklearn.cluster import KMeans

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


# ============================================================
# 1) GENERAR VECTORES CON COSENO OBJETIVO
# ============================================================

def generate_pair_with_cosine(d, cos_target, rng=None):
    """
    Genera x, y en R^d con cos(x,y) ~= cos_target.
    """
    if rng is None:
        rng = np.random.default_rng()

    u = rng.normal(size=d)
    u /= np.linalg.norm(u)

    v = rng.normal(size=d)
    v -= u * np.dot(u, v)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return generate_pair_with_cosine(d, cos_target, rng)
    v /= v_norm

    sin_t = np.sqrt(max(0.0, 1.0 - cos_target**2))
    x = u
    y = cos_target * u + sin_t * v
    return x, y


# ============================================================
# 2) DISCRETIZACIÓN K-MEDIAS
# ============================================================

def discretize_with_kmeans(a, b, k, random_state=0):
    """
    Ajusta un k-medias 1D sobre [a,b] concatenados y
    devuelve los vectores discretizados + centros.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = len(a)

    vals = np.concatenate([a, b])[:, None]

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(vals)

    centers = km.cluster_centers_.flatten()
    labels = km.labels_

    a_disc = centers[labels[:d]]
    b_disc = centers[labels[d:]]

    return a_disc, b_disc, centers


# ============================================================
# 3) PARTICIÓN EXACTA EN GRUPOS (p,q)
# ============================================================

def build_partition_groups(a_disc, b_disc, centers):
    """
    Particiona las coordenadas en grupos etiquetados por (p,q),
    donde p,q son índices de centros (0..k-1).

    Para cada i:
        p = idx(centro asignado a a_disc[i])
        q = idx(centro asignado a b_disc[i])
        agrupar por (min(p,q), max(p,q)).
    """
    a_disc = np.asarray(a_disc)
    b_disc = np.asarray(b_disc)

    centers_sorted = sorted(list(centers))
    center_to_idx = {c: i for i, c in enumerate(centers_sorted)}

    groups = defaultdict(list)

    for i in range(len(a_disc)):
        p = center_to_idx[a_disc[i]]
        q = center_to_idx[b_disc[i]]
        if p <= q:
            key = (p, q)
        else:
            key = (q, p)
        groups[key].append(i)

    return groups, centers_sorted, center_to_idx


# ============================================================
# 4) IDENTIDAD CLÁSICA POR GRUPOS (EXACTA)
# ============================================================

def classical_cos_from_groups(a_disc, b_disc, groups):
    """
    Implementa la identidad:

      cos(a,b)
      = [ Σ_{(p,q)} cos_{p,q} * sqrt(Na_{p,q} * Nb_{p,q}) ]
        / sqrt( (Σ Na_{p,q}) (Σ Nb_{p,q}) )

    con grupos (p,q) que forman una partición de las coordenadas.
    """
    a_disc = np.asarray(a_disc, float)
    b_disc = np.asarray(b_disc, float)

    Na_g = {}
    Nb_g = {}
    cos_g = {}

    for key, idxs in groups.items():
        idxs = np.asarray(idxs)
        sub_a = a_disc[idxs]
        sub_b = b_disc[idxs]

        na = float(sub_a @ sub_a)
        nb = float(sub_b @ sub_b)
        Na_g[key] = na
        Nb_g[key] = nb

        if na > 0 and nb > 0:
            cos_g[key] = float(sub_a @ sub_b) / np.sqrt(na * nb)
        else:
            cos_g[key] = 0.0

    Na = sum(Na_g.values())
    Nb = sum(Nb_g.values())

    num = sum(cos_g[key] * np.sqrt(Na_g[key] * Nb_g[key]) for key in groups)

    cos_hat = num / np.sqrt(Na * Nb)
    return cos_hat, cos_g, Na_g, Nb_g


# ============================================================
# 5) UTILIDADES PARA ESTADO DE FASE (FASES CONTINUAS)
# ============================================================

def next_power_of_two(m: int) -> int:
    """Menor potencia de 2 >= m (m>=1)."""
    if m <= 1:
        return 1
    return 1 << (m - 1).bit_length()


def pad_values_to_power_of_two(values: np.ndarray) -> np.ndarray:
    """
    Rellena el vector `values` con ceros hasta longitud potencia de 2.
    Esto introduce componentes adicionales con fase 0.
    """
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
    Construye el estado de fase:

      |ψ> = (1/√m) Σ_i e^{i * α * values[i]} |i>

    usando:
       - Hadamards para superposición uniforme,
       - DiagonalGate con fases e^{i φ_i}.
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

    # Superposición uniforme
    qc.h(qr)
    # Diagonal de fases
    diag_gate = DiagonalGate(diag_vals)
    qc.append(diag_gate, qr)

    return qc


def build_phase_swap_circuit(sub_a, sub_b, alpha: float):
    """
    Construye un SWAP test entre dos estados de fase codificados
    a partir de sub_a, sub_b (valores discretos del grupo).
    """
    sub_a = np.asarray(sub_a, float)
    sub_b = np.asarray(sub_b, float)
    if len(sub_a) != len(sub_b):
        raise ValueError("build_phase_swap_circuit: longitudes distintas en el grupo")

    # Aseguramos potencia de 2 con padding a 0.0
    vals_a = pad_values_to_power_of_two(sub_a)
    vals_b = pad_values_to_power_of_two(sub_b)

    m = len(vals_a)
    n = int(log2(m))

    anc = QuantumRegister(1, "anc")
    qa = QuantumRegister(n, "qa")
    qb = QuantumRegister(n, "qb")
    c = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, qa, qb, c)

    # Construir estados de fase en qa y qb
    phase_a = build_phase_state_from_values(vals_a, alpha)
    phase_b = build_phase_state_from_values(vals_b, alpha)

    qc.compose(phase_a, qa, inplace=True)
    qc.compose(phase_b, qb, inplace=True)

    # SWAP test estándar
    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], qa[i], qb[i])
    qc.h(anc[0])

    qc.measure(anc[0], c[0])

    return qc


def run_phase_swap(sub_a, sub_b, alpha: float, shots=2048, backend=None):
    """
    Ejecuta el SWAP test sobre los estados de fase generados a partir
    de los valores discretos del grupo.
    Devuelve p0_hat (frecuencia de medir ancilla en |0>).
    """
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
    Estimador cuántico tipo PES para un grupo (p,q) usando
    codificación de fase continua con φ_i = α * valor.
    """
    sub_a = np.asarray(sub_a, float)
    sub_b = np.asarray(sub_b, float)

    # 0) CASO TRIVIAL: grupo de 1 elemento
    if len(sub_a) == 1:
        # Magnitud SIEMPRE = 1 (las dos fases están en un estado 1D)
        sign_classic = np.sign(sub_a[0] * sub_b[0])
        if sign_classic == 0:
            sign_classic = 1.0
        return float(sign_classic)

    na = float(sub_a @ sub_a)
    nb = float(sub_b @ sub_b)
    if na == 0 or nb == 0:
        return 0.0

    # 1) Ejecutar SWAP-test con codificación de fase
    p0_hat = run_phase_swap(sub_a, sub_b, alpha, shots=shots, backend=backend)

    # 2) Módulo del solapamiento
    val = max(0.0, 2.0 * p0_hat - 1.0)
    overlap_mag = np.sqrt(val)

    # 3) Signo aproximado utilizando producto clásico
    sign_classic = np.sign(sub_a @ sub_b)
    if sign_classic == 0:
        sign_classic = 1.0

    return float(sign_classic * overlap_mag)



# ============================================================
# 6) ESTIMACIÓN ESTRATIFICADA CUÁNTICA (MULTISWAP)
# ============================================================

def stratified_cosine_estimator_quantum(a_disc, b_disc, groups,
                                        alpha: float,
                                        shots=2048, backend=None):
    """
    Estimador estratificado del coseno usando SWAP de fase
    por grupos (p,q):

      cos_hat = [ Σ cos_g * sqrt(Na_g * Nb_g) ] / sqrt(ΣNa_g ΣNb_g)

    donde cos_g se estima cuánticamente con estimate_cosine_group_quantum_phase.
    """
    a_disc = np.asarray(a_disc, float)
    b_disc = np.asarray(b_disc, float)

    cos_g = {}
    Na_g = {}
    Nb_g = {}

    for key, idxs in groups.items():
        idxs = np.asarray(idxs)
        sub_a = a_disc[idxs]
        sub_b = b_disc[idxs]

        na = float(sub_a @ sub_a)
        nb = float(sub_b @ sub_b)
        Na_g[key] = na
        Nb_g[key] = nb

        if na == 0 or nb == 0:
            cos_g[key] = 0.0
        else:
            cos_g[key] = estimate_cosine_group_quantum_phase(
                sub_a, sub_b, alpha=alpha, shots=shots, backend=backend
            )

    Na = sum(Na_g.values())
    Nb = sum(Nb_g.values())

    num = sum(cos_g[k] * np.sqrt(Na_g[k] * Nb_g[k]) for k in groups)

    cos_hat = num / np.sqrt(Na * Nb)
    return cos_hat, cos_g, Na_g, Nb_g


# ============================================================
# 7) MAIN
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PES-MULTISWAP con fases continuas (φ = α * valor_discreto)."
    )
    parser.add_argument("--dim", type=int, default=256, help="Dimensión de los vectores.")
    parser.add_argument("--k", type=int, default=4, help="Número de centros k.")
    parser.add_argument("--shots", type=int, default=2048, help="Shots por grupo.")
    parser.add_argument("--alpha", type=float, default=np.pi,
                        help="Escala de fase α, por defecto π.")
    parser.add_argument("--seed", type=int, default=1234, help="Semilla RNG.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cos_list =  [-0.9, -0.5, 0.0, 0.5, 0.9]
    for cos in cos_list:
        # 1) Vectores continuos
        x, y = generate_pair_with_cosine(args.dim, cos, rng)
        cos_cont = float(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

        print("=== VECTORES CONTINUOS ===")
        print(f"Coseno objetivo:          {cos:.6f}")
        print(f"Coseno continuo generado: {cos_cont:.6f}")

        # 2) Discretización
        a_disc, b_disc, centers = discretize_with_kmeans(x, y, args.k, args.seed)
        cos_disc_direct = float(a_disc @ b_disc) / (
            np.linalg.norm(a_disc) * np.linalg.norm(b_disc)
        )

        print("\n=== DISCRETIZACIÓN K-MEDIAS ===")
        print(f"Centros: {np.round(centers, 4)}")
        print(f"Coseno discretizado directo: {cos_disc_direct:.6f}")

        # 3) Grupos (p,q)
        groups, centers_sorted, center_to_idx = build_partition_groups(a_disc, b_disc, centers)

        # 4) Modo clásico estratificado (exacto)
        cos_classic, cos_g_classic, Na_g, Nb_g = classical_cos_from_groups(
            a_disc, b_disc, groups
        )
        print("\n=== MODO CLÁSICO (IDENTIDAD ESTRATIFICADA) ===")
        print(f"Coseno estratificado exacto: {cos_classic:.6f}")

        # 5) Modo cuántico estratificado (PES-MULTISWAP con fases)
        backend = AerSimulator()
        cos_quantum, cos_g_quantum, _, _ = stratified_cosine_estimator_quantum(
            a_disc, b_disc, groups, alpha=args.alpha, shots=args.shots, backend=backend
        )

        print("\n=== MODO CUÁNTICO (PES-MULTISWAP FASES) ===")
        print(f"alpha = {args.alpha:.4f}")
        print(f"Coseno (PES-MULTISWAP fases): {cos_quantum:.6f}")

        print("\n=== DETALLE GRUPOS (p,q) ===")
        for key in sorted(groups.keys()):
            print(
                f"Grupo {key}: n={len(groups[key]):3d}, "
                f"cl={cos_g_classic[key]: .4f}, "
                f"cu={cos_g_quantum[key]: .4f}"
            )


if __name__ == "__main__":
    main()
