# ============================================================
# Script completo: AE-SWAP vs PES data-driven vs PES mini-SWAP
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
    z -= x*np.dot(x,z)
    z /= np.linalg.norm(z)

    y = rho*x + math.sqrt(max(0,1-rho*rho))*z
    return x, y


def cos_sim(u, v):
    return float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))


def p0_from_counts(counts):
    shots = sum(counts.values())
    return counts.get("0",0)/max(1,shots)


def corr_abs_from_p0(p0):
    return math.sqrt(max(0.0, 2*p0 - 1.0))


def corr_to_cos(corr):
    c = max(-1,min(1,float(corr)))
    theta = (math.pi/2)*(1-c)
    return math.cos(theta)


# ============================================================
# AE-SWAP
# ============================================================

def build_amp_state(vec):
    vec = np.asarray(vec,float)
    n = int(np.ceil(np.log2(len(vec))))
    pad_len = 2**n
    amp = np.zeros(pad_len,complex)
    amp[:len(vec)] = vec/np.linalg.norm(vec)

    from qiskit.circuit.library import Initialize
    qr = QuantumRegister(n,"amp")
    qc = QuantumCircuit(qr)
    qc.append(Initialize(amp), qr)
    return qc


def build_swap_test(prepA, prepB):
    n = prepA.num_qubits
    anc = QuantumRegister(1,"anc")
    qa  = QuantumRegister(n,"a")
    qb  = QuantumRegister(n,"b")
    c   = ClassicalRegister(1,"m")

    qc = QuantumCircuit(anc,qa,qb,c)
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
    data = np.concatenate([x, y]).reshape(-1,1)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    km.fit(data)
    return np.sort(km.cluster_centers_.flatten())


def quantize_to_centers(v, centers):
    """Asigna cada componente al centro más cercano."""
    v = np.asarray(v)
    centers = np.asarray(centers)
    d = np.abs(v[:,None] - centers[None,:])
    idx = np.argmin(d,axis=1)
    return centers[idx], idx


def idx_to_phases(idx, K):
    """Convierte el índice del nivel a fase uniforme [0,2π)."""
    return 2*np.pi * idx / K


# ============================================================
# PES normal (una sola DiagonalGate)
# ============================================================

def build_phase_state(phases):
    phases = np.asarray(phases,float)
    m = len(phases)
    n = int(np.log2(m))
    if 2**n != m:
        # pad hasta potencia de 2
        new_m = 1
        while new_m < m:
            new_m*=2
        pad = np.zeros(new_m)
        pad[:m] = phases
        phases = pad
        m = new_m
        n = int(np.log2(m))

    q = QuantumRegister(n,"data")
    qc = QuantumCircuit(q)
    qc.h(q)
    diag = np.exp(1j * phases)
    qc.append(DiagonalGate(diag), q)
    return qc


def run_pes_datadriven(x, y, centers, shots=2048, seed=123, opt_level=3):
    sim = AerSimulator(seed_simulator=seed)

    qx, idx_x = quantize_to_centers(x, centers)
    qy, idx_y = quantize_to_centers(y, centers)

    cos_disc = cos_sim(qx, qy)

    K = len(centers)
    phix = idx_to_phases(idx_x, K)
    phiy = idx_to_phases(idx_y, K)

    px = build_phase_state(phix)
    py = build_phase_state(phiy)
    qc = build_swap_test(px,py)

    tqc = transpile(qc, sim, optimization_level=opt_level, seed_transpiler=seed)
    res = sim.run(tqc, shots=shots).result()

    cos_hat = corr_to_cos(corr_abs_from_p0(p0_from_counts(res.get_counts(tqc))))
    cos_real = cos_sim(x,y)

    return cos_real, cos_disc, cos_hat, abs(cos_real - cos_disc), abs(cos_real - cos_hat)


# ============================================================
# MINI-SWAP TESTS (sin hashing)
# ============================================================

def get_all_binary_groups(idx_x, idx_y, centers):
    """
    Dados dos vectores discretizados idx_x, idx_y (en índices de centros),
    devuelve TODOS los grupos posibles (k,l) y los minivectores asociados.

    Retorna un diccionario:
        grupos[(k,l)] = {
            "a_vals": valores reales del centro_k repetidos,
            "b_vals": valores reales del centro_l repetidos,
            "idx_list": posiciones originales pertenecientes al grupo
        }
    """

    idx_x = np.asarray(idx_x, int)
    idx_y = np.asarray(idx_y, int)
    K = len(centers)

    grupos = {}

    # Inicializamos todos los pares posibles (k,l)
    for k in range(K):
        for l in range(K):
            grupos[(k, l)] = {
                "a_vals": [],
                "b_vals": [],
                "idx_list": []
            }

    # Clasificamos los elementos reales en cada grupo (k,l)
    for i in range(len(idx_x)):
        k = idx_x[i]
        l = idx_y[i]
        grupos[(k, l)]["a_vals"].append(centers[k])
        grupos[(k, l)]["b_vals"].append(centers[l])
        grupos[(k, l)]["idx_list"].append(i)

    # Convertimos a arrays
    for key in grupos:
        grupos[key]["a_vals"] = np.array(grupos[key]["a_vals"], float)
        grupos[key]["b_vals"] = np.array(grupos[key]["b_vals"], float)
        grupos[key]["idx_list"] = np.array(grupos[key]["idx_list"], int)

    return grupos


def build_sub_phase_state(size, phase_index, K):
    if size == 1:
        return None

    m = 1
    while m < size:
        m <<= 1

    phases = np.full(m, 2*np.pi * phase_index / K)
    return build_phase_state(phases)


def run_pes_miniswap(x, y, centers, shots=4096, seed=123, opt_level=3):
    sim = AerSimulator(seed_simulator=seed)
    K = len(centers)

    _, idx_x = quantize_to_centers(x, centers)
    _, idx_y = quantize_to_centers(y, centers)

    groups = get_all_binary_groups(idx_x, idx_y, centers)
    d = len(x)

    cos_est = 0.0

    for (a,b), idxs in groups.items():
        size = len(idxs)

        if size == 0:
            continue

        # caso exacto size=1
        if size == 1:
            phi_a = 2*np.pi * a / K
            phi_b = 2*np.pi * b / K
            cos_est += (size/d) * math.cos(phi_a - phi_b)
            continue

        px = build_sub_phase_state(size, a, K)
        py = build_sub_phase_state(size, b, K)

        qc = build_swap_test(px,py)
        tqc = transpile(qc, sim, optimization_level=opt_level, seed_transpiler=seed)
        res = sim.run(tqc, shots=shots).result()

        cos_ab = corr_to_cos(corr_abs_from_p0(p0_from_counts(res.get_counts(tqc))))
        cos_est += (size/d) * cos_ab

    cos_real = cos_sim(x,y)
    mae = abs(cos_real - cos_est)

    return cos_real, cos_est, mae


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":
    dim   = 256
    K     = 2
    shots = 4096

    rhos = [-0.9, -0.5, 0, 0.5, 0.9]

    print("\n=== AE-SWAP vs PES-DATA-DRIVEN vs PES-MINI-SWAP (SIN HASHING) ===\n")
    print("rho | cos_real | AE | PES-DD | MAE_AE | MAE_DD")

    for i, rho in enumerate(rhos):
        seed = 1000 + i

        x, y = make_pair_with_cosine(dim, rho, seed)
        centers = learn_kmeans_centers(x, y, K=K, seed=seed)

        cos_ae = run_ae_swap(x, y, shots=shots, seed=seed)

        cos_real, cos_disc, cos_pes, mae_disc, mae_pes = run_pes_datadriven(
            x, y, centers, shots=shots, seed=seed
        )


        mae_ae = abs(cos_real - cos_ae)

        print(f"{rho:+.2f} | {cos_real:+.3f} | {cos_ae:+.3f} | {cos_pes:+.3f} | {mae_ae:.3f}")
