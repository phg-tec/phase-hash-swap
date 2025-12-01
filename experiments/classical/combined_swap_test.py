# ============================================================
# AE-SWAP vs PES multinivel (naive, opt φ_k, mini-SWAP magnitud)
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
    c = max(-1.0, min(1.0, float(corr)))
    theta = (math.pi / 2.0) * (1 - c)
    return math.cos(theta)


# ============================================================
# AE-SWAP (referencia)
# ============================================================

def build_amp_state(vec):
    vec = np.asarray(vec, float)
    n = int(np.ceil(np.log2(len(vec))))
    pad_len = 2 ** n
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
    corr = corr_abs_from_p0(p0)
    return corr_to_cos(corr)


# ============================================================
# K-means multinivel
# ============================================================

def quantize_to_centers(v, centers):
    v = np.asarray(v)
    centers = np.asarray(centers)
    d = np.abs(v[:, None] - centers[None, :])
    idx = np.argmin(d, axis=1)
    return centers[idx], idx


def learn_centers_global(dim, K, n_pairs, rhos, seed=0):
    """
    Aprende K centros globales con K-means sobre todos los valores de
    varios pares (x,y). Devuelve:
      - centers (array K)
      - train_pairs: lista de (idx_x, idx_y, cos_disc) para entrenar φ_k
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []

    for i in range(n_pairs):
        rho = rhos[i % len(rhos)]
        x, y = make_pair_with_cosine(dim, rho, seed + i)
        xs.append(x)
        ys.append(y)

    all_vals = np.concatenate(xs + ys).reshape(-1, 1)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    km.fit(all_vals)
    centers = np.sort(km.cluster_centers_.flatten())

    train_pairs = []
    for x, y in zip(xs, ys):
        qx, idx_x = quantize_to_centers(x, centers)
        qy, idx_y = quantize_to_centers(y, centers)
        cos_disc = cos_sim(qx, qy)
        train_pairs.append((idx_x, idx_y, cos_disc))

    return centers, train_pairs


# ============================================================
# PES ideal clásico (sin circuito): media de cos(Δφ)
# ============================================================

def pes_cos_from_indices(idx_x, idx_y, phi):
    idx_x = np.asarray(idx_x, int)
    idx_y = np.asarray(idx_y, int)
    d = len(idx_x)
    delta = phi[idx_x] - phi[idx_y]
    return float(np.mean(np.cos(delta)))


# ============================================================
# Optimización de fases φ_k para multinivel
# ============================================================

def optimize_phases(train_pairs, K, n_iters=400, lr=0.2, verbose=True):
    """
    Optimiza φ_k para que cos_PES ≈ cos_disc sobre los train_pairs.
    train_pairs: lista de (idx_x, idx_y, cos_disc).
    """
    phi = np.linspace(0, 2 * np.pi * (K - 1) / K, K)
    phi[0] = 0.0  # fijamos φ_0 = 0 para romper la simetría global

    N = len(train_pairs)

    for it in range(n_iters):
        grad = np.zeros(K)
        loss = 0.0

        for (idx_x, idx_y, cos_target) in train_pairs:
            idx_x = np.asarray(idx_x, int)
            idx_y = np.asarray(idx_y, int)
            d = len(idx_x)

            delta = phi[idx_x] - phi[idx_y]
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)

            g = float(np.mean(cos_d))  # cos_PES
            err = g - cos_target
            loss += err * err

            coef = 2.0 * err / N

            for k in range(K):
                mask_x = (idx_x == k)
                mask_y = (idx_y == k)
                term = 0.0
                if np.any(mask_x):
                    term += -np.sum(sin_d[mask_x])
                if np.any(mask_y):
                    term += np.sum(sin_d[mask_y])
                grad[k] += coef * (term / d)

        phi -= lr * grad
        phi = (phi + 2 * np.pi) % (2 * np.pi)
        phi[0] = 0.0

        if verbose and (it % max(1, n_iters // 10) == 0 or it == n_iters - 1):
            print(f"[iter {it:4d}] loss/N = {loss/N:.6f}, phi = {phi}")

    return phi


# ============================================================
# Estado de fase genérico (para mini-SWAP residual)
# ============================================================

def build_phase_state(phases):
    phases = np.asarray(phases, float)
    m = len(phases)
    n = int(np.log2(m))
    if 2 ** n != m:
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


# ============================================================
# MINI-SWAP MAGNITUDE-ENHANCED (OPCIÓN A)
# ============================================================

def build_sub_phase_state_residuals(residuals, alpha):
    """
    |ψ> = (1/sqrt(|G|)) Σ_i exp(i α residual_i) |i>, con padding a potencia de 2.
    """
    residuals = np.asarray(residuals, float)
    size = len(residuals)

    if size == 1:
        return None  # caso trivial, se maneja fuera

    m = 1
    while m < size:
        m <<= 1

    phases = np.zeros(m)
    phases[:size] = alpha * residuals

    return build_phase_state(phases)


def run_pes_miniswap_magnitude_enhanced(x, y, centers, alpha=20.0,
                                        shots=4096, seed=123, opt_level=3):
    """
    PES multinivel por grupos (k,l), codificando MAGNITUD real dentro de cada grupo.
    r_i = x_i - c_k, s_i = y_i - c_l  → fases α*r_i, α*s_i.
    """
    sim = AerSimulator(seed_simulator=seed)

    qx, idx_x = quantize_to_centers(x, centers)
    qy, idx_y = quantize_to_centers(y, centers)

    K = len(centers)
    d = len(x)

    groups = {}
    for i in range(d):
        k = idx_x[i]
        l = idx_y[i]
        groups.setdefault((k, l), []).append(i)

    cos_est = 0.0

    for (k, l), idxs in groups.items():
        size = len(idxs)
        if size == 0:
            continue

        c_k = centers[k]
        c_l = centers[l]

        residuals_x = x[idxs] - c_k
        residuals_y = y[idxs] - c_l

        # size=1 → resultado analítico
        if size == 1:
            r = residuals_x[0]
            s = residuals_y[0]
            cos_g = math.cos(alpha * (r - s))
            cos_est += (size / d) * cos_g
            continue

        px = build_sub_phase_state_residuals(residuals_x, alpha)
        py = build_sub_phase_state_residuals(residuals_y, alpha)

        qc = build_swap_test(px, py)
        tqc = transpile(qc, sim, optimization_level=opt_level,
                        seed_transpiler=seed)
        res = sim.run(tqc, shots=shots).result()

        p0 = p0_from_counts(res.get_counts(tqc))
        cos_g = corr_to_cos(corr_abs_from_p0(p0))

        cos_est += (size / d) * cos_g

    cos_real = cos_sim(x, y)
    mae = abs(cos_real - cos_est)

    return cos_real, cos_est, mae


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":
    dim        = 256
    K          = 4
    shots      = 4096
    n_pairs_tr = 2000

    rhos_train = [-0.9, -0.5, 0.0, 0.5, 0.9]
    rhos_test  = [-0.9, -0.5, 0.0, 0.5, 0.9]

    print("\n=== ENTRENANDO CENTROS K-MEANS Y FASES φ_k (PES multinivel) ===")
    centers, train_pairs = learn_centers_global(
        dim, K, n_pairs_tr, rhos_train, seed=42
    )
    print("Centros K-means:", centers)

    phi_opt = optimize_phases(train_pairs, K, n_iters=400, lr=0.2, verbose=True)
    print("\nFases optimizadas φ_k:", phi_opt)

    phi_naive = np.linspace(0, 2 * np.pi * (K - 1) / K, K)
    phi_naive[0] = 0.0

    print("\n=== TEST AE-SWAP vs PES (naive) vs PES (opt φ_k) vs PES-mini(A) ===")
    print("rho | cos_real | cos_disc | AE | PES_naive | PES_opt | PES_mini | "
          "MAE_AE | MAE_naive | MAE_opt | MAE_mini")

    for i, rho in enumerate(rhos_test):
        seed = 1000 + i
        x, y = make_pair_with_cosine(dim, rho, seed)
        cos_real = cos_sim(x, y)

        # AE-SWAP
        cos_ae = run_ae_swap(x, y, shots=shots, seed=seed)

        # Discretización con K-means
        qx, idx_x = quantize_to_centers(x, centers)
        qy, idx_y = quantize_to_centers(y, centers)
        cos_disc = cos_sim(qx, qy)

        # PES ideal clásico (naive y opt)
        cos_pes_naive = pes_cos_from_indices(idx_x, idx_y, phi_naive)
        cos_pes_opt   = pes_cos_from_indices(idx_x, idx_y, phi_opt)

        # PES mini-SWAP magnitude-enhanced
        # Escala automática basada en el rango de residuos del grupo
        max_r = max(np.max(np.abs(residuals_x)), np.max(np.abs(residuals_y)))
        alpha = np.pi / max_r if max_r > 1e-12 else 0.0

        _, cos_mini, mae_mini = run_pes_miniswap_magnitude_enhanced(
            x, y, centers, shots=shots, seed=seed
        )

        mae_ae    = abs(cos_real - cos_ae)
        mae_naive = abs(cos_real - cos_pes_naive)
        mae_opt   = abs(cos_real - cos_pes_opt)

        print(f"{rho:+.2f} | {cos_real:+.3f} | {cos_disc:+.3f} | {cos_ae:+.3f} | "
              f"{cos_pes_naive:+.3f} | {cos_pes_opt:+.3f} | {cos_mini:+.3f} | "
              f"{mae_ae:.3f} | {mae_naive:.3f} | {mae_opt:.3f} | {mae_mini:.3f}")
