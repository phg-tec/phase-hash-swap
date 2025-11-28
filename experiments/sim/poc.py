# src/quantum/swap_core.py
# =========================================================
# Core utilities for AE-SWAP and Phase-Hash (PES) SWAP
# Shared by all sweeps and experiments
# =========================================================

import math
import time
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import DiagonalGate


# ----------------------------------------------------------
# Basic math / data utilities
# ----------------------------------------------------------
def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def make_pair_with_cosine(dim, rho, seed=123):
    """
    Generate x,y in R^dim with cosine approximately rho.
    Construction: y = rho*x + sqrt(1-rho^2)*z, z orthogonal to x.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(dim)
    x /= np.linalg.norm(x)

    z = rng.standard_normal(dim)
    z -= x * np.dot(x, z)
    z /= np.linalg.norm(z)

    y = rho * x + math.sqrt(max(0.0, 1 - rho**2)) * z
    return x, y


# ----------------------------------------------------------
# Deterministic Hadamard-based hashing (no randomness)
# ----------------------------------------------------------
def _is_power_of_two(m: int) -> bool:
    return m > 0 and (m & (m - 1)) == 0


def hadamard_hash_real(x, m):
    """
    Deterministic projection using Walsh–Hadamard transform.

    - x: vector real (dim=d)
    - m: final number of components (hash size)

    Steps:
      D = next power of 2 >= max(d, m)
      pad x to D
      apply FWHT
      return first m coefficients (real values)
    """
    x = np.asarray(x, float)
    d = len(x)
    if d == 0:
        raise ValueError("hadamard_hash_real: x must be non-empty")

    # Compute D = next power of 2 >= max(d,m)
    target = max(d, m)
    D = 1
    while D < target:
        D <<= 1

    # Padding
    v = np.zeros(D, dtype=float)
    v[:d] = x

    # FWHT
    h = 1
    while h < D:
        step = h * 2
        for i in range(0, D, step):
            a = v[i:i+h].copy()
            b = v[i+h:i+step].copy()
            v[i:i+h] = a + b
            v[i+h:i+step] = a - b
        h <<= 1

    return v[:m]

def quantize_to_K_levels(real_vec, K):
    """
    Convert a real vector to K phase levels.

    Steps:
    - map real values to indexes {0,...,K-1} using uniform bins
    - convert indexes to phases in [0,2π)
    - return phi vector
    """
    rv = np.asarray(real_vec, float)

    rmin, rmax = rv.min(), rv.max()
    if rmax == rmin:
        # All equal — assign a constant phase (0)
        indices = np.zeros_like(rv, dtype=int)
    else:
        bins = np.linspace(rmin, rmax, K + 1)
        indices = np.digitize(rv, bins) - 1
        indices = np.clip(indices, 0, K - 1)

    # Map indices → phases
    phi = 2*np.pi * indices / K
    return phi

def build_phase_state_multilevel(phi):
    """
    Build a quantum state:

      |ψ(x)> = 1/sqrt(m) sum_k exp(i φ_k) |k>

    where len(phi)=m=2^n.

    phi: vector of phases in [0,2π)
    """
    phi = np.asarray(phi, float)
    m = len(phi)
    n = int(np.log2(m))
    if 2**n != m:
        raise ValueError("phase vector length m must be a power of 2")

    phases = np.exp(1j * phi)

    q = QuantumRegister(n, "data")
    qc = QuantumCircuit(q, name="phase_state_K")

    qc.h(q)   # uniform superposition
    qc.append(DiagonalGate(phases), q[:])
    return qc


# ----------------------------------------------------------
# Classical SimHash baselines
# ----------------------------------------------------------
def simhash_classical_corr(x, y, m=256, seed=123):
    """
    Classical random SimHash with Gaussian projection matrix.
    Returns:
      r       = average product of bits in {-1,1}
      c_class = cos_hat obtained from r via corr_to_cos
    """
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((m, len(x)))
    z_x = np.sign(R @ x); z_x[z_x == 0] = 1
    z_y = np.sign(R @ y); z_y[z_y == 0] = 1
    r = float(np.mean(z_x * z_y))
    c_class = corr_to_cos(r)
    return r, c_class


def simhash_hadamard_classical_corr(x, y, m):
    """
    Classical deterministic baseline:
    - Use Walsh–Hadamard-based hash instead of random Gaussian R.

    Returns:
      r       = average product of bits in {-1,1}
      c_class = cos_hat obtained from r via corr_to_cos
    """
    b_x = hadamard_hash_bits_pm1(x, m)
    b_y = hadamard_hash_bits_pm1(y, m)
    r = float(np.mean(b_x * b_y))
    c_class = corr_to_cos(r)
    return r, c_class


# ----------------------------------------------------------
# Quantum state preparation
# ----------------------------------------------------------
def build_amp_state(vec):
    """
    Amplitude encoding using Initialize.
    Pads to 2^n if needed.
    """
    from qiskit.circuit.library import Initialize

    vec = np.asarray(vec, dtype=float)
    n = int(math.ceil(math.log2(len(vec))))
    qc = QuantumCircuit(n, name="amp_state")

    amp = vec / np.linalg.norm(vec)

    if len(amp) < 2**n:
        pad = np.zeros(2**n, dtype=complex)
        pad[:len(amp)] = amp.astype(complex)
        amp = pad

    qc.append(Initialize(amp), qc.qubits)
    return qc


def build_phase_state(bits_pm1):
    """
    Phase-hash encoding:
    Start in uniform superposition, then DiagonalGate with ±1 phases.
    bits_pm1 length must be power of 2.
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


# ----------------------------------------------------------
# SWAP test circuit
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# Measurements / conversions
# ----------------------------------------------------------
def p0_from_counts(counts):
    shots = sum(counts.values())
    return counts.get("0", 0) / max(1, shots)


def corr_abs_from_p0(p0):
    return math.sqrt(max(0.0, 2*p0 - 1.0))


def corr_to_cos(corr_signed):
    c = max(-1.0, min(1.0, float(corr_signed)))
    theta = (math.pi/2) * (1 - c)
    return math.cos(theta)


def count_twoq(tcirc):
    """
    Two-qubit gate counting heuristic used in your sweeps.
    """
    ops = tcirc.count_ops()
    total = int(sum(ops.get(g, 0) for g in
                    ["cx","cz","swap","ecr","rxx","ryy","rzx","rzz"]))
    total += 3*int(ops.get("cswap", 0)) + 6*int(ops.get("ccx", 0))
    return total


# ----------------------------------------------------------
# Runners
# ----------------------------------------------------------
def run_swap_amp(x, y, shots=2048, seed=None, opt_level=3, measure_cost=False):
    """
    Returns:
      - overlap_hat (NOT cosine) if measure_cost=False
      - else (overlap_hat, elapsed, depth, twoq)
    """
    sim = AerSimulator(seed_simulator=seed)

    prep_x = build_amp_state(x)
    prep_y = build_amp_state(y)
    qc = build_swap_test(prep_x, prep_y)

    tqc = transpile(qc, sim, optimization_level=opt_level,
                    seed_transpiler=seed)

    t0 = time.time()
    res = sim.run(tqc, shots=shots).result()
    elapsed = time.time() - t0

    p0 = p0_from_counts(res.get_counts(tqc))
    overlap = corr_abs_from_p0(p0)

    if not measure_cost:
        return overlap, elapsed

    return overlap, elapsed, tqc.depth(), count_twoq(tqc)


def run_swap_phasehash(x, y, m, E, shots=2048, seed=123,
                       opt_level=3, measure_cost=False):
    """
    Phase-hash SWAP with E ensemble repetitions (random R).

    Returns:
      cos_hat, elapsed  (or plus avg depth,twoq if measure_cost)
    """
    sim = AerSimulator()
    rng = np.random.default_rng(seed)

    corr_list = []
    depths, twoqs = [], []

    t0 = time.time()
    for _ in range(E):
        R = rng.standard_normal((m, len(x)))
        b = np.where(R @ x >= 0, 1.0, -1.0)
        c = np.where(R @ y >= 0, 1.0, -1.0)

        sign_proxy = np.sign(np.mean(b*c)) or 1.0

        prep_b = build_phase_state(b)
        prep_c = build_phase_state(c)
        qc = build_swap_test(prep_b, prep_c)

        seed_t = int(rng.integers(0, 1 << 30))
        seed_s = int(rng.integers(0, 1 << 30))

        tqc = transpile(qc, sim, optimization_level=opt_level,
                        seed_transpiler=seed_t)
        res = sim.run(tqc, shots=shots, seed_simulator=seed_s).result()

        p0 = p0_from_counts(res.get_counts(tqc))
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


def run_swap_phasehash_hadamard_multilevel(
    x,
    y,
    m,
    K=4,
    shots=2048,
    opt_level=3,
    measure_cost=False
):
    """
    Deterministic Hadamard-PES with K phase levels and sign proxy (como PES-rand).

    Steps:
      - deterministic projection: FWHT(x), FWHT(y) -> hx, hy (reales)
      - sign proxy clásico a partir del signo de hx, hy
      - quantize to K phases
      - build quantum phase states
      - perform SWAP test
      - aplica el sign_proxy al resultado del SWAP (igual idea que run_swap_phasehash)
    """
    sim = AerSimulator()

    # ------------------------------------------------------
    # 1) Proyección determinista Hadamard (real, longitud m)
    # ------------------------------------------------------
    hx = hadamard_hash_real(x, m)
    hy = hadamard_hash_real(y, m)

    # ------------------------------------------------------
    # 2) Sign proxy clásico (versión Hadamard de PES-rand)
    # ------------------------------------------------------
    bits_x = np.where(hx >= 0.0, 1.0, -1.0)
    bits_y = np.where(hy >= 0.0, 1.0, -1.0)
    r_sign = float(np.mean(bits_x * bits_y))
    sign_proxy = float(np.sign(r_sign) or 1.0)

    # ------------------------------------------------------
    # 3) Cuantización a K niveles de fase en [0, 2π)
    # ------------------------------------------------------
    phix = quantize_to_K_levels(hx, K)  # vector de m fases
    phiy = quantize_to_K_levels(hy, K)

    # ------------------------------------------------------
    # 4) Construir los estados de fase y SWAP test
    # ------------------------------------------------------
    prep_x = build_phase_state_multilevel(phix)
    prep_y = build_phase_state_multilevel(phiy)

    qc = build_swap_test(prep_x, prep_y)
    tqc = transpile(qc, sim, optimization_level=opt_level)

    t0 = time.time()
    res = sim.run(tqc, shots=shots).result()
    elapsed = time.time() - t0

    # ------------------------------------------------------
    # 5) De p0 -> |corr| -> aplicar signo -> cos_hat
    # ------------------------------------------------------
    p0 = p0_from_counts(res.get_counts(tqc))
    corr_abs = corr_abs_from_p0(p0)

    corr_signed = corr_abs * sign_proxy
    cos_hat = corr_to_cos(corr_signed)

    if not measure_cost:
        return cos_hat, elapsed

    return cos_hat, elapsed, tqc.depth(), count_twoq(tqc)




if __name__ == "__main__":
    # Quick sanity check / demo for AE-SWAP, random PES-SWAP, and Hadamard-PES
    import numpy as np

    dim   = 2048      # dimensión de los vectores clásicos
    m     = 2048     # nº de hashes (para phase-hash); debe ser potencia de 2 y >= dim para Hadamard
    E     = 16       # nº de ensembles para el PES aleatorio
    shots = 8192    # nº de shots para los SWAP tests

    rhos = [-0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9]  # cosenos objetivo (solo positivos para evitar lío de signo en AE)

    print("=== Demo swap_core.py ===")
    print(f"dim = {dim}, m = {m}, E = {E}, shots = {shots}")
#    print("rho_target | cos_true | AE-SWAP | PES-rand | PES-Hadamard")
#    print("rho_target | cos_true | AE-SWAP | AE-TIME | PES-Hadamard | PES-TIME | MAE Difference (- está bien + está mal)")
#    print("rho_target | cos_true | PES-Hadamard | PES-TIME | MAE")
    print("rho_target | cos_true | PES-SH | TIME-Rand | PES-Hadamard | TIME-Hadamard | MAE Difference (- está bien + está mal)")
    for k, rho in enumerate(rhos):
        # Generamos un par (x,y) con cos(x,y) ≈ rho
        x, y = make_pair_with_cosine(dim, rho, seed=123 + k)
        cos_true = float(np.dot(x, y))  # debería ser muy cercano a rho

        # AE-SWAP: devuelve |<x|y>| ≈ |cos(x,y)|, como rho>=0 tomamos cos_ae = overlap
        #cos_ae, time_ae = run_swap_amp(x, y, shots=shots, seed=42, measure_cost=False)

        # PES aleatorio (matriz R ~ N(0,1))
        cos_pes_rand, time_pes_rand = run_swap_phasehash(
            x,
            y,
            m=m,
            E=E,
            shots=shots,
            seed=123,
            opt_level=3,
            measure_cost=False,
        )

        # PES determinista Hadamard
        cos_pes_had, time_pes_had = run_swap_phasehash_hadamard_multilevel(
            x,
            y,
            m=m,   # final hash length = 128 bases = 7 qubits
            K=2,     # 4-phase PES
            shots=shots
        )


        print(
            f"{rho:9.3f} | "
            f"{cos_true:9.3f} | "
#            f"{cos_ae:8.3f} | "
#            f"{time_ae:8.3f} | "
            f"{cos_pes_rand:8.3f} | "
            f"{time_pes_rand:8.3f} | "
            f"{cos_pes_had:8.3f} |"
            f"{time_pes_had:8.3f} | "
            f"{(abs(cos_true - cos_pes_had)-abs(cos_true - cos_pes_rand)):11.3f}"
#            f"{(abs(cos_true - cos_pes_had)):11.3f}"
        )
