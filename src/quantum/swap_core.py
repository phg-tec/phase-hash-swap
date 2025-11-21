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
# Classical SimHash baseline
# ----------------------------------------------------------
def simhash_classical_corr(x, y, m=256, seed=123):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((m, len(x)))
    z_x = np.sign(R @ x); z_x[z_x == 0] = 1
    z_y = np.sign(R @ y); z_y[z_y == 0] = 1
    r = float(np.mean(z_x * z_y))
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
    Phase-hash SWAP with E ensemble repetitions.

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
