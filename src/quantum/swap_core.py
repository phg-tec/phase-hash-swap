# =========================================================
# swap_core.py  (VERSIÓN COMPLETA CON EMBEDDINGS APRENDIDOS)
# =========================================================

import math
import time
import numpy as np
import torch

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import DiagonalGate


from src.quantum.phase_embedding_cma import (
    true_cos_from_bits,
    phases_from_cos,
    phases_to_complex,
    map_binary_to_complex,
)

# Cargar parámetros aprendidos UNA VEZ
_phase_params_cos = np.load("learned_phase_params_cos.npy")  # [a0,a1,a2,b0,b1,b2]

# =========================================================
# NORMALIZAR / UTILIDADES
# =========================================================

def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def make_pair_with_cosine(dim, rho, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(dim)
    x /= np.linalg.norm(x)

    z = rng.standard_normal(dim)
    z -= x * np.dot(x, z)
    z /= np.linalg.norm(z)

    y = rho*x + math.sqrt(max(0, 1-rho*rho))*z
    return x, y


# =========================================================
# SIMHASH CLÁSICO
# =========================================================

def simhash_classical_corr(x, y, m=256, seed=123):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((m, len(x)))
    z_x = np.sign(R @ x); z_x[z_x==0] = 1
    z_y = np.sign(R @ y); z_y[z_y==0] = 1
    r = float(np.mean(z_x*z_y))
    return r, corr_to_cos(r)


# =========================================================
# QUANTUM STATE PREPARATION
# =========================================================

def build_amp_state(vec):
    from qiskit.circuit.library import Initialize
    vec = np.asarray(vec, complex)
    n = int(math.ceil(math.log2(len(vec))))

    qc = QuantumCircuit(n, name="amp")
    amp = vec / np.linalg.norm(vec)

    if len(amp) < 2**n:
        tmp = np.zeros(2**n, dtype=complex)
        tmp[:len(amp)] = amp
        amp = tmp

    qc.append(Initialize(amp), qc.qubits)
    return qc




def build_phase_state_cos_dependent(bits_pm1_x, bits_pm1_y):
    """
    Construye el estado |psi_x> para un vector binario bits_pm1_x,
    usando fases que dependen del coseno entre (x,y).
    """
    bits_pm1_x = np.asarray(bits_pm1_x)
    bits_pm1_y = np.asarray(bits_pm1_y)

    m = len(bits_pm1_x)
    if len(bits_pm1_y) != m:
        raise ValueError("x e y deben tener la misma longitud.")

    n = int(math.log2(m))
    if 2**n != m:
        raise ValueError("La longitud debe ser potencia de 2.")

    # 1) coseno entre bits
    cos_xy = true_cos_from_bits(bits_pm1_x, bits_pm1_y)

    # 2) fases condicionadas al coseno
    phi_minus, phi_plus = phases_from_cos(cos_xy, _phase_params_cos)

    # 3) complejos unitarios
    z_minus, z_plus = phases_to_complex(phi_minus, phi_plus)

    # 4) mapear el vector x a complejos
    diag_x = map_binary_to_complex(bits_pm1_x, z_minus, z_plus)

    q = QuantumRegister(n, "data")
    qc = QuantumCircuit(q)
    qc.h(q)
    qc.append(DiagonalGate(diag_x), q)

    return qc
# =========================================================
# SWAP TEST
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
# MEASUREMENTS
# =========================================================

def p0_from_counts(counts):
    shots = sum(counts.values())
    return counts.get("0", 0) / max(1, shots)


def corr_abs_from_p0(p0):
    return math.sqrt(max(0.0, 2*p0 - 1.0))


def corr_to_cos(corr_signed):
    c = max(-1, min(1, float(corr_signed)))
    theta = (math.pi/2)*(1-c)
    return math.cos(theta)


def count_twoq(tcirc):
    ops = tcirc.count_ops()
    total = int(sum(ops.get(g, 0) for g in
                    ["cx","cz","swap","ecr","rxx","ryy","rzx","rzz"]))
    total += 3*int(ops.get("cswap", 0)) + 6*int(ops.get("ccx", 0))
    return total



# =========================================================
# RUNNERS: AE-SWAP
# =========================================================

def run_swap_amp(x, y, shots=2048, seed=None, opt_level=3, measure_cost=False):
    sim = AerSimulator(seed_simulator=seed)

    prep_x = build_amp_state(x)
    prep_y = build_amp_state(y)
    qc = build_swap_test(prep_x, prep_y)

    tqc = transpile(qc, sim, optimization_level=opt_level, seed_transpiler=seed)

    t0 = time.time()
    res = sim.run(tqc, shots=shots).result()
    elapsed = time.time() - t0

    p0 = p0_from_counts(res.get_counts(tqc))
    overlap = corr_abs_from_p0(p0)

    if not measure_cost:
        return overlap, elapsed, p0

    return overlap, elapsed, tqc.depth(), count_twoq(tqc)


# =========================================================
# RUNNERS: PES-SWAP CON EMBEDDINGS APRENDIDOS
# =========================================================

def run_swap_phasehash(x, y, m, E, shots=2048, seed=123,
                       opt_level=3, measure_cost=False,
                       model=None, device="cpu"):
    """
    PES adaptativo:
      - calcula cos_target ≈ correlación de b*c
      - obtiene (θ+, θ-) del modelo
      - codifica cada hash b,c con exp(iθ±)
      - ejecuta swap test
    """

    sim = AerSimulator()
    rng = np.random.default_rng(seed)

    corr_list = []
    depths, twoqs = [], []
    p0_avg = 0

    t0 = time.time()

    for _ in range(E):

        # === 1. Hash binario ===
        R = rng.standard_normal((m, len(x)))
        b = np.where(R @ x >= 0, 1.0, -1.0)
        c = np.where(R @ y >= 0, 1.0, -1.0)
        
        # === 4. Preparar estados cuánticos ===
        prep_b = build_phase_state_cos_dependent(b, c)
        prep_c = build_phase_state_cos_dependent(c, b)

        # === 5. Ejecutar SWAP ===
        seed_t = int(rng.integers(0, 1 << 30))
        seed_s = int(rng.integers(0, 1 << 30))

        qc = build_swap_test(prep_b, prep_c)
        tqc = transpile(qc, sim, optimization_level=opt_level,
                        seed_transpiler=seed_t)
        res = sim.run(tqc, shots=shots, seed_simulator=seed_s).result()

        p0 = p0_from_counts(res.get_counts(tqc))
        p0_avg += p0

        corr_abs = corr_abs_from_p0(p0)

        # firmar con la correlación binaria
        sign_proxy = np.sign(np.mean(x*y)) or 1.0
        corr_signed = corr_abs * sign_proxy
        corr_list.append(corr_signed)

        if measure_cost:
            depths.append(tqc.depth())
            twoqs.append(count_twoq(tqc))

    p0_avg /= E

    elapsed = time.time() - t0
    corr_hat = float(np.mean(corr_list))
    cos_hat = corr_to_cos(corr_hat)

    if not measure_cost:
        return cos_hat, elapsed, p0_avg

    return cos_hat, elapsed, float(np.mean(depths)), float(np.mean(twoqs))
