#!/usr/bin/env python3
# Compare AE-SWAP vs Phase-Hash SWAP en hardware real (Heron)

import numpy as np, time, math
from pathlib import Path
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from src.utils.secrets_loader import Secrets

# Cargar claves
sec = Secrets()

# =====================================================
# Configuración IBM Quantum
# =====================================================
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=sec.qiskit_api_key,
    instance=sec.qiskit_instance
)
backend = service.backend(sec.hardware_backend)
print("Usando backend:", backend.name)

CSV = "data/raw/hw/compare_swap_destructive_hw.csv"

# =====================================================
# Utilidades
# =====================================================
def make_pair_with_cosine(dim, rho, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(dim); x /= np.linalg.norm(x)
    z = rng.standard_normal(dim); z -= x * np.dot(x, z); z /= np.linalg.norm(z)
    y = rho * x + math.sqrt(1 - rho**2) * z
    return x, y

def build_amp_state(vec):
    from qiskit.circuit.library import Initialize
    n = int(math.ceil(math.log2(len(vec))))
    qc = QuantumCircuit(n, name="amp_state")
    amp = vec / np.linalg.norm(vec)
    if len(amp) < 2**n:
        pad = np.zeros(2**n, dtype=complex)
        pad[:len(amp)] = amp
        amp = pad
    qc.append(Initialize(amp), qc.qubits)
    return qc

def build_phase_state(bits_pm1):
    m = len(bits_pm1)
    n = int(math.log2(m))
    q = QuantumRegister(n, "data")
    qc = QuantumCircuit(q, name="phase_state")
    qc.h(q)
    phases = [1.0 if b > 0 else -1.0 for b in bits_pm1]
    qc.append(DiagonalGate(phases), q[:])
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
    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], qa[i], qb[i])
    qc.h(anc[0])
    qc.measure(anc[0], c[0])
    return qc
def build_destructive_swap(prepA, prepB):
    """
    Prepara un circuito para el Destructive SWAP Test
    para dos estados de igual número de qubits.
    """
    n = prepA.num_qubits

    qa = QuantumRegister(n, "a")
    qb = QuantumRegister(n, "b")
    c = ClassicalRegister(2*n, "m")

    qc = QuantumCircuit(qa, qb, c)

    # Insertar los prepared states
    qc.compose(prepA, qa, inplace=True)
    qc.compose(prepB, qb, inplace=True)

    # Destructive SWAP test
    for i in range(n):
        qc.cx(qa[i], qb[i])
        qc.h(qa[i])

    # Medidas
    for i in range(n):
        qc.measure(qa[i], c[2*i])
        qc.measure(qb[i], c[2*i + 1])

    return qc
def corr_from_destructive_counts(counts, n):
    """
    Calcula la correlación <Z_i Z'_i> a partir del destructive SWAP test.
    n = número de qubits del estado (log2(m) en Phase-Hash)
    """
    total_shots = sum(counts.values())

    corr_total = 0.0
    for i in range(n):
        same = 0
        diff = 0
        for bitstring, cnt in counts.items():
            a = bitstring[-2*n + 2*i]     # bit de a_i
            b = bitstring[-2*n + 2*i + 1] # bit de b_i
            if a == b:
                same += cnt
            else:
                diff += cnt
        corr_i = (same - diff) / total_shots
        corr_total += corr_i

    return corr_total / n
def run_phasehash_destructive_hw(x, y, m, E, shots=1024, seed=123):
    rng = np.random.default_rng(seed)
    corr_list, depths, twoqs = [], [], []
    t0 = time.time()
    for i in range(E):
        print("Essemble: ", i)
        # Phase-Hash: generar sketch binario
        R = rng.standard_normal((m, len(x)))
        b = np.where(R @ x >= 0, 1.0, -1.0)
        c = np.where(R @ y >= 0, 1.0, -1.0)
        sign_proxy = np.sign(np.mean(b*c)) or 1.0

        # Preparar estados de fase
        prep_b = build_phase_state(b)
        prep_c = build_phase_state(c)

        # Circuito destructive SWAP
        qc = build_destructive_swap(prep_b, prep_c)
        tqc = transpile(qc, backend, optimization_level=1)

        sampler = SamplerV2(mode=backend)
        job = sampler.run([tqc], shots=shots)
        pub_result = job.result()[0]

        counts = pub_result.join_data().get_counts()

        # correlación por destructive SWAP
        n = prep_b.num_qubits
        corr = corr_from_destructive_counts(counts, n)

        corr_list.append(corr * sign_proxy)
        depths.append(tqc.depth())
        twoqs.append(count_twoq(tqc))  # sigue contando tus puertas

    corr_hat = np.mean(corr_list)
    cos_hat  = corr_to_cos(corr_hat)
    elapsed = time.time() - t0
    return cos_hat, elapsed, float(np.mean(depths)), float(np.mean(twoqs))

def p0_from_counts(counts):
    shots = sum(counts.values())
    return counts.get("0", 0) / max(1, shots)

def corr_abs_from_p0(p0):
    return math.sqrt(max(0.0, 2*p0 - 1.0))

def corr_to_cos(corr_signed):
    c = max(-1.0, min(1.0, corr_signed))
    theta = (math.pi/2) * (1 - c)
    return math.cos(theta)

def count_twoq(tcirc):
    ops = tcirc.count_ops()
    total = int(sum(ops.get(g, 0) for g in ["cx","cz","swap","ecr","rxx","ryy","rzx","rzz"]))
    total += 3*int(ops.get("cswap",0)) + 6*int(ops.get("ccx",0))
    return total

# =====================================================
# AE-SWAP en hardware real
# =====================================================
def run_swap_amp_hw(x, y, shots=1024):
    prep_x = build_amp_state(x)
    prep_y = build_amp_state(y)
    qc = build_swap_test(prep_x, prep_y)
    tqc = transpile(qc, backend, optimization_level=1)

    sampler = SamplerV2(mode=backend)
    t0 = time.time()
    job = sampler.run([tqc], shots=shots)
    pub_result = job.result()[0]
    elapsed = time.time() - t0

    counts = pub_result.join_data().get_counts()
    p0 = p0_from_counts(counts)
    overlap = corr_abs_from_p0(p0)

    depth = tqc.depth()
    twoq = count_twoq(tqc)

    return overlap, elapsed, depth, twoq

# =====================================================
# Phase-Hash SWAP en hardware real
# =====================================================
def run_swap_phasehash_hw(x, y, m, E, shots=1024, seed=123):
    rng = np.random.default_rng(seed)
    corr_list, depths, twoqs = [], [], []

    t0 = time.time()
    for i in range(E):
        print("Essemble: ", i)
        R = rng.standard_normal((m, len(x)))
        b = np.where(R @ x >= 0, 1.0, -1.0)
        c = np.where(R @ y >= 0, 1.0, -1.0)
        sign_proxy = np.sign(np.mean(b*c)) or 1.0

        prep_b = build_phase_state(b)
        prep_c = build_phase_state(c)
        qc = build_swap_test(prep_b, prep_c)

        tqc = transpile(qc, backend, optimization_level=1)
        sampler = SamplerV2(mode=backend)
        job = sampler.run([tqc], shots=shots)
        pub_result = job.result()[0]

        counts = pub_result.join_data().get_counts()
        p0 = p0_from_counts(counts)

        corr_abs = corr_abs_from_p0(p0)
        corr_signed = corr_abs * sign_proxy

        corr_list.append(corr_signed)
        depths.append(tqc.depth())
        twoqs.append(count_twoq(tqc))

    elapsed = time.time() - t0
    return corr_to_cos(np.mean(corr_list)), elapsed, np.mean(depths), np.mean(twoqs)

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    Path("data/raw/hw").mkdir(parents=True, exist_ok=True)

    dims = [8]     # reducido para hardware real
    m_list = [4]
    E_list = [2]
    cos_targets = [0.2, 0.5, 0.7, 0.9]
    shots_list = [1024]
    reps = 1

    rows = []

    for rep in range(reps):
        for dim in dims:
            base_seed = 2000*rep + dim
            for cos_target in cos_targets:
                print("Cosine: ",cos_target)
                x, y = make_pair_with_cosine(dim, cos_target, seed=base_seed)

                for shots in shots_list:
                    # AE-SWAP
                    cos_amp, t_amp, d_amp, q_amp = run_swap_amp_hw(x, y, shots)
                    rows.append([
                        "AE-SWAP", rep, dim, cos_target, "-", "-", shots,
                        cos_amp, abs(cos_amp - cos_target), t_amp, d_amp, q_amp
                    ])
                    # Phase-Hash SWAP
                    for m in m_list:
                        for E in E_list:
                            cos_ph, t_ph, d_ph, q_ph = run_phasehash_destructive_hw(
                                x, y, m, E, shots, seed=base_seed+m+E
                            )
                            rows.append([
                                "Phase-Hash (SWAP)", rep, dim, cos_target, m, E, shots,
                                cos_ph, abs(cos_ph - cos_target), t_ph, d_ph, q_ph
                            ])

    df = pd.DataFrame(rows, columns=[
        "Method", "Rep", "Dim", "TrueCos", "m", "E", "Shots",
        "EstCos", "AbsErr", "Time_s", "Depth", "TwoQ"
    ])
    df.to_csv(CSV, index=False)
    print(f"✅ Guardadas {len(df)} ejecuciones en {CSV}")
