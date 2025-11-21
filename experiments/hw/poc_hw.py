#!/usr/bin/env python3
# Ejecutar Phase-Hash SWAP en hardware real (ibm_heron2)
# =====================================================

import numpy as np, time, math
import pandas as pd
from pathlib import Path

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

# =====================================================
# TUS FUNCIONES ORIGINALES (SIN CAMBIAR)
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
# ADAPTACIÓN: EJECUTAR EN HARDWARE REAL
# =====================================================
def run_swap_phasehash_hw(x, y, m, E, shots=2048, seed=123):
    rng = np.random.default_rng(seed)

    corr_list, depths, twoqs = [], [], []
    t0 = time.time()

    # E repeticiones exactamente igual que en la versión simulada
    for i in range(E):
        print("Enssemble: ", i)
        R = rng.standard_normal((m, len(x)))
        b = np.where(R @ x >= 0, 1.0, -1.0)
        c = np.where(R @ y >= 0, 1.0, -1.0)
        sign_proxy = np.sign(np.mean(b*c)) or 1.0

        prep_b = build_phase_state(b)
        prep_c = build_phase_state(c)
        qc = build_swap_test(prep_b, prep_c)

        # transpile para ibm_fez
        tqc = transpile(qc, backend, optimization_level=1,
                        seed_transpiler=int(rng.integers(0,1<<30)))

        # ejecutar en hardware real
        sampler = SamplerV2(mode=backend)
        job = sampler.run([tqc], shots=shots)
        pub_result = job.result()[0]

        # === AQUÍ ESTÁ LA CLAVE ===
        # Extraer counts de SamplerV2
        counts = pub_result.join_data().get_counts()

        # resto idéntico
        p0 = p0_from_counts(counts)
        corr_abs = corr_abs_from_p0(p0)
        corr_signed = corr_abs * sign_proxy
        corr_list.append(corr_signed)
        depths.append(tqc.depth())
        twoqs.append(count_twoq(tqc))

    elapsed = time.time() - t0
    corr_hat = float(np.mean(corr_list))
    cos_hat  = corr_to_cos(corr_hat)

    return cos_hat, elapsed, float(np.mean(depths)), float(np.mean(twoqs))




# =====================================================
# MAIN (IDÉNTICO AL TUYO)
# =====================================================
if __name__ == "__main__":
    Path("data/raw/hw").mkdir(parents=True, exist_ok=True)
    CSV = "data/raw/hw/pof_hw.csv"

    dim = 128
    m = 64
    E = 16
    cos_targets = [0.2, 0.5, 0.7, 0.9]
    shots = 4096
    reps = 1

    rows = []

    for rep in range(reps):
        base_seed = 2000*rep + dim
        print("Ejecución: ", rep)
        for cos_target in cos_targets:
            print("Coseno: ", cos_target)
            x, y = make_pair_with_cosine(dim, cos_target, seed=base_seed)
            cos_ph, t_ph, d_ph, q_ph = run_swap_phasehash_hw(
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
    print(f"✅ Guardadas {len(df)} ejecuciones high-dim en {CSV}")
    print("🚀 Ejecutado en hardware real ibm_heron2")
