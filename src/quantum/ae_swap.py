# src/quantum/ae_swap.py
import numpy as np
import math
import time

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import Initialize
from qiskit_aer import AerSimulator

from src.utils.basic import p0_from_counts, corr_abs_from_p0, corr_to_cos


def build_amp_state(vec: np.ndarray) -> QuantumCircuit:
    """
    Construye un circuito que prepara el estado |ψ⟩ con amplitudes
    proporcionales a 'vec' (amplitude encoding).
    """
    vec = np.asarray(vec, float)
    n = int(np.ceil(np.log2(len(vec))))
    pad_len = 2 ** n

    amp = np.zeros(pad_len, complex)
    amp[:len(vec)] = vec / np.linalg.norm(vec)

    qr = QuantumRegister(n, "amp")
    qc = QuantumCircuit(qr, name="amp_state")
    qc.append(Initialize(amp), qr)
    return qc


def build_swap_test(prepA: QuantumCircuit, prepB: QuantumCircuit) -> QuantumCircuit:
    """
    SWAP test estándar entre dos subcircuitos de preparación prepA, prepB.
    """
    n = prepA.num_qubits

    anc = QuantumRegister(1, "anc")
    qa = QuantumRegister(n, "a")
    qb = QuantumRegister(n, "b")
    c = ClassicalRegister(1, "m")

    qc = QuantumCircuit(anc, qa, qb, c, name="ae_swap")

    qc.compose(prepA, qa, inplace=True)
    qc.compose(prepB, qb, inplace=True)

    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], qa[i], qb[i])
    qc.h(anc[0])

    qc.measure(anc[0], c[0])
    return qc


def run_ae_swap(x,
                y,
                shots: int = 2048,
                seed: int = 123,
                opt_level: int = 3):
    """
    Ejecuta AE-SWAP y devuelve:
      cos_est, t_quantum
    donde t_quantum incluye build + transpile + run.
    """
    sim = AerSimulator(seed_simulator=seed)

    t0 = time.perf_counter()

    prep_x = build_amp_state(x)
    prep_y = build_amp_state(y)
    qc = build_swap_test(prep_x, prep_y)

    tqc = transpile(qc, sim, optimization_level=opt_level, seed_transpiler=seed)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts(tqc)

    t1 = time.perf_counter()
    t_quantum = t1 - t0

    p0 = p0_from_counts(counts)
    corr_abs = corr_abs_from_p0(p0)
    cos_est = corr_to_cos(corr_abs)

    return cos_est, t_quantum, p0