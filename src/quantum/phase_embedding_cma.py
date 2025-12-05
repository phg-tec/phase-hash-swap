# ============================================================
#   phase_embedding_cma.py (versión mejorada, misma API)
# ============================================================

import math
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

# ============================================================
#   MLP(cos) -> (phi_minus, phi_plus)
# ============================================================

IN_DIM = 1
HIDDEN = 32   # AUMENTAMOS CAPACIDAD
OUT_DIM = 2

# Parámetros:
# W1: (HIDDEN, 1)   -> HIDDEN
# b1: (HIDDEN,)     -> HIDDEN
# W2: (OUT_DIM, HIDDEN) -> OUT_DIM * HIDDEN
# b2: (OUT_DIM,)    -> OUT_DIM
N_PARAMS_MLP = HIDDEN * IN_DIM + HIDDEN + OUT_DIM * HIDDEN + OUT_DIM


def get_num_params_cos_mlp() -> int:
    return N_PARAMS_MLP


def _unpack_mlp_params(theta: np.ndarray):
    """
    Desempaqueta vector theta en W1, b1, W2, b2 para el MLP(cos).
    """
    theta = np.asarray(theta, float)
    if theta.size != N_PARAMS_MLP:
        raise ValueError(f"theta.size = {theta.size}, esperado {N_PARAMS_MLP}")

    offset = 0

    # W1
    W1_size = HIDDEN * IN_DIM
    W1 = theta[offset:offset + W1_size].reshape(HIDDEN, IN_DIM)
    offset += W1_size

    # b1
    b1 = theta[offset:offset + HIDDEN]
    offset += HIDDEN

    # W2
    W2_size = OUT_DIM * HIDDEN
    W2 = theta[offset:offset + W2_size].reshape(OUT_DIM, HIDDEN)
    offset += W2_size

    # b2
    b2 = theta[offset:offset + OUT_DIM]
    offset += OUT_DIM

    return W1, b1, W2, b2


def phases_from_cos(cos_val: float, params: np.ndarray):
    """
    cos_val: escalar en [-1,1]
    params: vector de longitud N_PARAMS_MLP

    MLP mejorado:
        h = tanh(W1 * cos + b1)
        out = tanh(W2 * h + b2) * pi
    Esto limita la salida a [-pi,pi] sin explosiones.
    """
    c = float(cos_val)
    x = np.array([c], dtype=float).reshape(IN_DIM,)

    W1, b1, W2, b2 = _unpack_mlp_params(params)

    # Capa 1
    h = W1 @ x + b1
    h = np.tanh(h)

    # Salida (tanh para evitar saltos al envolver)
    out = W2 @ h + b2
    out = np.tanh(out) * math.pi   # FASES SIEMPRE EN [-pi,pi]

    phi_minus = float(out[0])
    phi_plus  = float(out[1])

    return phi_minus, phi_plus


def phases_to_complex(phi_minus: float, phi_plus: float):
    z_minus = np.cos(phi_minus) + 1j * np.sin(phi_minus)
    z_plus  = np.cos(phi_plus)  + 1j * np.sin(phi_plus)
    return z_minus, z_plus


def map_binary_to_complex(bits_pm1: np.ndarray, z_minus: complex, z_plus: complex):
    bits_pm1 = np.asarray(bits_pm1)
    return np.where(bits_pm1 == -1, z_minus, z_plus)


def build_phase_state_from_bits(bits_pm1: np.ndarray,
                                z_minus: complex,
                                z_plus: complex) -> QuantumCircuit:
    bits_pm1 = np.asarray(bits_pm1)
    m = len(bits_pm1)
    n = int(math.log2(m))
    if 2**n != m:
        raise ValueError(f"Vector length {m} is not power of 2.")

    diag = map_binary_to_complex(bits_pm1, z_minus, z_plus)

    q = QuantumRegister(n, "data")
    qc = QuantumCircuit(q, name="phase_state")
    qc.h(q)
    qc.append(DiagonalGate(diag), q)

    return qc


def build_swap_test_circuit(bits_x: np.ndarray,
                            bits_y: np.ndarray,
                            phi_minus: float,
                            phi_plus: float) -> QuantumCircuit:

    if len(bits_x) != len(bits_y):
        raise ValueError("bits_x y bits_y deben tener misma longitud")

    m = len(bits_x)
    n = int(math.log2(m))
    if 2**n != m:
        raise ValueError("La longitud de los vectores debe ser potencia de 2.")

    z_minus, z_plus = phases_to_complex(phi_minus, phi_plus)

    qc_x = build_phase_state_from_bits(bits_x, z_minus, z_plus)
    qc_y = build_phase_state_from_bits(bits_y, z_minus, z_plus)

    anc = QuantumRegister(1, "anc")
    qx = QuantumRegister(n, "x")
    qy = QuantumRegister(n, "y")
    c  = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, qx, qy, c)

    qc = qc.compose(qc_x, qubits=qx)
    qc = qc.compose(qc_y, qubits=qy)

    qc.h(anc)
    for i in range(n):
        qc.cswap(anc, qx[i], qy[i])
    qc.h(anc)
    qc.measure(anc, c)

    return qc


def run_swap_test_p0(bits_x: np.ndarray,
                     bits_y: np.ndarray,
                     phi_minus: float,
                     phi_plus: float,
                     shots: int = 2048,
                     backend=None) -> float:

    if backend is None:
        backend = AerSimulator()

    qc = build_swap_test_circuit(bits_x, bits_y, phi_minus, phi_plus)
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    n0 = counts.get('0', 0)
    n1 = counts.get('1', 0)
    total = n0 + n1
    if total == 0:
        return 0.5
    return n0 / total


def true_cos_from_bits(bits_x: np.ndarray, bits_y: np.ndarray) -> float:
    x = np.asarray(bits_x, float)
    y = np.asarray(bits_y, float)
    dot = np.dot(x, y)
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)
    if normx == 0 or normy == 0:
        return 0.0
    return dot / (normx * normy)


def p0_target_from_bits(bits_x: np.ndarray, bits_y: np.ndarray) -> float:
    cos_xy = true_cos_from_bits(bits_x, bits_y)
    return 0.5 * (1.0 + cos_xy)
