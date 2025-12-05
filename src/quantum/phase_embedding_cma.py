# src/quantum/phase_embedding_cma.py

import math
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

# ============================================================
#   MLP(cos) -> (phi_minus, phi_plus)
# ============================================================

# Arquitectura: input=1 -> hidden=8 -> output=2
IN_DIM = 1
HIDDEN = 8
OUT_DIM = 2

# Parámetros:
# W1: (HIDDEN, 1)   -> HIDDEN
# b1: (HIDDEN,)     -> HIDDEN
# W2: (OUT_DIM, HIDDEN) -> OUT_DIM * HIDDEN
# b2: (OUT_DIM,)    -> OUT_DIM
N_PARAMS_MLP = HIDDEN * IN_DIM + HIDDEN + OUT_DIM * HIDDEN + OUT_DIM  # 34


def get_num_params_cos_mlp() -> int:
    """
    Número de parámetros del MLP(cos).
    """
    return N_PARAMS_MLP


def _unpack_mlp_params(theta: np.ndarray):
    """
    Desempaqueta vector theta en W1, b1, W2, b2 para el MLP(cos).
    """
    theta = np.asarray(theta, float)
    if theta.size != N_PARAMS_MLP:
        raise ValueError(f"theta.size = {theta.size}, esperado {N_PARAMS_MLP}")

    offset = 0

    # W1: (HIDDEN, 1)
    W1_size = HIDDEN * IN_DIM
    W1 = theta[offset:offset + W1_size].reshape(HIDDEN, IN_DIM)
    offset += W1_size

    # b1: (HIDDEN,)
    b1 = theta[offset:offset + HIDDEN]
    offset += HIDDEN

    # W2: (OUT_DIM, HIDDEN)
    W2_size = OUT_DIM * HIDDEN
    W2 = theta[offset:offset + W2_size].reshape(OUT_DIM, HIDDEN)
    offset += W2_size

    # b2: (OUT_DIM,)
    b2 = theta[offset:offset + OUT_DIM]
    offset += OUT_DIM

    if offset != N_PARAMS_MLP:
        raise RuntimeError("Error interno desempaquetando parámetros MLP.")

    return W1, b1, W2, b2


def phases_from_cos(cos_val: float, params: np.ndarray):
    """
    cos_val: escalar en [-1,1]
    params: vector de parámetros del MLP(cos), longitud N_PARAMS_MLP.

    MLP:
      h = tanh(W1 * c + b1)
      out = W2 * h + b2  -> [phi_minus, phi_plus]
    Luego envolvemos fases a [-pi, pi] para estabilidad.
    """
    c = float(cos_val)
    x = np.array([c], dtype=float).reshape(IN_DIM,)

    W1, b1, W2, b2 = _unpack_mlp_params(params)

    # Forward
    # capa 1
    h = W1 @ x + b1          # shape (HIDDEN,)
    h = np.tanh(h)           # no-linealidad suave

    # salida
    out = W2 @ h + b2        # shape (2,)

    phi_minus = float(out[0])
    phi_plus  = float(out[1])

    # Envolver a [-pi, pi]
    two_pi = 2.0 * math.pi
    phi_minus = (phi_minus + math.pi) % two_pi - math.pi
    phi_plus  = (phi_plus  + math.pi) % two_pi - math.pi

    return phi_minus, phi_plus


def phases_to_complex(phi_minus: float, phi_plus: float):
    """
    Dadas dos fases reales, devuelve los complejos unitarios z_- y z_+.
    """
    z_minus = np.cos(phi_minus) + 1j * np.sin(phi_minus)
    z_plus  = np.cos(phi_plus)  + 1j * np.sin(phi_plus)
    return z_minus, z_plus


def map_binary_to_complex(bits_pm1: np.ndarray, z_minus: complex, z_plus: complex):
    """
    bits_pm1: array de -1/+1, shape (m,)
    Devuelve array de complejos de la misma shape.
    """
    bits_pm1 = np.asarray(bits_pm1)
    return np.where(bits_pm1 == -1, z_minus, z_plus)


def build_phase_state_from_bits(bits_pm1: np.ndarray,
                                z_minus: complex,
                                z_plus: complex) -> QuantumCircuit:
    """
    Construye el estado |psi_x> a partir de un vector binario usando DiagonalGate
    con los complejos aprendidos.
    """
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
    """
    Construye circuito SWAP test completo para dos vectores binarios x,y en {-1,1}^m.
    """
    if len(bits_x) != len(bits_y):
        raise ValueError("bits_x y bits_y deben tener misma longitud")

    m = len(bits_x)
    n = int(math.log2(m))
    if 2**n != m:
        raise ValueError("La longitud de los vectores debe ser potencia de 2.")

    # Fases -> complejos
    z_minus, z_plus = phases_to_complex(phi_minus, phi_plus)

    # Estados de fase para x e y
    qc_x = build_phase_state_from_bits(bits_x, z_minus, z_plus)
    qc_y = build_phase_state_from_bits(bits_y, z_minus, z_plus)

    # Registros: ancilla + n qubits para x + n qubits para y
    anc = QuantumRegister(1, "anc")
    qx = QuantumRegister(n, "x")
    qy = QuantumRegister(n, "y")
    c  = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, qx, qy, c)

    # Preparar estados
    qc = qc.compose(qc_x, qubits=qx)
    qc = qc.compose(qc_y, qubits=qy)

    # SWAP test estándar
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
    """
    Ejecuta el SWAP test en AerSimulator y devuelve p0 (freq de medir 0 en la ancilla).
    """
    if backend is None:
        backend = AerSimulator()  # puedes fijar seed aquí si quieres

    qc = build_swap_test_circuit(bits_x, bits_y, phi_minus, phi_plus)
    qc = qc.copy()  # por si acaso
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    n0 = counts.get('0', 0)
    n1 = counts.get('1', 0)
    total = n0 + n1
    if total == 0:
        return 0.5  # degenerado, no debería pasar
    return n0 / total


def true_cos_from_bits(bits_x: np.ndarray, bits_y: np.ndarray) -> float:
    """
    Coseno 'real' entre dos vectores binarios {-1,+1}^m,
    usando la fórmula estándar <x,y> / (||x|| ||y||).
    """
    x = np.asarray(bits_x, float)
    y = np.asarray(bits_y, float)
    dot = np.dot(x, y)
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)
    if normx == 0 or normy == 0:
        return 0.0
    return dot / (normx * normy)


def p0_target_from_bits(bits_x: np.ndarray, bits_y: np.ndarray) -> float:
    """
    Define p0_real a partir del coseno real. Aquí uso:
        p0_real = (1 + cos(x,y)) / 2
    que es un mapping razonable en [0,1].
    Puedes cambiarlo si quieres imitar AE-SWAP u otra cosa.
    """
    cos_xy = true_cos_from_bits(bits_x, bits_y)
    return 0.5 * (1.0 + cos_xy)
