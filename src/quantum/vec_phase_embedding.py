# src/quantum/vec_phase_embedding.py

import numpy as np

# Dimensión fija para este modelo (m = 32)
M = 128
IN_DIM = 2 * M   # concatenamos a y b
H1 = 16
H2 = 8

# Número total de parámetros del MLP
# Layer1: W1 (H1 x IN_DIM) + b1 (H1)
# Layer2: W2 (H2 x H1) + b2 (H2)
# Layer3: W3 (2 x H2) + b3 (2)
N_PARAMS = H1 * IN_DIM + H1 + H2 * H1 + H2 + 2 * H2 + 2

def phases_to_complex(phi_minus: float, phi_plus: float):
    """
    Convierte dos fases reales en complejos unitarios z_minus, z_plus.
    """
    z_minus = np.cos(phi_minus) + 1j * np.sin(phi_minus)
    z_plus  = np.cos(phi_plus)  + 1j * np.sin(phi_plus)
    return z_minus, z_plus


def get_num_params():
    """Devuelve el número de parámetros del modelo."""
    return N_PARAMS


def unpack_params(theta: np.ndarray):
    """
    Desempaqueta el vector de parámetros 'theta' en
    W1, b1, W2, b2, W3, b3.
    """
    theta = np.asarray(theta, float)
    if theta.size != N_PARAMS:
        raise ValueError(f"theta.size = {theta.size}, esperado = {N_PARAMS}")

    offset = 0

    # W1: (H1, IN_DIM)
    W1_size = H1 * IN_DIM
    W1 = theta[offset:offset + W1_size].reshape(H1, IN_DIM)
    offset += W1_size

    # b1: (H1,)
    b1 = theta[offset:offset + H1]
    offset += H1

    # W2: (H2, H1)
    W2_size = H2 * H1
    W2 = theta[offset:offset + W2_size].reshape(H2, H1)
    offset += W2_size

    # b2: (H2,)
    b2 = theta[offset:offset + H2]
    offset += H2

    # W3: (2, H2)
    W3_size = 2 * H2
    W3 = theta[offset:offset + W3_size].reshape(2, H2)
    offset += W3_size

    # b3: (2,)
    b3 = theta[offset:offset + 2]
    offset += 2

    if offset != N_PARAMS:
        raise RuntimeError("Error interno en el desempaquetado de parámetros.")

    return W1, b1, W2, b2, W3, b3


def mlp_forward_phases(bits_x: np.ndarray,
                       bits_y: np.ndarray,
                       theta: np.ndarray):
    """
    bits_x, bits_y: vectores binarios en {-1,+1} de longitud M=32.
    theta: vector de parámetros del MLP (longitud N_PARAMS).

    Devuelve:
      phi_minus, phi_plus (escalares float)
    """
    bits_x = np.asarray(bits_x, float)
    bits_y = np.asarray(bits_y, float)

    if bits_x.shape != (M,) or bits_y.shape != (M,):
        raise ValueError(f"bits_x y bits_y deben tener shape ({M},)")

    # Entrada al MLP: concat(a,b) -> shape (2M,)
    inp = np.concatenate([bits_x, bits_y], axis=0)  # (64,)

    W1, b1, W2, b2, W3, b3 = unpack_params(theta)

    # Layer 1: ReLU(W1 @ inp + b1)
    h1 = W1 @ inp + b1
    h1 = np.maximum(h1, 0.0)  # ReLU

    # Layer 2: ReLU(W2 @ h1 + b2)
    h2 = W2 @ h1 + b2
    h2 = np.maximum(h2, 0.0)

    # Layer 3: salida lineal (2,)
    out = W3 @ h2 + b3  # [phi_minus, phi_plus]

    phi_minus = out[0]
    phi_plus = out[1]

    # Opcional: envolver fases a [-pi, pi] para estabilidad
    phi_minus = (phi_minus + np.pi) % (2 * np.pi) - np.pi
    phi_plus  = (phi_plus + np.pi) % (2 * np.pi) - np.pi

    return phi_minus, phi_plus
