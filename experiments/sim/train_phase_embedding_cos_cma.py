# experiments/sim/train_phase_embedding_cos_cma.py
#
# Entrenamiento CMA-ES del MLP(cos) pequeño y estable (HIDDEN=8).
# MISMA API / mismo nombre de fichero.

import numpy as np
import cma

from qiskit_aer import AerSimulator
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.quantum.phase_embedding_cma import (
    run_swap_test_p0,
    true_cos_from_bits,
    p0_target_from_bits,
    phases_from_cos,
    get_num_params_cos_mlp,
)


# ============================================================
#   Dataset: mezcla discreto + refuerzo cerca de 0
# ============================================================

def make_pair_with_target_cos(dim: int, target_cos: float, rng: np.random.Generator):
    """
    Genera x,y en {-1,+1}^dim con cos(x,y) ≈ target_cos.
    """
    x = rng.choice([-1, 1], size=dim)

    num_equal = int((1.0 + target_cos) / 2.0 * dim)
    num_equal = max(0, min(dim, num_equal))
    num_diff = dim - num_equal

    y = np.empty(dim, dtype=int)
    idx = rng.permutation(dim)
    eq_idx = idx[:num_equal]
    df_idx = idx[num_equal:]

    y[eq_idx] = x[eq_idx]
    y[df_idx] = -x[df_idx]

    return x, y


def generate_training_dataset(dim: int = 256,
                              n_pairs: int = 300,
                              rng_seed: int = 123):
    """
    Dataset con cosenos:
      - principalmente en la rejilla [-1,-0.8,...,1]
      - un 20% de muestras extra en [-0.2,0.2] (zona difícil)
    """
    rng = np.random.default_rng(rng_seed)
    cos_grid = np.linspace(-1.0, 1.0, num=11)  # -1, -0.8, ..., 1

    X_list = []
    Y_list = []

    for _ in range(n_pairs):
        if rng.random() < 0.2:
            # refuerzo cerca de 0
            c = rng.uniform(-0.2, 0.2)
        else:
            c = rng.choice(cos_grid)

        x, y = make_pair_with_target_cos(dim, c, rng)
        X_list.append(x)
        Y_list.append(y)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y


# ============================================================
#   Función objetivo + L2 suave
# ============================================================

def evaluate_params(params, X, Y, backend, shots=4096, l2=1e-4):
    """
    Función objetivo para CMA-ES:
      MSE(p0_real, p0_quant) + L2 suave sobre params.
    """
    N = X.shape[0]
    sq_errors = []

    for i in range(N):
        bits_x = X[i]
        bits_y = Y[i]

        cos_xy = true_cos_from_bits(bits_x, bits_y)
        phi_minus, phi_plus = phases_from_cos(cos_xy, params)

        p0_real = p0_target_from_bits(bits_x, bits_y)

        p0_quant = run_swap_test_p0(
            bits_x,
            bits_y,
            phi_minus,
            phi_plus,
            shots=shots,
            backend=backend,
        )

        sq_errors.append((p0_real - p0_quant) ** 2)

    mse = float(np.mean(sq_errors))
    reg = l2 * float(np.sum(params * params))
    return mse + reg


# ============================================================
#   Entrenamiento CMA-ES
# ============================================================

def main():
    dim = 128          # dimensión de entrenamiento (parecida a tus casos reales)
    n_pairs = 300      # nº de pares de entrenamiento
    shots = 4096

    print(f"Generando dataset de entrenamiento: dim={dim}, n_pairs={n_pairs}")
    X, Y = generate_training_dataset(dim=dim, n_pairs=n_pairs, rng_seed=123)

    backend = AerSimulator()

    n_params = get_num_params_cos_mlp()
    print(f"Número de parámetros del MLP(cos): {n_params}")

    rng = np.random.default_rng(42)
    x0 = rng.normal(loc=0.0, scale=0.1, size=n_params)
    sigma0 = 0.3

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': 10,
        'maxiter': 60,
        'seed': 1234,
    })

    def objective(theta_vec):
        return evaluate_params(theta_vec, X, Y, backend=backend, shots=shots, l2=1e-4)

    print("Comenzando optimización CMA-ES (MLP(cos) estable)...")
    while not es.stop():
        solutions = es.ask()
        values = [objective(s) for s in solutions]
        es.tell(solutions, values)
        es.disp()

    res = es.result
    best_params = res.xbest
    best_mse = res.fbest

    print("\n=== RESULTADO FINAL (MLP(cos) estable) ===")
    print(f"Mejor MSE = {best_mse:.6e}")
    print("Parámetros [len = {}]".format(len(best_params)))
    print(best_params)

    np.save("learned_phase_params_cos.npy", np.array(best_params, dtype=float))
    print("Parámetros guardados en learned_phase_params_cos.npy")


if __name__ == "__main__":
    main()
