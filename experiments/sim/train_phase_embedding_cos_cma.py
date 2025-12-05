# ============================================================
#  train_phase_embedding_cos_cma.py (versión mejorada)
# ============================================================

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
#   Dataset mejorado
# ============================================================

def make_pair_with_target_cos(dim: int, target_cos: float, rng):
    """
    Genera x,y en {-1,+1}^dim con cos(x,y) ≈ target_cos.
    Distribución aleatoria continua del cos: mucho mejor para entrenar.
    """
    x = rng.choice([-1, 1], size=dim)

    # número de iguales/diferentes según target_cos
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


def generate_training_dataset(dim=128, n_pairs=400, rng_seed=123):
    """
    Dataset mixto y continuo:
        - cos ~ Uniform(-1,1)
        - más densidad en [-0.2,0.2] (zona difícil)
    """
    rng = np.random.default_rng(rng_seed)

    X_list = []
    Y_list = []

    for _ in range(n_pairs):
        # 70% uniforme en [-1,1], 30% más denso cerca de 0
        if rng.random() < 0.7:
            c = rng.uniform(-1.0, 1.0)
        else:
            c = rng.uniform(-0.2, 0.2)

        x, y = make_pair_with_target_cos(dim, c, rng)
        X_list.append(x)
        Y_list.append(y)

    return np.stack(X_list), np.stack(Y_list)


# ============================================================
#   Función objetivo + regularización
# ============================================================

def evaluate_params(params, X, Y, backend, shots=4096, l2=1e-4, noise=0.0):
    """
    MSE + regularización L2 + ruido opcional.
    """
    N = X.shape[0]
    err = 0.0

    for i in range(N):
        bits_x = X[i]
        bits_y = Y[i]

        cos_xy = true_cos_from_bits(bits_x, bits_y)

        phi_minus, phi_plus = phases_from_cos(cos_xy, params)

        p0_real = p0_target_from_bits(bits_x, bits_y)
        p0_quant = run_swap_test_p0(bits_x, bits_y, phi_minus, phi_plus,
                                    shots=shots, backend=backend)

        if noise > 0:
            p0_quant += np.random.normal(0, noise)

        err += (p0_real - p0_quant)**2

    mse = err / N
    reg = l2 * np.sum(params * params)
    return mse + reg


# ============================================================
#  ENTRENAMIENTO CMA MEJORADO
# ============================================================

def main():
    dim = 128
    n_pairs = 400
    shots = 4096

    print(f"Generando dataset: dim={dim}, n_pairs={n_pairs}")
    X, Y = generate_training_dataset(dim=dim, n_pairs=n_pairs)

    backend = AerSimulator()

    n_params = get_num_params_cos_mlp()
    print(f"MLP parameters = {n_params}")

    rng = np.random.default_rng(42)
    x0 = rng.normal(0.0, 0.1, size=n_params)

    es = cma.CMAEvolutionStrategy(
        x0,
        0.2,
        {
            'popsize': 20,
            'maxiter': 80,
            'seed': 123,
        },
    )

    def objective(theta_vec):
        return evaluate_params(theta_vec, X, Y, backend,
                               shots=shots, l2=1e-4, noise=0.005)

    print("Entrenando MLP(cos) con CMA-ES mejorado...")
    while not es.stop():
        sols = es.ask()
        vals = [objective(s) for s in sols]
        es.tell(sols, vals)
        es.disp()

    res = es.result
    best = res.xbest
    print("\n=== FINAL RESULT ===")
    print("MSE =", res.fbest)

    np.save("learned_phase_params_cos.npy", best)
    print("Saved to learned_phase_params_cos.npy")


if __name__ == "__main__":
    main()
