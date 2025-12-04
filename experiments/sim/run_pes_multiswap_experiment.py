# experiments/sim/run_pes_multiswap_experiment.py
#!/usr/bin/env python
"""
PES-multilevel SWAP: AE-SWAP vs PES-MULTISWAP (fases continuas)
----------------------------------------------------------------

Incluye:

  - AE-SWAP (amplitude encoding).
  - PES-MULTISWAP estratificado por grupos (p,q) usando fases continuas
    φ = α * valor_discretizado (k-medias sobre x,y).

Se comparan los MAE de cada método respecto al coseno "real" continuo.
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.basic import make_pair_with_cosine, cos_sim
from src.utils.discretization import learn_kmeans_centers
from src.quantum.ae_swap import run_ae_swap
from src.quantum.pes_multiswap import run_pes_multiswap_phase


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AE-SWAP vs PES-MULTISWAP (fases continuas)."
    )
    parser.add_argument("--dim", type=int, default=256,
                        help="Dimensión de los vectores x,y.")
    parser.add_argument("--K", type=int, default=4,
                        help="Número de centros K para k-means.")
    parser.add_argument("--shots", type=int, default=2048,
                        help="Número de shots para los SWAP tests.")
    parser.add_argument("--alpha", type=float, default=0.0005,
                        help="Factor de escala de fase φ = α * valor.")
    parser.add_argument("--seed0", type=int, default=123,
                        help="Semilla base para generar pares (x,y).")

    args = parser.parse_args()

    dim = args.dim
    K = args.K
    shots = args.shots
    alpha = args.alpha
    seed = args.seed0
    
    rhos = [0.9, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -0.9]

    print("\n=== AE-SWAP vs PES-MULTISWAP ===\n")
    #print("Real | classic_disc | quantum_disc | MAE_class | MAE_quant | Δ(class-quant) | t_AE | t_PES | t_pre")
    print("P0_Real | P0_Quanutm | Cos_Real | Cos_quantum")
    for i, rho in enumerate(rhos):
        seed+=i
        x, y = make_pair_with_cosine(dim, rho, seed)
        cos_real = cos_sim(x, y)
        p0_real = (1 + cos_real**2)/2
        centers = learn_kmeans_centers(x, y, K=K, seed=seed)

        # --- AE-SWAP ---
        #cos_ae, t_ae = run_ae_swap(x, y, shots=shots, seed=seed)
        #mae_ae = abs(cos_real - cos_ae)
        mae_ae, cos_ae, t_ae = 0,0,0
        # --- PES-MULTISWAP ---
        res = run_pes_multiswap_phase(
            x, y, centers,
            alpha=alpha,
            shots=shots,
            seed=seed,
            verbose=False
        )

        cos_real = res["cos_real"]
        cos_quantum = res["cos_quantum"]
        cos_mae = res["cos_mae"]
        p0_global = res["p0_global"]
        

        #print(
        #      f"real={p0:+.3f} | "
        #      f"classic_disc={cos_classic:+.3f} | "
        #      f"quantum_disc={cos_pes_ms:+.3f} | "
        #      f"MAE_class={mae_classic:.3f} | "
        #      f"MAE_quant={mae_ms:.3f} | "
        #      f"Δ(class-q)={diff_classic_quantum:.3f} | "
        #      f"t_AE={t_ae:.4f} | t_PES={t_pes:.4f} | t_pre={t_pre:.4f}"
        #)
        print(
            f"cos_real={cos_real:+.3f} | "
            f"cos_quant={cos_quantum:+.3f} | "
            f"p0_real={p0_real:+.3f} | "
            f"p0_quantum={p0_global:+.3f} | "
            f"cos_MAE={cos_mae:.3f} | "
            f"p0_MAE={abs(p0_real-p0_global):.3f}"
        )




if __name__ == "__main__":
    main()
