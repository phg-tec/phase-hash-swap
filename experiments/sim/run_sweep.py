#!/usr/bin/env python3
# experiments/sim/run_sweep.py
# =========================================================
# Generic sweep runner for AE-SWAP / Phase-Hash SWAP
# Reads a JSON config and outputs a CSV.
# =========================================================

import json
from pathlib import Path
import pandas as pd

import os,sys
import math
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.quantum.swap_core import (
    make_pair_with_cosine,
    simhash_classical_corr,
    run_swap_amp,
    run_swap_phasehash,
    corr_to_cos,
)
from src.quantum.swap_core import run_swap_phasehash

def main(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Required fields (assumed present in your configs)
    dims = cfg["dims"]
    cos_targets = cfg["cos_targets"]
    shots_list = cfg["shots_list"]
    m_list = cfg.get("m_list", [])
    E_list = cfg.get("E_list", [])
    reps = int(cfg.get("reps", 1))
    out_csv = cfg["output_csv"]

    include_classical = bool(cfg.get("include_classical", False))
    measure_cost = bool(cfg.get("measure_cost", False))
    opt_level = int(cfg.get("opt_level", 3))
    base_seed0 = int(cfg.get("base_seed0", 0))

    rows = []
    dims = [256]
    m_list = [128]
    E_list = [64]
    cos_targets = [0.0, 0.25, 0.5, 0.75, 1]
    reps = 1
    include_classical = False
    for rep in range(reps):
        for dim in dims:
            base_seed = base_seed0 + 1000*rep + dim
            for true_cos in cos_targets:
                x, y = make_pair_with_cosine(dim, true_cos, seed=base_seed)

                for shots in shots_list:

                    # --- Classical SimHash baseline ---
                    if include_classical:
                        for m in m_list:
                            r_class, cos_class = simhash_classical_corr(
                                x, y, m=m, seed=base_seed
                            )
                            rows.append([
                                "SimHash-classical", rep, dim, true_cos, m, "-",
                                shots, cos_class, abs(cos_class - true_cos),
                                0.0, "-", "-"
                            ])

                    # --- AE SWAP ---
                    overlap_amp, t_amp, d_amp, q_amp = (None, None, "-", "-")
                    if measure_cost:
                        overlap_amp, t_amp, d_amp, q_amp = run_swap_amp(
                            x, y, shots=shots, seed=base_seed,
                            opt_level=opt_level, measure_cost=True
                        )
                    else:
                        i=0
                        overlap_amp, t_amp, p0_ae = run_swap_amp(
                            x, y, shots=shots, seed=base_seed,
                            opt_level=opt_level, measure_cost=False
                        )

                    cos_amp = corr_to_cos(overlap_amp)
                    rows.append([
                        "AE-SWAP", rep, dim, true_cos, "-", "-",
                        shots, cos_amp, abs(cos_amp - true_cos),
                        t_amp, d_amp, q_amp
                    ])

                    # --- Phase-Hash SWAP ---
                    for m in m_list:
                        for E in E_list:
                            if measure_cost:
                                cos_ph, t_ph, d_ph, q_ph = run_swap_phasehash(
                                    x, y, m=m, E=E, shots=shots,
                                    seed=base_seed + m + E,
                                    opt_level=opt_level,
                                    measure_cost=True
                                )
                            else:


                                cos_ph, t_ph, p0_pes = run_swap_phasehash(
                                    x, y,
                                    m=m, E=E,
                                    shots=shots,
                                    seed=base_seed + m + E,
                                    measure_cost=False
                                )
                                d_ph, q_ph = "-", "-"
                            rows.append([
                                "Phase-Hash (SWAP)", rep, dim, true_cos, m, E,
                                shots, cos_ph, abs(cos_ph - true_cos),
                                t_ph, d_ph, q_ph
                            ])
                    p0_real=(1+true_cos**2)/2
                    print(f"p0_real={p0_real} | ",
                          f"p0_pes={p0_pes} | ",
                          f"p0_ae={p0_ae} | ",
                          f"cos_real={true_cos} | "
                          f"cos_pes={math.sqrt(max(0,2*p0_pes -1))} | "
                          f"cos_ae={math.sqrt(max(0,2*p0_ae -1))} | ",
                          f"mae_cos={abs(true_cos - math.sqrt(max(0,2*p0_pes -1)))}"
                          f"\n (+bueno, -malo)={abs(p0_real-p0_ae) - abs(p0_real-p0_pes)}"
                    )

    df = pd.DataFrame(rows, columns=[
        "Method", "Rep", "Dim", "TrueCos", "m", "E", "Shots",
        "EstCos", "AbsErr", "Time_s", "Depth", "TwoQ"
    ])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Sweep guardado: {out_csv} ({len(df)} filas)")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python experiments/sim/run_sweep.py experiments/sim/config_lowdim.json")
    main(sys.argv[1])
