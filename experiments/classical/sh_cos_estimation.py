import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ============================================================
#                 SIMHASH CORE FUNCTIONS
# ============================================================

def simhash(x, m, rng):
    """
    Compute m-bit SimHash for vector x.
    """
    n = len(x)
    R = rng.normal(size=(m, n))
    proj = R @ x
    return (proj >= 0).astype(int)


def estimate_cos_from_hash(hx, hy):
    """
    Classical SimHash cosine estimation.
    """
    m = len(hx)
    p = np.mean(hx == hy)
    theta = np.pi * (1 - p)
    return np.cos(theta)


def generate_unit_vector(n, rng):
    x = rng.normal(size=n)
    return x / np.linalg.norm(x)


# ============================================================
#        GENERATE VALID (n, m) COMBINATIONS
# ============================================================

def generate_valid_pairs():
    n_list = [64, 128, 256, 512, 1024]
    m_list = [16, 32, 64, 128, 256, 512]

    valid_pairs = []

    for n in n_list:
        for m in m_list:
            if m == n // 2 or m == n // 4:
                valid_pairs.append((n, m))

    return valid_pairs


# ============================================================
#               RUN EXPERIMENT FOR EACH (n, m)
# ============================================================

def run_experiment_for_pair(n, m, N=10000, seed=123):
    rng = np.random.default_rng(seed)

    true_cos = []
    est_cos = []
    errors = []

    for _ in range(N):
        x = generate_unit_vector(n, rng)
        y = generate_unit_vector(n, rng)

        c = x @ y
        hx = simhash(x, m, rng)
        hy = simhash(y, m, rng)
        c_hat = estimate_cos_from_hash(hx, hy)
        err = abs(c_hat - c)

        true_cos.append(c)
        est_cos.append(c_hat)
        errors.append(err)

    return np.array(true_cos), np.array(est_cos), np.array(errors)


# ============================================================
#                    MAIN EXPERIMENT
# ============================================================

def run_all_experiments(N=10000):

    valid_pairs = generate_valid_pairs()
    print("\nValid (n,m) combinations:")
    print(valid_pairs)

    os.makedirs("data/raw/classical", exist_ok=True)
    os.makedirs("data/figs", exist_ok=True)

    mae_table = []

    scatter_global = []  # For combined scatter plot

    for n, m in tqdm(valid_pairs, desc="Running all (n,m) experiments"):

        tc, ec, err = run_experiment_for_pair(n, m, N=N)

        # Save CSV
        df = pd.DataFrame({
            "true_cos": tc,
            "est_cos": ec,
            "error": err,
            "n": n,
            "m": m
        })
        filename = f"data/raw/classical/simhash_scatter_n{n}_m{m}.csv"
        df.to_csv(filename, index=False)

        # Append to global MAE table
        mae = np.mean(err)
        mae_table.append([n, m, mae])

        # Add to global scatter buffer
        scatter_global.append((n, m, tc, err))

    # Save global MAE table
    df_mae = pd.DataFrame(mae_table, columns=["n", "m", "MAE"])
    df_mae.to_csv("data/raw/classical/simhash_mae_all.csv", index=False)

    return scatter_global, df_mae


# ============================================================
#                          PLOTTING
# ============================================================

def plot_global_scatter(scatter_global):
    plt.figure(figsize=(10, 7))

    for (n, m, tc, err) in scatter_global:
        label = f"n={n}, m={m}"
        plt.scatter(tc, err, s=5, alpha=0.3, label=label)

    plt.xlabel("True cosine")
    plt.ylabel("Absolute error")
    plt.title("SimHash Error vs True Cosine (All Valid (n,m) Pairs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figs/classical/global_scatter.png", dpi=300)
    plt.show()


def plot_mae_vs_dimension(df_mae):
    plt.figure(figsize=(10, 6))

    for m in sorted(df_mae["m"].unique()):
        df_m = df_mae[df_mae["m"] == m]
        plt.plot(df_m["n"], df_m["MAE"], marker="o", label=f"m={m}")

    plt.xlabel("Dimension n")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("SimHash MAE vs Dimension for Valid m")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/classical/mae_vs_dimension.png", dpi=300)
    plt.show()


def plot_mae_heatmap(df_mae):
    pivot = df_mae.pivot(index="n", columns="m", values="MAE")

    plt.figure(figsize=(9, 6))
    plt.imshow(pivot, cmap="viridis", aspect="auto")
    plt.colorbar(label="MAE")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel("m")
    plt.ylabel("n")
    plt.title("Heatmap of SimHash MAE for (n,m)")
    plt.tight_layout()
    plt.savefig("figs/classical/mae_heatmap.png", dpi=300)
    plt.show()


# ============================================================
#                           MAIN
# ============================================================

if __name__ == "__main__":

    N = 20000  # Number of samples per pair

    scatter_global, df_mae = run_all_experiments(N=N)

    # PLOTS
    plot_global_scatter(scatter_global)
    plot_mae_vs_dimension(df_mae)
    plot_mae_heatmap(df_mae)

    print("\nAll experiments complete!")
    print("CSV files in: data/raw/classical/")
    print("Figures in: data/figs/classical/")
