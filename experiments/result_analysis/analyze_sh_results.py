import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Ajustes visuales
plt.style.use("seaborn-v0_8-darkgrid")

# =======================================================
#                 CARGA DE ARCHIVOS
# =======================================================

def load_all_scatter_csv():
    files = glob("data/raw/classical/simhash_scatter_n*_m*.csv")
    dfs = []

    for f in files:
        df = pd.read_csv(f)
        # deduce n and m from filename
        basename = os.path.basename(f)
        parts = basename.replace(".csv", "").split("_")
        n = int(parts[2][1:])
        m = int(parts[3][1:])
        df["n"] = n
        df["m"] = m
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def load_mae_table():
    return pd.read_csv("data/raw/classical/simhash_mae_all.csv")


# =======================================================
#                       PLOTS
# =======================================================

def plot_global_scatter(df):
    plt.figure(figsize=(10, 7))
    for m in sorted(df["m"].unique()):
        sub = df[df["m"] == m]
        plt.scatter(sub["true_cos"], sub["error"], s=5, alpha=0.25, label=f"m={m}")

    plt.xlabel("True cosine")
    plt.ylabel("Absolute error")
    plt.title("Global Error vs Cosine (All n,m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/figs/classical/plot_global_scatter.png", dpi=300)
    plt.show()


def plot_scatter_by_m(df):
    m_vals = sorted(df["m"].unique())
    rows = len(m_vals)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 3*rows))

    for i, m in enumerate(m_vals):
        ax = axes[i] if rows > 1 else axes
        sub = df[df["m"] == m]
        ax.scatter(sub["true_cos"], sub["error"], s=5, alpha=0.3)
        ax.set_title(f"Error vs Cosine for m={m}")
        ax.set_ylabel("Error")

    axes[-1].set_xlabel("True cosine")
    plt.tight_layout()
    plt.savefig("data/figs/classical/plot_scatter_by_m.png", dpi=300)
    plt.show()


def plot_scatter_by_n(df):
    n_vals = sorted(df["n"].unique())
    rows = len(n_vals)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 3*rows))

    for i, n in enumerate(n_vals):
        ax = axes[i] if rows > 1 else axes
        sub = df[df["n"] == n]
        ax.scatter(sub["true_cos"], sub["error"], s=5, alpha=0.3)
        ax.set_title(f"Error vs Cosine for n={n}")
        ax.set_ylabel("Error")

    axes[-1].set_xlabel("True cosine")
    plt.tight_layout()
    plt.savefig("data/figs/classical/plot_scatter_by_n.png", dpi=300)
    plt.show()


def plot_mae_vs_n(df_mae):
    plt.figure(figsize=(10, 7))

    for m in sorted(df_mae["m"].unique()):
        sub = df_mae[df_mae["m"] == m]
        plt.plot(sub["n"], sub["MAE"], marker="o", label=f"m={m}")

    plt.xlabel("Dimension n")
    plt.ylabel("MAE")
    plt.title("MAE vs Dimension for Each m")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/figs/classical/plot_mae_vs_n.png", dpi=300)
    plt.show()


def plot_mae_vs_m(df_mae):
    plt.figure(figsize=(10, 7))

    for n in sorted(df_mae["n"].unique()):
        sub = df_mae[df_mae["n"] == n]
        plt.plot(sub["m"], sub["MAE"], marker="o", label=f"n={n}")

    plt.xlabel("m (SimHash bits)")
    plt.ylabel("MAE")
    plt.title("MAE vs m for Each n")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/figs/classical/plot_mae_vs_m.png", dpi=300)
    plt.show()


def plot_mae_heatmap(df_mae):
    pivot = df_mae.pivot(index="n", columns="m", values="MAE")

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot, cmap="viridis", aspect="auto")
    plt.colorbar(label="MAE")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel("m (SimHash bits)")
    plt.ylabel("n (Dimension)")
    plt.title("MAE Heatmap for (n,m) Pairs")
    plt.tight_layout()
    plt.savefig("data/figs/classical/plot_mae_heatmap.png", dpi=300)
    plt.show()


# =======================================================
#                         MAIN
# =======================================================

if __name__ == "__main__":

    os.makedirs("figs", exist_ok=True)

    print("\nLoading CSVs...")
    df = load_all_scatter_csv()
    df_mae = load_mae_table()

    print("\nGenerating figures...")

    plot_global_scatter(df)
    plot_scatter_by_m(df)
    plot_scatter_by_n(df)
    plot_mae_vs_n(df_mae)
    plot_mae_vs_m(df_mae)
    plot_mae_heatmap(df_mae)

    print("\nAll figures saved in 'data/figs/classical/'")
