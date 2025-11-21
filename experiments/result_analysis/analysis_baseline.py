import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/baseline_reps.csv")

methods = ["SimHash-classical", "Phase-Hash (SWAP)", "AE-SWAP"]

plt.figure(figsize=(6,4))

for method in methods:
    sub = df[df["Method"] == method]
    grouped = sub.groupby("TrueCos").agg(
        mean=("EstCos", "mean"),
        std=("EstCos", "std")
    )
    plt.errorbar(
        grouped.index,
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        capsize=4,
        label=method
    )

plt.plot([0.2,0.5,0.7,0.9], [0.2,0.5,0.7,0.9], linestyle="--")  # línea identidad
plt.xlabel("True cosine")
plt.ylabel("Estimated cosine (mean ± std)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_stability.pdf")
plt.show()
