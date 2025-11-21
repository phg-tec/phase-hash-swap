#!/usr/bin/env python3
# ============================================================
#  Dual-regime analysis (LOW vs HIGH dimensionality)
#  Q1 Figures – PES-SWAP uses:
#     LOW DIM → all (m,E)
#     HIGH DIM → best configuration m=256, E=32
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

LOW_CSV  = "data/raw/compare_swap_lowdim.csv"
HIGH_CSV = "data/raw/compare_swap_highdim.csv"
OUT_DIR  = Path("data/figs/q1_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load datasets
# ============================================================
df_low  = pd.read_csv(LOW_CSV)
df_high = pd.read_csv(HIGH_CSV)

df_low["Regime"]  = "low"
df_high["Regime"] = "high"

print("Detected methods:", set(df_low["Method"]) | set(df_high["Method"]))
print("Columns:", df_low.columns.tolist())

HAS_DEPTH = "Depth" in df_low.columns and "Depth" in df_high.columns
HAS_TWOQ  = "TwoQ"  in df_low.columns and "TwoQ"  in df_high.columns

# ============================================================
# Filter logic:
#   LOW DIM  → keep everything (calibration)
#   HIGH DIM → only PES-SWAP with m=256, E=32
# ============================================================

BEST_M = "256"
BEST_E = "32"

# LOW DIM remains untouched
df_low_clean = df_low.copy()

# HIGH DIM:
df_high_ae = df_high[df_high["Method"] == "AE-SWAP"].copy()

df_high_pes = df_high[
    (df_high["Method"] == "Phase-Hash (SWAP)") &
    (df_high["m"] == BEST_M) &
    (df_high["E"] == BEST_E)
].copy()

# Merge final dataset
df = pd.concat([df_low_clean, df_high_ae, df_high_pes], ignore_index=True)

print("\n>>> FILTER APPLIED:")
print("   LOW DIM  → PES-SWAP: all m,E")
print(f"   HIGH DIM → PES-SWAP: only m={BEST_M}, E={BEST_E}")
print(f"   Final dataset size = {len(df)} rows\n")


# ============================================================
# Cosine band assignment
# ============================================================
bands   = [0.0, 0.3, 0.6, 0.8, 1.0]
labels  = ["[0.0,0.3]", "(0.3,0.6]", "(0.6,0.8]", "(0.8,1.0]"]

df["cos_band"] = pd.cut(
    np.abs(df["TrueCos"]),
    bins=bands,
    labels=labels,
    include_lowest=True
)

# ============================================================
# FIGURE 1 – MAE vs shots (low-Dim calibration)
# ============================================================
df_low_shots = df[df["Regime"]=="low"].groupby(["Method","Shots"]).agg(
    mae_mean=("AbsErr","mean")
).reset_index()

plt.figure(figsize=(6,4))
for method in df_low_shots["Method"].unique():
    sub = df_low_shots[df_low_shots["Method"] == method]
    if len(sub["Shots"].unique()) > 1:
        plt.plot(sub["Shots"], sub["mae_mean"], marker="o", label=method)

plt.xlabel("Shots")
plt.ylabel("MAE")
plt.title("Calibration curve: MAE vs shots (Low-dimensional regime)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "F_shots_calibration.png", dpi=300)
plt.close()


# ============================================================
# FIGURES 2 & 3 – MAE per cosine band (LOW / HIGH)
# ============================================================
for regime, tag in [("low","LOW"), ("high","HIGH")]:

    sub = df[df["Regime"] == regime]

    agg = sub.groupby(["cos_band","Method"]).agg(
        mae_mean=("AbsErr","mean")
    ).reset_index()

    plt.figure(figsize=(7,4))
    width = 0.35
    x = np.arange(len(labels))

    for i, method in enumerate(["AE-SWAP", "Phase-Hash (SWAP)"]):
        vals = [
            agg[(agg["cos_band"]==band) & (agg["Method"]==method)]["mae_mean"].values
            for band in labels
        ]
        vals = [v[0] if len(v)>0 else np.nan for v in vals]

        plt.bar(x + i*width - width/2, vals, width=width, label=method)

    plt.xticks(x, labels)
    plt.ylabel("MAE")
    plt.title(f"MAE per cosine band – {tag} regime")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"F_mae_cosband_{tag}.png", dpi=300)
    plt.close()


# ============================================================
# FIGURE 4 – MAE vs dimension (4096 shots)
# ============================================================
if "Dim" in df.columns:

    df_4096 = df[df["Shots"] == 4096]

    agg_dim = df_4096.groupby(["Dim","Regime","Method"]).agg(
        mae_mean=("AbsErr","mean")
    ).reset_index()

    plt.figure(figsize=(7,4))

    for method in ["AE-SWAP", "Phase-Hash (SWAP)"]:
        sub_m = agg_dim[agg_dim["Method"] == method]

        low_sub  = sub_m[sub_m["Regime"]=="low"]
        high_sub = sub_m[sub_m["Regime"]=="high"]

        if len(low_sub) > 0:
            plt.plot(low_sub["Dim"], low_sub["mae_mean"],
                     marker="o", linestyle="--",
                     label=f"{method} (low)")

        if len(high_sub) > 0:
            plt.plot(high_sub["Dim"], high_sub["mae_mean"],
                     marker="o", linestyle="-",
                     label=f"{method} (high)")

    plt.xlabel("Dimension d")
    plt.ylabel("MAE")
    plt.title("MAE vs dimension (4096 shots)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "F_mae_vs_dim_low_high_4096.png", dpi=300)
    plt.close()


# ============================================================
# FIGURE 5 – Time vs dimension (HIGH)
# ============================================================
sub_high = df[df["Regime"]=="high"]

agg_time = sub_high.groupby(["Dim","Method"]).agg(
    time_mean=("Time_s","mean")
).reset_index()

plt.figure(figsize=(6,4))
for method in agg_time["Method"].unique():
    sub = agg_time[agg_time["Method"]==method]
    plt.plot(sub["Dim"], sub["time_mean"], marker="o", label=method)

plt.xlabel("Dimension d")
plt.ylabel("Time (s)")
plt.title("Execution time vs dimension (high-dimensional regime)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "F_time_vs_dim_HIGH.png", dpi=300)
plt.close()


print("\n✓ Final Q1 figures generated in:", OUT_DIR)
