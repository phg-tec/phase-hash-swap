# experiments/sim/train_phase_embedding.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import os,sys


# Ajusta la ruta raíz si hace falta
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.learning.dataset_phase_embedding import PhaseEmbeddingCosDataset
from src.learning.phase_embedding import PhaseEmbeddingCosModel

# ------------------------------------------------------------
# Función que calcula el p0_swap aproximado SIN usar Qiskit   ✓ RÁPIDA
# ------------------------------------------------------------
def p0_from_phases(a, b, theta_plus, theta_minus):
    """
    Calcula p0_swap = (1 + |M|^2)/2 donde
       M = promedio_i [ z(a_i) * conj(z(b_i)) ]

    con z(+1)=e^{iθ_plus}, z(-1)=e^{iθ_minus}
    """
    z_plus = torch.exp(1j * theta_plus)   # complejos
    z_minus = torch.exp(1j * theta_minus)

    za = torch.where(a > 0, z_plus, z_minus)
    zb = torch.where(b > 0, z_plus, z_minus)

    # M = <ψ(a)|ψ(b)>
    M = torch.mean(za * torch.conj(zb))
    M_abs2 = torch.real(M * torch.conj(M))
    p0 = 0.5 * (1.0 + M_abs2)
    return p0.real


# ------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------
def main():
    device = "cpu"
    print("Generando dataset (puede tardar unos segundos)...")

    dataset = PhaseEmbeddingCosDataset(
        n_samples=5000,
        dim_min=128,
        dim_max=128,
        device=device
    )

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = PhaseEmbeddingCosModel(hidden_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("\nEntrenando modelo...\n")
    for epoch in range(1, 41):
        model.train()
        train_losses = []

        for cos_target, a, b, p0_real in train_loader:
            cos_target = cos_target.unsqueeze(1)
            theta_plus, theta_minus = model(cos_target)

            p0_pred = torch.stack([
                p0_from_phases(a[i], b[i], theta_plus[i], theta_minus[i])
                for i in range(len(a))
            ])

            loss = loss_fn(p0_pred, p0_real)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        # validación
        model.eval()
        val_losses = []
        with torch.no_grad():
            for cos_target, a, b, p0_real in val_loader:
                cos_target = cos_target.unsqueeze(1)
                theta_plus, theta_minus = model(cos_target)

                p0_pred = torch.stack([
                    p0_from_phases(a[i], b[i], theta_plus[i], theta_minus[i])
                    for i in range(len(a))
                ])

                loss = loss_fn(p0_pred, p0_real)
                val_losses.append(loss.item())

        print(f"[Epoch {epoch:3d}] Train MSE={np.mean(train_losses):.6f} | "
              f"Val MSE={np.mean(val_losses):.6f}")

    # Guardar modelo
    out_path = "results/phase_embedding_by_cos.pt"
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"\nModelo guardado en: {out_path}\n")

    # Mostrar tabla de ángulos aprendidos
    print("Ejemplo de ángulos aprendidos:")
    for c_test in [-0.9, -0.5, 0.0, 0.5, 0.9]:
        cos_t = torch.tensor([[c_test]], dtype=torch.float32)
        thp, thm = model(cos_t)
        print(f"  cos={c_test:+.3f} → θ+= {float(thp):+.3f}, θ-= {float(thm):+.3f}")


if __name__ == "__main__":
    main()
