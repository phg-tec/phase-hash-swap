# ============================================================
# dataset_phase_embedding_by_cos.py
# ============================================================
import torch
from torch.utils.data import Dataset
import numpy as np


def gen_pair_with_cosine(dim: int, cos_target: float):
    """
    Genera dos vectores ±1 EXACTAMENTE con cos ≈ cos_target
    sin rechazo, en tiempo O(dim).
    """
    rng = np.random.default_rng()

    # número de coincidencias necesarias
    k_same = int(round((1 + cos_target) * dim / 2))
    k_diff = dim - k_same

    # vector base
    a = rng.choice([1, -1], size=dim)

    # b empieza igual que a
    b = a.copy()

    # índices a cambiar
    idx = rng.choice(dim, size=k_diff, replace=False)
    b[idx] *= -1

    # cos real exacto
    cos_real = (a @ b) / dim
    return a.astype(float), b.astype(float), cos_real



class PhaseEmbeddingCosDataset(Dataset):
    """
    Cada item contiene:
      - cos_target  (float)
      - vector a (±1)
      - vector b (±1)
      - p0_real = (1 + cos^2)/2
    """
    def __init__(self, n_samples=20000, dim_min=16, dim_max=256, device="cpu"):
        self.device = device

        cos_values = np.random.uniform(-1, 1, size=n_samples).astype(np.float32)

        self.cos_list = []
        self.a_list = []
        self.b_list = []
        self.p0_list = []

        for c in cos_values:
            dim = np.random.randint(dim_min, dim_max + 1)
            a, b, c_real = gen_pair_with_cosine(dim, float(c))

            p0 = 0.5 * (1.0 + c_real * c_real)

            self.cos_list.append(torch.tensor(c_real, dtype=torch.float32))
            self.a_list.append(torch.tensor(a, dtype=torch.float32))
            self.b_list.append(torch.tensor(b, dtype=torch.float32))
            self.p0_list.append(torch.tensor(p0, dtype=torch.float32))

        self.cos_list = [t.to(device) for t in self.cos_list]
        self.a_list = [t.to(device) for t in self.a_list]
        self.b_list = [t.to(device) for t in self.b_list]
        self.p0_list = [t.to(device) for t in self.p0_list]

    def __len__(self):
        return len(self.cos_list)

    def __getitem__(self, idx):
        return (
            self.cos_list[idx],
            self.a_list[idx],
            self.b_list[idx],
            self.p0_list[idx],
        )
