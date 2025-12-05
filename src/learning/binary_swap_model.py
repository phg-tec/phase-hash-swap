# src/quantum/binary_swap_model.py

import torch
import torch.nn as nn

class BinarySwapEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi_minus = nn.Parameter(torch.tensor(0.0))
        self.phi_plus  = nn.Parameter(torch.tensor(0.0))

    def get_complex_embeddings(self):
        z_minus = torch.complex(torch.cos(self.phi_minus), torch.sin(self.phi_minus))
        z_plus  = torch.complex(torch.cos(self.phi_plus), torch.sin(self.phi_plus))
        return z_minus, z_plus

    def forward(self, x, y):
        # Para entrenar, pero no lo usaremos aquí
        raise NotImplementedError
