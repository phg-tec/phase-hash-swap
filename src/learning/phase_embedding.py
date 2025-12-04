# ============================================================
# model_phase_embedding_by_cos.py
# ============================================================
import torch
import torch.nn as nn
import math


class PhaseEmbeddingCosModel(nn.Module):
    """
    Entrada:   cos_target ∈ [-1,1]
    Salida:    θ_plus, θ_minus  ∈ [-π, π]
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, cos_value):
        """
        cos_value: tensor shape [batch, 1]
        Return: θ_plus, θ_minus in [-π, π]
        """
        x = self.net(cos_value)
        # limitar ángulos en [-π,π] con tanh
        x = math.pi * torch.tanh(x)
        theta_plus = x[:, 0]
        theta_minus = x[:, 1]
        return theta_plus, theta_minus
