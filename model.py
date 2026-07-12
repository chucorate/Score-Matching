"""
Define la red de difusión. Importable como from model import Modelo.

class Modelo(nn.Module):
    def __init__(self, **config): ...        # construye la UNet condicional
        def forward(self, x, t, y=None): ... # retorna s_theta / eps_theta;
                                             # y=None usa el token nulo
        def time_embedding(self, t): ...     # embedding temporal (auxiliar)
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def alpha_defecto(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * t / 2)


def alpha_derivada_defecto(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.pi * torch.cos(torch.pi * t / 2)


def beta_defecto(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(torch.pi * t / 2)


def beta_derivada_defecto(t: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.pi * torch.sin(torch.pi * t / 2)


class Modelo(nn.Module):
    # Arquitectura de la red ====================================================

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        embedding_dim: int = 256,
        n_labels: int = 10,
        cfg: bool = True,
        alpha: Callable[[torch.Tensor], torch.Tensor] = alpha_defecto,
        beta: Callable[[torch.Tensor], torch.Tensor] = beta_defecto,
        alpha_derivada: Callable[[torch.Tensor], torch.Tensor] = alpha_derivada_defecto,
        beta_derivada: Callable[[torch.Tensor], torch.Tensor] = beta_derivada_defecto,
        **config,
    ):
        super().__init__()

        # Funciones alpha y beta usadas para el entrenamiento
        self.alpha = alpha
        self.beta = beta
        self.alpha_derivada = alpha_derivada
        self.beta_derivada = beta_derivada

        # Embeddings
        self.embedding_dim = embedding_dim

        # se incluye el token nulo dependiendo de si la red usa cfg
        self.cfg = cfg

        cantidad_labels = n_labels + 1 if cfg else n_labels
        self.token_nulo = n_labels
        self.label_embedding = nn.Embedding(cantidad_labels, embedding_dim)

        self.mlp_1 = nn.Sequential(
            nn.Linear(self.embedding_dim, base_channels * 8),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels * 8, base_channels),
        )

        self.mlp_2 = nn.Sequential(
            nn.Linear(self.embedding_dim, base_channels * 8),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels * 8, base_channels * 2),
        )

        self.mlp_3 = nn.Sequential(
            nn.Linear(self.embedding_dim, base_channels * 8),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels * 8, base_channels * 4),
        )

        # Capas encoder

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.downscale_1 = nn.MaxPool2d(2)
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.downscale_2 = nn.MaxPool2d(2)
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.downscale_3 = nn.MaxPool2d(2)

        # Capa midcoder

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.SiLU(inplace=True),
        )

        # Capas decoder

        self.upscale_3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, 2, stride=2
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.upscale_2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 2, stride=2
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.upscale_1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 2, stride=2
        )
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )

        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Devuelve eps_theta(x, y, t) aprendido."""
        n_batches = x.shape[0]
        if y is None:
            if not self.cfg:
                raise ValueError(
                    "Usar token nulo sólo es válido cuando la red usa CFG."
                )
            y = torch.full((n_batches,), self.token_nulo).to(x.device)

        t_emb = self.time_embedding(t).to(x.device)
        y_emb = self.label_embedding(y.long()).to(x.device)

        t_emb1 = self.mlp_1(t_emb).view(n_batches, -1, 1, 1)
        y_emb1 = self.mlp_1(y_emb).view(n_batches, -1, 1, 1)
        t_emb2 = self.mlp_2(t_emb).view(n_batches, -1, 1, 1)
        y_emb2 = self.mlp_2(y_emb).view(n_batches, -1, 1, 1)
        t_emb3 = self.mlp_3(t_emb).view(n_batches, -1, 1, 1)
        y_emb3 = self.mlp_3(y_emb).view(n_batches, -1, 1, 1)

        # Capas de encoders

        e1 = self.encoder_1(x)
        p1 = self.downscale_1(e1)
        e2 = self.encoder_2(p1 + t_emb1 + y_emb1)  # incrustamos embeddings en bajada
        p2 = self.downscale_2(e2)
        e3 = self.encoder_3(p2 + t_emb2 + y_emb2)
        p3 = self.downscale_3(e3)

        # Capa midcoder

        b = self.bottleneck(p3 + t_emb3 + y_emb3)

        # Capas de decoders

        u3 = self.upscale_3(b)
        d3 = self.decoder_3(torch.cat([u3, e3], dim=1))
        u2 = self.upscale_2(d3)
        d2 = self.decoder_2(
            torch.cat(
                [u2 + t_emb2 + y_emb2, e2], dim=1
            )  # incrustamos embeddings en subida
        )
        u1 = self.upscale_1(d2)
        d1 = self.decoder_1(torch.cat([u1 + t_emb1 + y_emb1, e1], dim=1))

        return self.final(d1)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(
            -emb * torch.arange(half_dim, dtype=torch.float32, device=t.device)
        )
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        return emb

    # Camino de probabilidad ====================================================
    # incrustamos lo relacionado al camino de probabilidad en model
    # por temas de conveniencia al importar y para evitar llamar muchos argumentos
    # extra en otras funciones (pues se revisarán de forma semiautomática),
    # pero en estricto rigor esto no es parte de la red

    def score_condicional(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Devuelve el score condicional de forma explícita."""
        alpha = self.alpha(t).view(-1, 1, 1, 1)
        beta = self.beta(t).view(-1, 1, 1, 1)

        return -(x - alpha * z) / (beta**2 + 1e-5)

    def score(self, x, t, y=None):
        """Devuelve el score_theta en función del error eps_theta aprendido."""
        eps = self(x, t, y)
        beta = self.beta(t).view(-1, 1, 1, 1)
        return -eps / (beta + 1e-5)

    def a(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_derivada(t) / (self.alpha(t) + 1e-5)

    def b(self, t: torch.Tensor) -> torch.Tensor:
        beta = self.beta(t)

        return self.a(t) * beta**2 - self.beta_derivada(t) * beta

    def drift(
        self,
        score: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula el drift en función del score."""
        a = self.a(t).view(-1, 1, 1, 1)
        b = self.b(t).view(-1, 1, 1, 1)

        return a * x + b * score
