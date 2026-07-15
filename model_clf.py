"""
Define el clasificador auxiliar. Importable como from model_clf import Clasificador.

class Clasificador(nn.Module):
    def __init__(self, **config): ...
    def forward(self, x): ...           # logits sobre las clases
"""

import torch
import torch.nn as nn


class Clasificador(nn.Module):
    def __init__(self, **config):
        """Inicializa la CNN utilizando un diccionario."""
        super().__init__()

        # Extraer hiperparametros con valores por defecto seguros para MNIST
        num_clases = config.get("num_clases", 10)
        p_dropout = config.get("dropout", 0.2)

        # BLOQUE 1: Extractor de rasgos inicial
        self.bloque1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # BLOQUE 2: Extractor de formas complejas
        self.bloque2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # BLOQUE FC (Fully Connected): Proyeccion lineal y Decision Bayesiana
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(128, num_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa la imagen a traves de la jerarquia visual convolucional.
        Retorna: logits sobre las clases (forma: B, num_clases) sin aplicar Softmax.
        """
        # Extraer caracteristicas geometricas base
        x = self.bloque1(x)

        # Combinar rasgos en representaciones de mayor jerarquia semantica
        x = self.bloque2(x)

        # Clasificar el vector de caracteristicas para estimar P(y | x)
        logits = self.fc(x)

        return logits
