"""
Define la red de difusión. Importable como from model import Modelo.

class Modelo(nn.Module):
    def __init__(self, **config): ...        # construye la UNet condicional
        def forward(self, x, t, y=None): ... # retorna s_theta / eps_theta;
                                             # y=None usa el token nulo
        def time_embedding(self, t): ...     # embedding temporal (auxiliar)
"""