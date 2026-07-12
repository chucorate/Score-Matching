"""
Descarga MNIST mediante torchvision y deja los datos disponibles para
train.py y train_clf.py. La ruta de destino se fija en el código y debe ser la misma que usan los
scripts de entrenamiento; no se entregan los datos descargados, solo este script.
"""

from pathlib import Path

import torchvision


DATA_DIR = Path("data")


# Reescalamos y normalizamos para facilitar el entrenamiento
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(16),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)


dataset_entrenamiento = torchvision.datasets.MNIST(
    root=DATA_DIR / "data_entrenamiento",
    train=True,
    download=True,
    transform=transform,
)

dataset_prueba = torchvision.datasets.MNIST(
    root=DATA_DIR / "data_prueba",
    train=False,
    download=True,
    transform=transform,
)
