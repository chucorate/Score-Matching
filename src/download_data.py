"""
Descarga MNIST mediante torchvision y deja los datos disponibles para
train.py y train_clf.py. La ruta de destino se fija en el código y debe ser la misma que usan los
scripts de entrenamiento; no se entregan los datos descargados, solo este script.
"""
from pathlib import Path
import torchvision


DATA_DIR = Path("data")


dataset_entrenamiento = torchvision.datasets.MNIST(
    root=DATA_DIR / "data_entrenamiento",
    train=True,
    download=True,
)

dataset_prueba = torchvision.datasets.MNIST(
    root=DATA_DIR / "data_prueba",
    train=False,
    download=True,
)
