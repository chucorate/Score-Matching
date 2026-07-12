'''
Entrena una red de difusión. La opción 'label-dropout' controla cuál de las dos redes
se obtiene: 0.0 produce la red condicional pura (modelo_cond.pt); un valor >0 produce la red con
CFG (modelo_cfg.pt).

def train(model, data, *, n_epochs, batch_size, lr, device, seed, label_dropout=0.0, **kwargs):
    """Entrena 'model' con mini-batches. Retorna un dict con el historial.""""""
    ...

if __name__ == "__main__":
    # parsea hiperparametros y device (argparse): --device --epochs
    # --batch-size --lr --seed --label-dropout --out, etc.
    # Carga MNIST, entrena, y guarda el checkpoint y la curva de perdida (.png).

El dispositivo y los hiperparámetros deben ser configurables desde la línea de comandos, no estar
fijados en el código (hardcoded).
'''

import argparse
from tqdm import tqdm  # type: ignore
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset

from model import Modelo


def guardar_historial(historial: list[float], archivo: str | Path) -> None:
    """Guarda la curva de pérdida durante el entrenamiento."""

    plt.figure(figsize=(7, 4))
    plt.plot(historial, linewidth=2)

    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Historial de entrenamiento")

    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.yscale("log")
    plt.savefig(archivo)
    plt.close()


EPS = 5e-3


def train(
    model: Modelo,
    data: Dataset,
    *,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    label_dropout: float,
    **kwargs,
) -> list[float]:
    """Entrena `model` en mini-batches usando `data`. Retorna una lista con el historial."""

    # Volvemos el entrenamiento completamente determinista
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    historial_perdida = []

    criterio = torch.nn.MSELoss()
    optimizador = optim.Adam(model.parameters(), lr=lr)

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    for epoch in tqdm(range(n_epochs), desc="Entrenando modelo"):
        perdida_total = 0

        model.train()

        for z, labels in dataloader:
            optimizador.zero_grad()

            z = z.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            n_batches = z.shape[0]

            # Evitamos entrenar en los extremos dado que el score condicional se vuelve inestable
            t = EPS + (1 - 2 * EPS) * torch.rand(
                (n_batches,), device=device, dtype=torch.float32
            )

            # Sampleamos las etiquetas que se evaluaran en la red,
            # interpretando label_dropout como la probabilidad de reemplazo del token nulo
            bernoulli_eta = (
                torch.rand((n_batches,), dtype=torch.float32, device=device)
                < label_dropout
            )
            labels_entrenamiento = torch.where(
                bernoulli_eta,
                torch.full_like(labels, model.token_nulo, device=device),
                labels,
            )

            ruido = torch.randn_like(z)
            x = (
                model.alpha(t).view(-1, 1, 1, 1) * z
                + model.beta(t).view(-1, 1, 1, 1) * ruido
            )

            ruido_predicho = model(x, t, labels_entrenamiento)

            loss = criterio(ruido_predicho, ruido)
            loss.backward()

            optimizador.step()

            perdida_total += loss.item()

        historial_perdida.append(
            perdida_total / len(dataloader)
        )  # guardamos la pérdida promedio en cada época

    return historial_perdida


if __name__ == "__main__":
    # Cargamos datos y fijamos parámetros iniciales
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--label-dropout", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=Path("data"))

    args = parser.parse_args()

    DATA_PATH = Path(args.out)
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(16),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset_mnist = torchvision.datasets.MNIST(
        root=DATA_PATH / "data_entrenamiento",
        train=True,
        transform=transform,
        download=False,
    )

    usar_cfg = args.label_dropout > 0

    CONFIGURACION = {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 32,
        "embedding_dim": 256,
        "n_labels": 10,
        "cfg": usar_cfg,
    }

    # Inicializamos el modelo a entrenar
    model = Modelo(**CONFIGURACION).to(device=device, dtype=torch.float32)

    tipo = "CFG" if usar_cfg else "condicional"
    print(f"Iniciando entrenamiento {tipo} en el dispositivo: {device}")

    historial = train(
        model=model,
        data=dataset_mnist,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
        label_dropout=args.label_dropout,
    )

    nombre_guardado = "modelo_cfg" if usar_cfg else "modelo_cond"

    torch.save(
        obj={"model_config": CONFIGURACION, "model_state": model.state_dict()},
        f=DATA_PATH / f"{nombre_guardado}.pt",
    )

    guardar_historial(historial, DATA_PATH / f"{nombre_guardado}_perdida.png")
