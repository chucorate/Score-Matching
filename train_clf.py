'''
Entrena el clasificador auxiliar sobre los datos reales y guarda clasificador.pt.

def train(model, data, *, n_epochs, batch_size, lr, device, seed, label_dropout=0.0, **kwargs):
    """Entrena 'model' con mini-batches. Retorna un dict con el historial."""
    ...

if __name__ == "__main__":
    # parsea hiperparametros y device (argparse): --device --epochs
    # --batch-size --lr --seed --label-dropout --out, etc.
    # Carga MNIST, entrena, y guarda el checkpoint y la curva de perdida (.png).
'''

import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model_clf import Clasificador


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    model, data, *, n_epochs, batch_size, lr, device, seed, checkpoint_path, **kwargs
):
    """Entrena 'model' con mini-batches."""
    set_seed(seed)
    model = model.to(device)

    # Split 90/10 para Validación
    val_size = int(0.1 * len(data))
    train_dataset, val_dataset = torch.utils.data.random_split(
        data, [len(data) - val_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_epoch = 0

    # --- SISTEMA REANUDAR ENTRENAMIENTO (ANTI-CORTES) ---
    if checkpoint_path.exists():
        print(f"==> Detectado progreso previo en {checkpoint_path.name}. Reanudando...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        print(f"==> Reseteado con éxito desde la Época {start_epoch}")

    print(
        f"Entrenando en {device} desde la época {start_epoch + 1} hasta {n_epochs}..."
    )

    for epoch in range(start_epoch, n_epochs):
        # --- ENTRAR A ENTRENAR ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Autocast para acelerar hardware NVIDIA moderno (Precisión Mixta)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

        # --- VALIDACIÓN ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(dim=1) == targets).sum().item()
                val_total += targets.size(0)

        # Guardar en historial
        history["train_loss"].append(running_loss / total)
        history["train_acc"].append(correct / total)
        history["val_loss"].append(val_loss / val_total)
        history["val_acc"].append(val_correct / val_total)

        print(
            f"Época [{epoch + 1}/{n_epochs}] -> Loss (T/V): {history['train_loss'][-1]:.4f}/{history['val_loss'][-1]:.4f} | Acc (T/V): {history['train_acc'][-1] * 100:.1f}%/{history['val_acc'][-1] * 100:.1f}%"
        )

        # GUARDAR RESPALDO DE LA ÉPOCA ACTUAL
        checkpoint_snapshot = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "model_config": {"num_clases": 10, "dropout": 0.2},
        }
        torch.save(checkpoint_snapshot, checkpoint_path)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-dropout", type=float, default=0.0
    )  # Se pide por interfaz externa, se ignora aquí.

    args = parser.parse_args()

    # Carga de datos
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    mnist_data = datasets.MNIST(
        root=DEFAULT_DATA_DIR, train=True, download=True, transform=transform
    )

    # Instanciación
    clasificador = Clasificador(num_clases=10, dropout=0.2)
    checkpoint_file = OUTPUT_DIR / "clasificador.pt"

    # Ejecución
    historial = train(
        model=clasificador,
        data=mnist_data,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=torch.device(args.device),
        seed=args.seed,
        checkpoint_path=checkpoint_file,
    )

    # --- EXPORTAR GRÁFICA FINAL (.PNG) ---
    plt.figure(figsize=(9, 4))
    plt.plot(historial["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(historial["val_loss"], label="Val Loss", color="crimson", linestyle="--")
    plt.title("Historial de Pérdida - Clasificador Auxiliar")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = OUTPUT_DIR / "curva_perdida.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(
        f"\n[Terminado] Todo guardado en la carpeta 'outputs/':\n - Pesos: {checkpoint_file.name}\n - Gráfica: {plot_path.name}"
    )
