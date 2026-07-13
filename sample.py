'''
Genera imágenes con una red de difusión dada, integrando la SDE reversa.

def sample(model, condition, *, n_steps, w, device, **kwargs):
    """
    `condition` contiene la(s) etiqueta(s) y y/o el ruido inicial X0 ~ N(0,I).
    Integra la SDE reversa (Euler-Maruyama) aplicando CFG con escala `w`
    (w=1 recupera el muestreo condicional puro) y retorna las imagenes
    generadas, forma (B, C, H, W).
    """
    ...

if __name__ == "__main__":
    # carga un checkpoint de difusion, genera de forma reproducible (seed
    # fija) y guarda una grilla .png etiquetada por clase.

'''

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Modelo


EPS = 5e-3


@torch.no_grad()
def sample(
    model: Modelo,
    condition: torch.Tensor,
    *,
    n_steps: int,
    w: float,
    device: torch.device,
    sigma: Callable[[torch.Tensor], torch.Tensor],
    labels: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, list]:
    """
    `condition` contiene la(s) etiqueta(s) y y/o el ruido inicial X0 ~ N(0,I).
    Integra la SDE reversa (Euler-Maruyama) aplicando CFG con escala `w`
    (w=1 recupera el muestreo condicional puro) y retorna las imagenes
    generadas, forma (B, C, H, W).
    """
    model.eval()

    delta_t = 1 / n_steps
    sqrt_dt = torch.sqrt(torch.tensor(delta_t, device=device))

    save_steps = {
        0,
        int(n_steps * 0.25),
        int(n_steps * 0.5),
        int(n_steps * 0.75),
        n_steps - 1,
    }
    imgs_per_step = []

    if w == 1.0:
        score_guiado_funcion = lambda x, t, y: model.score(x, t, y=y)

    else:
        score_guiado_funcion = lambda x, t, y: (
            (1 - w) * model.score(x, t, y=None) + w * model.score(x, t, y=y)
        )

    estado_actual = condition  # Se asume que ya es de la forma (B, C, H, W)
    n_batches = condition.shape[0]

    # evitamos evaluar exactamente en t = 0, ya que la parametrización
    # u = a(t)x + b(t)s contiene la razón alpha'(t)/alpha(t) que es indeterminada en 0
    for index, t in enumerate(torch.linspace(EPS, 1 - EPS, n_steps, device=device)):
        t_batch = torch.full((n_batches,), t.item(), device=device, dtype=torch.float32)

        sigma_t = sigma(t_batch).view(
            -1, 1, 1, 1
        )  # batcheamos el tiempo para evitar problemas de broadcasting
        score_guiado = score_guiado_funcion(estado_actual, t_batch, labels)
        drift_guiado = model.drift(score_guiado, estado_actual, t_batch)

        termino_gaussiano = (
            sigma_t * sqrt_dt * torch.randn_like(estado_actual, device=device)
        )
        termino_drift = delta_t * (drift_guiado + 0.5 * sigma_t**2 * score_guiado)

        estado_actual += termino_drift + termino_gaussiano

        # guardamos imágenes en los pasos especificados para ver el proceso de generación
        if index in save_steps:
            imgs_per_step.append(estado_actual.clone().cpu().numpy())

    return estado_actual, imgs_per_step


def exportar_grilla_muestras(
    samples: torch.Tensor,
    labels: torch.Tensor,
    ruta_salida: Path,
    title: str,
) -> None:
    samples_np = samples.squeeze(1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    num_images = samples_np.shape[0]

    nrows, ncols = 2, 5
    fig = plt.figure(figsize=(14, 7))
    subfigs = fig.subfigures(nrows, ncols, hspace=0.00)
    subfigs_flat = subfigs.flatten()

    for i in range(num_images):
        sf = subfigs_flat[i]
        sf.suptitle(f"Clase: {labels_np[i]}", fontsize=11, weight="bold", y=0.90)

        ax = sf.subplots()
        ax.imshow(samples_np[i], cmap="gray")
        ax.axis("off")

    fig.suptitle(
        f"Muestras finales generadas, usando difusión {title}",
        fontsize=14,
        weight="bold",
        y=1.025,
    )
    plt.savefig(ruta_salida, bbox_inches="tight", dpi=300)
    plt.close()


def exportar_proceso_muestreo(
    imgs_per_step: list[np.ndarray],
    labels: torch.Tensor,
    ruta_salida: Path,
    title: str,
) -> None:
    proceso_np = np.stack(imgs_per_step, axis=0).squeeze(2)
    labels_np = labels.cpu().numpy()

    num_pasos, num_imagenes, H, W = proceso_np.shape
    titulos_columnas = ["t=0.0 (Inicio)", "t=0.25", "t=0.50", "t=0.75", "t=1.0 (Fin)"]

    # creamos una grilla donde cada fila es una clase y cada columna es un paso temporal
    fig = plt.figure(figsize=(15, 2 * num_imagenes))
    subfigs = fig.subfigures(num_imagenes, 1, hspace=0.2)
    fig.subplots_adjust(left=0.2, top=0.90)

    for idx_img in range(num_imagenes):
        sf = subfigs[idx_img]
        sf.suptitle(
            f"Evolución Clase: {labels_np[idx_img]}",
            fontsize=11,
            weight="bold",
            x=0.08,
            y=0.5,
            ha="left",
        )

        # generamos las 5 columnas para la muestra actual
        axes = sf.subplots(1, num_pasos)
        for idx_paso in range(num_pasos):
            ax = axes[idx_paso]
            img = proceso_np[idx_paso, idx_img]

            ax.imshow(img, cmap="gray")
            ax.axis("off")

            if idx_img == 0:
                ax.set_title(titulos_columnas[idx_paso], fontsize=10)

    fig.suptitle(
        f"Proceso de generación de imágenes, usando difusión {title}",
        fontsize=19,
        weight="bold",
        y=1.025,
    )
    plt.savefig(ruta_salida, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    # carga un checkpoint de difusion, genera de forma reproducible (seed
    # fija) y guarda una grilla .png etiquetada por clase.

    # Reproducibilidad

    OUTPUT_PATH = Path("outputs")
    SEED = 42
    sigma = lambda t: (
        0.5 * torch.ones_like(t)
    )  # el sigma que mejores resultados produjo, escogido experimentalmente

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Cargamos el modelo entrenado

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    RUTA_MODELO = OUTPUT_PATH / "modelo_cond.pt"
    diccionario_estados = torch.load(RUTA_MODELO)

    CONFIGURACION = diccionario_estados["model_config"]
    print(CONFIGURACION)
    MODEL_STATE = diccionario_estados["model_state"]

    model = Modelo(**CONFIGURACION).to(DEVICE)
    model.load_state_dict(MODEL_STATE)

    # Generamos los samples

    ruido_inicial = torch.randn((10, 1, 16, 16), device=DEVICE)
    labels = torch.tensor(
        [2, 5, 1, 2, 0, 9, 8, 4, 3, 6], dtype=torch.long, device=DEVICE
    )

    samples, proceso = sample(
        model,
        ruido_inicial,
        n_steps=250,
        w=1.0,
        device=DEVICE,
        sigma=sigma,
        labels=labels,
    )

    # Graficamos y guardamos los samples

    exportar_grilla_muestras(
        samples.to(device="cpu").detach(),
        labels,
        OUTPUT_PATH / "muestras_difusion_cond.png",
        title="condicional",
    )

    exportar_proceso_muestreo(
        proceso,
        labels,
        OUTPUT_PATH / "proceso_difusion_cond.png",
        title="condicional",
    )

    # =====================================

    RUTA_MODELO = OUTPUT_PATH / "modelo_cfg.pt"
    diccionario_estados = torch.load(RUTA_MODELO)

    CONFIGURACION = diccionario_estados["model_config"]
    print(CONFIGURACION)
    MODEL_STATE = diccionario_estados["model_state"]

    model = Modelo(**CONFIGURACION).to(DEVICE)
    model.load_state_dict(MODEL_STATE)

    # Generamos los samples

    ruido_inicial = torch.randn((10, 1, 16, 16), device=DEVICE)
    labels = torch.tensor(
        [2, 5, 1, 2, 0, 9, 8, 4, 3, 6], dtype=torch.long, device=DEVICE
    )

    samples, proceso = sample(
        model,
        ruido_inicial,
        n_steps=250,
        w=1.5,
        device=DEVICE,
        sigma=sigma,
        labels=labels,
    )

    # Graficamos y guardamos los samples

    exportar_grilla_muestras(
        samples.to(device="cpu").detach(),
        labels,
        OUTPUT_PATH / "muestras_difusion_cfg.png",
        title="CFG",
    )

    exportar_proceso_muestreo(
        proceso,
        labels,
        OUTPUT_PATH / "proceso_difusion_cfg.png",
        title="CFG",
    )
