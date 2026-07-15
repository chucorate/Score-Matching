"""
Carga las dos redes de difusión y el clasificador, genera muestras por clase con cada
red (w = 1 y el w ̸= 1 elegido) y reporta fidelidad y diversidad. Debe producir las tablas y gráficos
especificados en la validación experimental y los que el estudiante estime convenientes.
"""

from pathlib import Path
from typing import cast, Any

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Modelo
from model_clf import Clasificador
from sample import sample


CLASES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# CLASIFICACION


def clasificar_clases(
    model: Modelo,
    clasificador: Clasificador,
    clase: int,
    n_samples: int,
    device: torch.device,
    w: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera `n_samples` muestras de la clase `clase` usando el modelo de generación de imágenes `model`,
    y las clasifica usando el clasificador `clasificador`.
    """
    ruido_inicial = torch.randn((n_samples, 1, 16, 16), device=device)
    labels = torch.tensor(
        [clase for _ in range(0, n_samples)], dtype=torch.long, device=device
    )
    sigma = lambda t: 0.5 * torch.ones_like(t)

    muestras, _ = sample(
        model=model,
        condition=ruido_inicial,
        n_steps=250,
        w=w,
        device=device,
        sigma=sigma,
        labels=labels,
    )

    with torch.no_grad():
        logits = clasificador(muestras)
        probabilidades = torch.softmax(logits, dim=1)
        predicciones = probabilidades.argmax(dim=1)

    return muestras.cpu().numpy(), predicciones.cpu().numpy()


def evaluar_clasificacion(
    muestras: np.ndarray, predicciones: np.ndarray, clase: int
) -> tuple[np.ndarray, np.ndarray]:
    fidelidad = np.mean(predicciones == clase)

    # calculamos las diferencias a pares
    x = muestras.reshape(muestras.shape[0], -1)

    diferencias = x[:, None, :] - x[None, :, :]
    distancias = np.linalg.norm(diferencias, axis=-1)

    i, j = np.triu_indices(x.shape[0], k=1)

    diversidad = distancias[i, j].mean()

    return fidelidad, diversidad


# TABLAS Y GRÁFICOS


def graficar_evaluacion(predicciones: np.ndarray, modelo: str, clase: int) -> None:
    output_dir = Path("resultados_clasificacion")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))

    plt.hist(
        predicciones,
        bins=(np.arange(11) - 0.5).tolist(),
        rwidth=0.9,
    )

    plt.xticks(CLASES)
    plt.xlabel("Clase predicha")
    plt.ylabel("Frecuencia")
    plt.title(f"{modelo} - clase {clase}")
    plt.tight_layout()

    plt.savefig(output_dir / f"{modelo}_clase_{clase}.png")
    plt.close()


def crear_dataframe_metricas(
    resultados: list[tuple[float, float]],
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Clase": CLASES,
            "Fidelidad": [r[0] for r in resultados],
            "Diversidad": [r[1] for r in resultados],
        }
    )

    promedio = pd.DataFrame(
        {
            "Clase": ["Promedio"],
            "Fidelidad": [df["Fidelidad"].mean()],
            "Diversidad": [df["Diversidad"].mean()],
        }
    )

    return pd.concat([df, promedio], ignore_index=True)


def graficar_comparacion(
    df_cond: pd.DataFrame,
    df_cfg: pd.DataFrame,
    columna: str,
    ruta_salida: Path,
) -> None:
    """Compara ambos modelos para una métrica ("Fidelidad" o "Diversidad")."""

    x = np.arange(len(CLASES))
    width = 0.35

    plt.figure(figsize=(8, 4))

    plt.bar(x - width / 2, df_cond.iloc[:-1][columna], width, label="Condicional")
    plt.bar(x + width / 2, df_cfg.iloc[:-1][columna], width, label="CFG")

    plt.xticks(x, CLASES)  # type: ignore
    plt.xlabel("Clase")
    plt.ylabel(columna)
    plt.legend()

    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()


if __name__ == "__main__":
    # Configuramos parámetros iniciales

    OUTPUT_PATH = Path("outputs")
    MODELO_CFG_PATH = OUTPUT_PATH / "modelo_cfg.pt"
    MODELO_COND_PATH = OUTPUT_PATH / "modelo_cond.pt"
    CLASIFICADOR_PATH = OUTPUT_PATH / "clasificador.pt"

    SEED = 42
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Cargamos modelos

    CFG_DICT = cast(dict[str, Any], torch.load(MODELO_CFG_PATH))
    COND_DICT = cast(dict[str, Any], torch.load(MODELO_COND_PATH))
    CLASIFICADOR_DICT = cast(dict[str, Any], torch.load(CLASIFICADOR_PATH))

    model_cfg = Modelo(**CFG_DICT["model_config"]).to(DEVICE)
    model_cfg.load_state_dict(CFG_DICT["model_state"])
    model_cfg.eval()

    model_cond = Modelo(**COND_DICT["model_config"]).to(DEVICE)
    model_cond.load_state_dict(COND_DICT["model_state"])
    model_cond.eval()

    clasificador = Clasificador(**CLASIFICADOR_DICT["model_config"]).to(DEVICE)
    clasificador.load_state_dict(CLASIFICADOR_DICT["model_state"])
    clasificador.eval()

    # Configuramos las instancias de modelos condicional y CFG

    INSTANCIA_CFG = {"model": model_cfg, "w": 1.5, "title": "CFG"}

    INSTANCIA_COND = {"model": model_cond, "w": 1.0, "title": "condicional"}

    N_SAMPLES = 2000

    # Dataframes por instancia para después graficar resultados

    dataframe_por_instancia = {}

    for instancia in [INSTANCIA_CFG, INSTANCIA_COND]:
        resultados = []

        for clase in CLASES:
            muestras, prediccion = clasificar_clases(
                instancia["model"],
                clasificador,
                clase=clase,
                n_samples=N_SAMPLES,
                device=DEVICE,
                w=instancia["w"],
            )
            fidelidad, diversidad = evaluar_clasificacion(
                muestras, prediccion, clase=clase
            )
            resultados.append((fidelidad, diversidad))
            graficar_evaluacion(prediccion, modelo=instancia["title"], clase=clase)

        df = crear_dataframe_metricas(resultados)

        print(f"\n{instancia['title']}")
        print(df)

        df.to_csv(OUTPUT_PATH / f"metricas_{instancia['title']}.csv", index=False)
        dataframe_por_instancia[instancia["title"]] = df

    graficar_comparacion(
        dataframe_por_instancia["condicional"],
        dataframe_por_instancia["CFG"],
        "Fidelidad",
        OUTPUT_PATH / "comparacion_fidelidad.png",
    )

    graficar_comparacion(
        dataframe_por_instancia["condicional"],
        dataframe_por_instancia["CFG"],
        "Diversidad",
        OUTPUT_PATH / "comparacion_diversidad.png",
    )
