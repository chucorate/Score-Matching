# Tarea 3 MA5606 - Score Matching

Este proyecto implementa una red de difusión para generar muestras de dígitos MNIST, junto con un clasificador auxiliar. El flujo principal consiste en:

1. Entrenar el clasificador auxiliar.
2. Entrenar la red de difusión en modo condicional puro.
3. Entrenar la misma red en modo CFG.
4. Ejecutar el muestreo para generar grillas de muestras tanto para la versión condicional como para CFG.

## Requisitos

Instala las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

Los scripts esperan que los datos de MNIST estén disponibles en:

- data/data_entrenamiento
- data/data_prueba

Para descargarlos, se debe usar el siguiente comando:

```bash
python download_data.py
```

## 1. Entrenamiento del clasificador auxiliar

Este paso entrena el clasificador auxiliar y guarda los resultados en la carpeta outputs.

```bash
python train_clf.py
```

Parámetros por defecto usados por el script:

- --device: cuda si está disponible, de lo contrario cpu
- --epochs: 100
- --batch-size: 4096
- --lr: 0.001
- --seed: 42

Archivos generados:

- outputs/clasificador.pt
- outputs/clasificador_perdida.png

## 2. Entrenamiento de la red de difusión

### 2.1 Modelo condicional puro (sin CFG)

Para entrenar la red sin CFG, se debe usar label-dropout en 0.0.

```bash
python train.py --label-dropout 0.0
```

Esto genera:

- outputs/modelo_cond.pt
- outputs/modelo_cond_perdida.png

### 2.2 Modelo con CFG

Para entrenar la variante con classifier-free guidance, se usa el valor por defecto de label-dropout.

```bash
python train.py
```

Equivalente a:

```bash
python train.py --label-dropout 0.3
```

Esto genera:

- outputs/modelo_cfg.pt
- outputs/modelo_cfg_perdida.png

Parámetros por defecto usados por train.py:

- --epochs: 1000
- --batch-size: 4096
- --lr: 1e-4
- --seed: 42
- --device: cuda
- --label-dropout: 0.3
- --out: outputs

## 3. Muestreo para la versión condicional y para CFG

Una vez entrenados ambos modelos, se puede ejecutar el muestreo con:

```bash
python sample.py
```

Este script:

- carga outputs/modelo_cond.pt para generar muestras en modo condicional,
- carga outputs/modelo_cfg.pt para generar muestras en modo CFG,

Archivos generados:

- outputs/muestras_difusion_cond.png
- outputs/proceso_difusion_cond.png

## Resumen de comandos recomendados

```bash
pip install -r requirements.txt
python download_data.py
python train_clf.py
python train.py --label-dropout 0.0
python train.py
python sample.py
```

> Nota: el caso sin CFG debe ejecutarse explícitamente con --label-dropout 0.0, porque el valor por defecto del entrenamiento de la red de difusión es 0.3 y activa CFG.
