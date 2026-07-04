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