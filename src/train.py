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
'''
