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