import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

nome_encoding = "RBF Classico"

def _kernel_target_alignment(K, y):
    """Calcolo interno del KTA per l'ottimizzazione rapida."""
    y = np.asarray(y).flatten()
    Y_target = np.outer(y, y)
    inner_product = np.sum(K * Y_target)
    norm_K = np.linalg.norm(K, ord='fro')
    norm_Y = np.linalg.norm(Y_target, ord='fro')
    return inner_product / (norm_K * norm_Y)

def funzione_adattamento_dati(set_preparato):
    # IMPORTAZIONE DEI DATI.
    train_set_preparato, val_set_preparato, test_set_preparato = set_preparato

    X_train_preparato, y_train_preparato = train_set_preparato
    X_val_preparato, y_val_preparato = val_set_preparato
    X_test_preparato, y_test_preparato = test_set_preparato

    # ADATTAMENTO DEL DATASET AL KERNEL CLASSICO.
    # Per l'RBF classico utilizziamo StandardScaler (media 0, varianza 1).
    # A differenza del MinMaxScaler usato per l'IQP, la standardizzazione rende
    # isotropa la distanza euclidea tra le feature, che è il comportamento ottimale
    # da dare in pasto all'esponenziale del kernel RBF.
    scalatore = StandardScaler()
    
    X_train_scalato = scalatore.fit_transform(X_train_preparato)
    X_val_scalato = scalatore.transform(X_val_preparato)
    X_test_scalato = scalatore.transform(X_test_preparato)

    # ESPORTAZIONE DEI DATI.
    train_set_adattato = (X_train_scalato, y_train_preparato)
    val_set_adattato = (X_val_scalato, y_val_preparato)
    test_set_adattato = (X_test_scalato, y_test_preparato)

    set_adattato = (train_set_adattato, val_set_adattato, test_set_adattato)

    return set_adattato


def kernel(set_preparato, gammas=None):
    
    # Se la griglia di gamma non viene fornita, esploriamo in scala logaritmica
    if gammas is None:
        gammas = np.logspace(-3, 2, 20)
        
    # IMPORTAZIONE DEI DATI.
    set_adattato = funzione_adattamento_dati(set_preparato)

    train_set_adattato, val_set_adattato, test_set_adattato = set_adattato

    X_train_adattato, y_train_adattato = train_set_adattato
    X_val_adattato, _ = val_set_adattato
    X_test_adattato, _ = test_set_adattato

    # GRID SEARCH SUL PARAMETRO GAMMA TRAMITE KERNEL-TARGET ALIGNMENT.
    miglior_gamma = None
    miglior_kta = -1.0
    
    print(f"Ricerca del gamma ottimale per {nome_encoding} tramite KTA...")
    for g in gammas:
        # Calcoliamo la Gram matrix temporanea solo sul train set per valutarne l'allineamento
        K_train_tmp = rbf_kernel(X_train_adattato, gamma=g)
        kta = _kernel_target_alignment(K_train_tmp, y_train_adattato)
        
        if kta > miglior_kta:
            miglior_kta = kta
            miglior_gamma = g
            
    print(f"Miglior gamma selezionato: {miglior_gamma:.4f} (KTA: {miglior_kta:.4f})")

    # CALCOLO DELLE MATRICI DI GRAM DEFINITIVE.
    matrice_gram_train = rbf_kernel(X_train_adattato, gamma=miglior_gamma)
    matrice_gram_val = rbf_kernel(X_val_adattato, X_train_adattato, gamma=miglior_gamma)
    matrice_gram_test = rbf_kernel(X_test_adattato, X_train_adattato, gamma=miglior_gamma)

    # ESPORTAZIONE DELLE MATRICI DI GRAM E DEI DATI. 
    matrici_gram = matrice_gram_train, matrice_gram_val, matrice_gram_test

    return set_adattato, matrici_gram