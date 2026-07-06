import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from ..metriche import kernel_target_alignment

nome_encoding = "RBF Classico"

def kernel(train_set, test_set, gammas=None):
    
    if gammas is None:
        gammas = np.logspace(-3, 2, 20)
        
    scaler = StandardScaler()
    X_train_scalato = scaler.fit_transform(train_set[0])
    X_test_scalato = scaler.transform(test_set[0])

    train_adattato = (X_train_scalato, train_set[1])
    test_adattato = (X_test_scalato, test_set[1])

    miglior_gamma = None
    miglior_kta = -1.0
    
    for g in gammas:
        K_train_tmp = rbf_kernel(X_train_scalato, gamma=g)
        kta = kernel_target_alignment(K_train_tmp, train_set[1])
        if kta > miglior_kta:
            miglior_kta = kta
            miglior_gamma = g

    matrice_gram_train = rbf_kernel(X_train_scalato, gamma=miglior_gamma)
    matrice_gram_test = rbf_kernel(X_test_scalato, X_train_scalato, gamma=miglior_gamma)

    return (train_adattato, test_adattato), (matrice_gram_train, matrice_gram_test)