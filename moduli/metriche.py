import numpy as np

def kernel_target_alignment(K, y):
    y = np.asarray(y).flatten()
    
    # Conversione a {-1, +1}
    valori_unici = np.unique(y)
    if not np.array_equal(np.sort(valori_unici), np.array([-1, 1])):
        if len(valori_unici) != 2:
            raise ValueError("kernel_target_alignment richiede un problema a 2 classi.")
        y = np.where(y == valori_unici[0], -1, 1)
    
    # Matrice target y * y^T
    Y_target = np.outer(y, y)
    
    # Prodotto interno di Frobenius tra K e Y_target
    inner_product = np.sum(K * Y_target)
    
    # Norme di Frobenius
    norm_K = np.linalg.norm(K, ord='fro')
    norm_Y = np.linalg.norm(Y_target, ord='fro')
    
    # Calcolo dell'allineamento
    alignment = inner_product / (norm_K * norm_Y)
    
    return alignment

def coefficiente_geometrico(K_classico, K_quantistico, lambda_reg=1e-6):
    N = K_classico.shape[0]
    
    K_q_reg = K_quantistico + lambda_reg * np.eye(N)
    
    # Calcolo dell'inversa del kernel quantistico
    try:
        K_q_inv = np.linalg.inv(K_q_reg)
    except np.linalg.LinAlgError:
        K_q_inv = np.linalg.pinv(K_q_reg)
        
    # Calcolo della radice quadrata del kernel classico
    eigvals_c, eigvecs_c = np.linalg.eigh(K_classico)
    
    # Tronca eventuali autovalori negativi dovuti a errori numerici
    eigvals_c = np.maximum(eigvals_c, 0)
    
    K_c_sqrt = eigvecs_c @ np.diag(np.sqrt(eigvals_c)) @ eigvecs_c.T
    
    # Calcolo della matrice combinata: sqrt(K_c) * inv(K_q) * sqrt(K_c)
    M = K_c_sqrt @ K_q_inv @ K_c_sqrt
    
    # Il coefficiente è la radice quadrata della norma spettrale di M
    # (la norma spettrale per matrici simmetriche è il massimo autovalore assoluto)
    eigvals_M = np.linalg.eigvalsh(M)
    norma_spettrale = np.max(np.abs(eigvals_M))
    
    g = np.sqrt(norma_spettrale)
    
    return g
