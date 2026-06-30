import numpy as np

def kernel_target_alignment(K, y):
    """
    Calcola il Kernel-Target Alignment (KTA) per una matrice di Gram K e un vettore di target y.
    
    Il KTA misura la similitudine tra la geometria indotta dal kernel e la classificazione ideale.
    Valori più alti (vicini a 1) indicano che il kernel separa bene le classi.
    
    Parametri:
    K (np.ndarray): Matrice di Gram (N, N).
    y (np.ndarray): Vettore dei target di dimensione (N,), tipicamente a valori {-1, 1}.
    
    Ritorna:
    float: Il valore del Kernel-Target Alignment.
    """
    # Assicurati che y sia un vettore riga o colonna appropriato
    y = np.asarray(y).flatten()
    
    # Matrice target ideale y * y^T
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
    """
    Calcola il Coefficiente Geometrico (Geometric Difference) g(K_c, K_q).
    
    Questa metrica (Huang et al., 2021) quantifica quanto la geometria dello spazio
    delle feature quantistico differisce da quella classica.
    La formula calcola la norma spettrale di sqrt(K_c) * inv(K_q) * sqrt(K_c).
    
    Parametri:
    K_classico (np.ndarray): Matrice di Gram del kernel classico (es. RBF), (N, N).
    K_quantistico (np.ndarray): Matrice di Gram del kernel quantistico, (N, N).
    lambda_reg (float): Termine di regolarizzazione per evitare singolarità nell'inversione di K_q.
    
    Ritorna:
    float: Il valore del coefficiente geometrico.
    """
    N = K_classico.shape[0]
    
    # Regolarizzazione del kernel quantistico per l'inversione
    K_q_reg = K_quantistico + lambda_reg * np.eye(N)
    
    # Calcolo dell'inversa del kernel quantistico
    try:
        K_q_inv = np.linalg.inv(K_q_reg)
    except np.linalg.LinAlgError:
        # Pseudo-inversa come fallback se la matrice è troppo mal condizionata
        K_q_inv = np.linalg.pinv(K_q_reg)
        
    # Calcolo della radice quadrata del kernel classico
    # Poiché K_c è semi-definita positiva, usiamo la decomposizione spettrale
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