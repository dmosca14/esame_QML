import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def prepara_dataset(train_set, test_set, numero_features):
    X_train_importato, y_train_importato = train_set
    X_test_importato, y_test_importato = test_set
    
    numero_features_originali_dataset = X_train_importato.shape[1]

    # per aggiornare e ritornare il numero di feature corretto per il dataset corrente
    if numero_features > numero_features_originali_dataset:
        numero_features = numero_features_originali_dataset
        

    # fit solo su train, solo transform su test
    scaler = StandardScaler() 
    X_train_scalato = scaler.fit_transform(X_train_importato) 
    X_test_scalato = scaler.transform(X_test_importato)

    # fit solo su train, solo transform su test
    pca = PCA(n_components = numero_features)
    X_train_preparato = pca.fit_transform(X_train_scalato)
    X_test_preparato = pca.transform(X_test_scalato)
    print(f"Varianza spiegata dalla PCA ({numero_features} componenti): {pca.explained_variance_ratio_.sum()*100:.2f}%")
            
    return (X_train_preparato, y_train_importato), (X_test_preparato, y_test_importato), numero_features