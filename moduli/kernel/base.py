import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler

def applica_minmax_con_margine(train_set, test_set, feature_range, limite_inf, limite_sup):

    # Funzione per applicare minmax scaling (serve in projected, iqp, basis, angle).

    X_train, y_train = train_set
    X_test, y_test = test_set

    scaler = MinMaxScaler(feature_range=feature_range)
    X_train_scalato = scaler.fit_transform(X_train)
    X_test_scalato = scaler.transform(X_test)

    X_test_clippato = np.clip(X_test_scalato, limite_inf, limite_sup)

    frazione_clippati = np.sum(X_test_scalato != X_test_clippato) / X_test_scalato.size

    if frazione_clippati > 0.05:
        print(f"Anomalie nel test/val set: il {frazione_clippati*100:.2f}% dei dati è stato clippato.")

    return (X_train_scalato, y_train), (X_test_clippato, y_test)

def alloca_device(numero_qubits):

    # Funzione per utilizzare la GPU anziché la CPU.

    SOGLIA_GPU = 16

    try:
        if numero_qubits >= SOGLIA_GPU:

            print(f"[Hardware] Allocati {numero_qubits} qubit. Backend: GPU.")
            return qml.device("lightning.gpu", wires=numero_qubits)
        
        else:
            print(f"[Hardware] Allocati {numero_qubits} qubit. Backend: CPU.")
            return qml.device("default.qubit", wires=numero_qubits)
        
    except qml.DeviceError:

        print("[Avviso] GPU richiesta ma non disponibile. Fallback su CPU.")
        return qml.device("lightning.qubit", wires=numero_qubits)

def calcola_matrici_gram_fedelta(X_train, X_test, qnode_circuito):
    matrice_gram_train = qml.kernels.square_kernel_matrix(X_train, qnode_circuito, assume_normalized_kernel=True)
    matrice_gram_test = qml.kernels.kernel_matrix(X_test, X_train, qnode_circuito)
    return matrice_gram_train, matrice_gram_test