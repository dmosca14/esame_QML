import numpy as np
import pennylane as qml
from sklearn.metrics.pairwise import rbf_kernel
from .base import applica_minmax_con_margine, alloca_device

nome_encoding = "Projected Quantum Kernel"

def kernel(train_set, test_set, numero_features, gamma=1.0, numero_layer=1):
    
    margine = 0.05 * np.pi
    train_adattato, test_adattato = applica_minmax_con_margine(
        train_set, test_set, feature_range=(margine, np.pi - margine), limite_inf=0.0, limite_sup=np.pi
    )

    dev_projected = alloca_device(numero_features)

    def blocco_entangling(wires):
        n = len(wires)
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])

    @qml.qnode(dev_projected)
    def circuito_quantistico(campione):
        for _ in range(numero_layer):
            qml.AngleEmbedding(campione, wires=range(numero_features), rotation="Y") # S
            blocco_entangling(range(numero_features)) # W

        osservabili = []
        for w in range(numero_features):
            osservabili.extend([qml.expval(qml.PauliX(w)), qml.expval(qml.PauliY(w)), qml.expval(qml.PauliZ(w))])
        return osservabili

    Phi_train = np.array([circuito_quantistico(x) for x in train_adattato[0]])
    Phi_test = np.array([circuito_quantistico(x) for x in test_adattato[0]])

    gamma_effettivo = gamma / numero_features
    matrice_gram_train = rbf_kernel(Phi_train, Phi_train, gamma=gamma_effettivo)
    matrice_gram_test = rbf_kernel(Phi_test, Phi_train, gamma=gamma_effettivo)

    return (train_adattato, test_adattato), (matrice_gram_train, matrice_gram_test), numero_features