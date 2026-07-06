import numpy as np
import pennylane as qml
from .base import applica_minmax_con_margine, alloca_device, calcola_matrici_gram_fedelta

nome_encoding = "Basis Encoding"

def converti_in_binario_numero_reale(dati, tau):
    
    massimo_valore = (2**tau) - 1
    dati_interi = np.round(dati * massimo_valore).astype(np.uint8)
    matrice_8bit = np.unpackbits(dati_interi[:, :, np.newaxis], axis=2)
    return matrice_8bit[:, :, -tau:].reshape(dati.shape[0], dati.shape[1] * tau)

def kernel(train_set, test_set, numero_features, tau_bit_per_feature):
    train_scalato, test_scalato = applica_minmax_con_margine(
        train_set, test_set, feature_range=(0.05, 0.95), limite_inf=0.0, limite_sup=1.0
    )

    X_train_bin = converti_in_binario_numero_reale(train_scalato[0], tau_bit_per_feature)
    X_test_bin = converti_in_binario_numero_reale(test_scalato[0], tau_bit_per_feature)

    train_adattato = (X_train_bin, train_scalato[1])
    test_adattato = (X_test_bin, test_scalato[1])

    numero_qubits_basis = numero_features * tau_bit_per_feature
    dev_basis = alloca_device(numero_qubits_basis)

    @qml.qnode(dev_basis)
    def circuito_quantistico(campione_1, campione_2): 
        qml.BasisEmbedding(features=campione_1, wires=range(numero_qubits_basis)) 
        qml.adjoint(qml.BasisEmbedding)(features=campione_2, wires=range(numero_qubits_basis)) 
        return qml.expval(qml.Projector(np.zeros(numero_qubits_basis), wires=range(numero_qubits_basis)))

    matrici_gram = calcola_matrici_gram_fedelta(X_train_bin, X_test_bin, circuito_quantistico)
    return (train_adattato, test_adattato), matrici_gram, numero_qubits_basis