import numpy as np
import pennylane as qml
from sklearn.preprocessing import normalize
from .base import alloca_device, calcola_matrici_gram_fedelta

nome_encoding = "Amplitude Encoding"

def kernel(train_set, test_set, numero_features):
    
    X_train_scalato = normalize(train_set[0], norm="l2")
    X_test_scalato = normalize(test_set[0], norm="l2")

    train_adattato = (X_train_scalato, train_set[1])
    test_adattato = (X_test_scalato, test_set[1])

    numero_qubits_amplitude = int(np.ceil(np.log2(numero_features)))
    dev_amplitude = alloca_device(numero_qubits_amplitude)

    @qml.qnode(dev_amplitude)
    def circuito_quantistico(campione_1, campione_2): 
        qml.AmplitudeEmbedding(features=campione_1, wires=range(numero_qubits_amplitude), normalize=True, pad_with=0.)
        qml.adjoint(qml.AmplitudeEmbedding)(features=campione_2, wires=range(numero_qubits_amplitude), normalize=True, pad_with=0.) 
        return qml.expval(qml.Projector(np.zeros(numero_qubits_amplitude), wires=range(numero_qubits_amplitude)))

    matrici_gram = calcola_matrici_gram_fedelta(X_train_scalato, X_test_scalato, circuito_quantistico)
    return (train_adattato, test_adattato), matrici_gram, numero_qubits_amplitude