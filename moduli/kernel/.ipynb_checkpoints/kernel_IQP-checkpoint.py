import numpy as np
import pennylane as qml
from .base import applica_minmax_con_margine, alloca_device, calcola_matrici_gram_fedelta

nome_encoding = "IQP Encoding"

def kernel(train_set, test_set, numero_features, numero_ripetizioni, pattern):
    train_adattato, test_adattato = applica_minmax_con_margine(
        train_set, test_set, feature_range=(-0.9, 0.9), limite_inf=-1.0, limite_sup=1.0
    )

    dev_IQP = alloca_device(numero_features)

    @qml.qnode(dev_IQP)
    def circuito_quantistico(campione_1, campione_2): 
        qml.IQPEmbedding(features=campione_1, wires=range(numero_features), n_repeats=numero_ripetizioni, pattern=pattern)
        qml.adjoint(qml.IQPEmbedding)(features=campione_2, wires=range(numero_features), n_repeats=numero_ripetizioni, pattern=pattern) 
        return qml.expval(qml.Projector(np.zeros(numero_features), wires=range(numero_features)))
    
    matrici_gram = calcola_matrici_gram_fedelta(train_adattato[0], test_adattato[0], circuito_quantistico)
    return (train_adattato, test_adattato), matrici_gram, numero_features