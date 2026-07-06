import numpy as np
import pennylane as qml
from .base import applica_minmax_con_margine, alloca_device, calcola_matrici_gram_fedelta

nome_encoding = "Angle Encoding"

def kernel(train_set, test_set, numero_features):
    
    margine = 0.05 * np.pi
    train_adattato, test_adattato = applica_minmax_con_margine(
        train_set, test_set, feature_range=(margine, np.pi - margine), limite_inf=0.0, limite_sup=np.pi
    )

    dev_angle = alloca_device(numero_features)

    @qml.qnode(dev_angle)
    def circuito_quantistico(campione_1, campione_2):
        qml.AngleEmbedding(campione_1, wires=range(numero_features), rotation="Y") 
        qml.adjoint(qml.AngleEmbedding)(campione_2, wires=range(numero_features), rotation="Y") 
        return qml.expval(qml.Projector(np.zeros(numero_features), wires=range(numero_features)))

    matrici_gram = calcola_matrici_gram_fedelta(train_adattato[0], test_adattato[0], circuito_quantistico)
    return (train_adattato, test_adattato), matrici_gram, numero_features