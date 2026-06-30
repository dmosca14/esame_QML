import pandas as pd
import numpy as np
from .metriche import kernel_target_alignment, coefficiente_geometrico
from moduli.kernel import kernel_rbf
from .qsvm import costruisci_qsvm

def esegui_screening_e_pipeline(
    set_preparato, 
    numero_features, 
    modulo_encoding_quantistico, 
    griglia_quantistica, 
    iperparametri_classici, 
    soglia_kta=0.05
):
    """
    Esegue lo screening geometrico (KTA e Geometric Difference) su una griglia di
    iperparametri quantistici. Addestra la QSVM solo sulle configurazioni che superano
    la soglia critica di allineamento del kernel.
    """
    
    # 1. CALCOLO BASELINE CLASSICA (RBF)
    print("=== FASE 1: Calcolo Baseline Classica ===")
    _, matrici_gram_rbf = kernel_rbf.kernel(set_preparato)
    K_classico_train = matrici_gram_rbf[0]
    
    risultati_screening = []
    
    print("\n=== FASE 2: Screening Geometrico degli Spazi di Hilbert ===")
    
    # Estrazione delle liste dalla griglia quantistica.
    lista_ripetizioni = griglia_quantistica.get('numero_ripetizioni', [1])
    lista_pattern = griglia_quantistica.get('pattern', [None])
    
    for rep in lista_ripetizioni:
        for pat in lista_pattern:
            print(f"\nValutazione configurazione: {rep} ripetizioni, pattern '{pat}'")
            
            # Generazione temporanea delle matrici quantistiche per l'analisi spaziale.
            # Il blocco try/except garantisce la compatibilità con eventuali encoding 
            # (es. Angle o Amplitude) che non accettano rep e pat nella loro firma.
            try:
                set_adattato_q, matrici_gram_q = modulo_encoding_quantistico.kernel(
                    set_preparato, numero_features, rep, pat
                )
            except TypeError:
                set_adattato_q, matrici_gram_q = modulo_encoding_quantistico.kernel(
                    set_preparato, numero_features
                )
            
            K_quantistico_train = matrici_gram_q[0]
            y_train = set_adattato_q[0][1] 
            
            # Calcolo delle metriche
            kta = kernel_target_alignment(K_quantistico_train, y_train)
            g_diff = coefficiente_geometrico(K_classico_train, K_quantistico_train)
            
            passa_filtro = kta >= soglia_kta
            print(f"  > KTA: {kta:.4f} | Geometric Difference: {g_diff:.4f} | Approvato: {passa_filtro}")
            
            risultati_screening.append({
                "numero_ripetizioni": rep,
                "pattern": pat,
                "KTA": kta,
                "Geometric_Difference": g_diff,
                "Passa_Filtro": passa_filtro
            })
            
            # FASE 3: ADDESTRAMENTO SELETTIVO
            if passa_filtro:
                print(f"  > Configurazione valida. Inizializzazione QSVM...")
                
                # CORREZIONE CHIAVE: Scikit-Learn ParameterGrid esige liste. 
                # Incapsuliamo rep e pat per rispettare i requisiti di addestramento.py
                iperparametri_quantistici_correnti = {
                    'numero_ripetizioni': [rep],
                    'pattern': [pat]
                }
                
                # Invocazione dell'infrastruttura originale
                costruisci_qsvm(
                    set_preparato,
                    numero_features,
                    modulo_encoding_quantistico,
                    iperparametri_quantistici_correnti,
                    iperparametri_classici
                )
            else:
                print(f"  > [ALERT] Configurazione scartata: Expressibility sterile o collasso numerico. Salto il training classico.")
                
    # Tabulazione finale
    df_report = pd.DataFrame(risultati_screening)
    return df_report