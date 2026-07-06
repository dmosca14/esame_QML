import os
from .configurazione_cartella_run import cartella_run_attuale
from .addestramento import addestramento
from .visualizzazione import test_e_visualizzazione

def costruisci_qsvm(train_set_raw, test_set_raw, numero_features, modulo_encoding, iperparametri_quantistici, iperparametri_classici, nome_dataset=""):

    # Creazione della cartella relativa alla run attuale.

    nome_encoding = getattr(modulo_encoding, "nome_encoding")
    nome_pulito_encoding = nome_encoding.replace(" ", "_").lower() 
   
    timestamp_run = os.path.basename(cartella_run_attuale)

    nome_cartella_encoding = f"{nome_pulito_encoding}_{timestamp_run}"
    percorso_finale = os.path.join(cartella_run_attuale, nome_cartella_encoding)
    os.makedirs(percorso_finale, exist_ok = True)

    # Costruzione della QSVM.
    
    print(f"\nDestinazione output: {percorso_finale}")

    # Addestramento e validazione della QSVM.

    set_adattato, matrici_gram, modello, storico = addestramento(
        nome_encoding, train_set_raw, test_set_raw, numero_features, modulo_encoding, 
        iperparametri_quantistici, iperparametri_classici, percorso_finale, nome_dataset
    )

    # Test della miglior QSVM e visualizzazione dei risultati.

    test_e_visualizzazione(nome_encoding, set_adattato, matrici_gram, modello, percorso_finale, nome_dataset)

    return modello, set_adattato, matrici_gram