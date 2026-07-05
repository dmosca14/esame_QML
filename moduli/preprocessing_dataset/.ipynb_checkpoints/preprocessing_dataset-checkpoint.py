from . import importazione_dataset

def preprocessa_dataset(dataset_scelto, frazione_dati_per_train):
    print("\nIMPORTAZIONE DATASET\n")
    set_importato = importazione_dataset.importa_dataset(
        dataset_scelto, frazione_dati_per_train
    )
    print("Importazione del dataset completata.\n")
    return set_importato