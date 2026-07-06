import numpy as np
from sklearn.model_selection import train_test_split

def importa_dataset(dataset_scelto, frazione_train):
    X, y = dataset_scelto

    # Split con stratify (si usa sia per test e split, sia per test e val dentro il 5 fold CV). 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=frazione_train, stratify=y, random_state=42
    )
    return (X_train, y_train), (X_test, y_test)