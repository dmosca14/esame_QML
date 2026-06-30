#!/bin/bash

# Creiamo una cartella nascosta sul tuo PC per i dati di runtime di Jupyter
# così non sporchiamo la radice del progetto
mkdir -p .jupyter_runtime

docker run -it --rm \
  -p 8888:8888 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.jupyter_runtime \
  jupyter-quantum
