#!/bin/bash
mkdir -p .jupyter_runtime

docker run -it --rm --gpus all \
  --shm-size=2g \
  --cpus="4" \
  -p 8888:8888 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.jupyter_runtime \
  qml-jupyter-env
