FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ipython \
    numpy \ 
    scipy \
    matplotlib \
    qiskit \
    qiskit-aer \
    pennylane \
    jupyterlab \
    notebook \
    scikit-learn \
    pandas \
    pennylane-lightning \
    pennylane-lightning-gpu \
    cuquantum-cu12 \
    seaborn \
    tabulate \ 
    tqdm \
    openpyxl \
    joblib

# Configurazione ottimale per evitare conflitti con joblib
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1

EXPOSE 8888
WORKDIR /workspace
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
