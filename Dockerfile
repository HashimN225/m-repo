FROM python:3.11-slim

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# ── Core data science ─────────────────────────────────────────────
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    pandera \
    matplotlib \
    seaborn \
    pyarrow

# ── AWS / MinIO access ────────────────────────────────────────────
RUN pip install --no-cache-dir \
    boto3 \
    python-dotenv

# ── Kubeflow pipeline ─────────────────────────────────────────────
RUN pip install --no-cache-dir \
    kfp \
    kfp-kubernetes \
    kubernetes \
    kubeflow-training

# ── MLflow ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    mlflow

# ── Feast feature store ───────────────────────────────────────────
RUN pip install --no-cache-dir \
    "feast[redis,postgres]" \
    s3fs \
    fsspec

# Copy source code and install project package
COPY setup.py .
COPY src/ ./src/
COPY _feast/ ./_feast/
COPY _kubeflow/ ./_kubeflow/
COPY _mlflow/ ./_mlflow/

RUN pip install --no-cache-dir .
