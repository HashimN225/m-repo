# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /install
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: Runtime (lighter)
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY setup.py .
COPY src/ ./src/
COPY _feast/ ./_feast/
COPY _kubeflow/ ./_kubeflow/
COPY _mlflow/ ./_mlflow/

RUN pip install --no-cache-dir .