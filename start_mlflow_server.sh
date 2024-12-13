#!/bin/bash

# Set project directory and MLflow server configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLFLOW_BACKEND_URI="sqlite:///${PROJECT_DIR}/mlflow/mlflow.db"
MLFLOW_ARTIFACT_ROOT="${PROJECT_DIR}/mlflow/artifacts"

# Create necessary directories if they don't exist
mkdir -p "${PROJECT_DIR}/mlflow/artifacts"

# Start MLflow server
mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port 5000