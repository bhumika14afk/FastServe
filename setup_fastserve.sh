#!/bin/bash
# FastServe project scaffold
# Run this from inside your ~/projects/FastServe directory (with venv activated)

set -e

echo "Scaffolding FastServe repo structure..."

# Core directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/baseline
mkdir -p models/quantized
mkdir -p models/distilled
mkdir -p serving
mkdir -p eval
mkdir -p benchmarks
mkdir -p notebooks
mkdir -p scripts
mkdir -p configs
mkdir -p tests

# .gitkeep files so empty dirs are tracked by git
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/baseline/.gitkeep
touch models/quantized/.gitkeep
touch models/distilled/.gitkeep

# Starter files
cat > serving/api.py << 'EOF'
"""
FastAPI gateway for FastServe.
Serves requests to whichever model backend is configured (baseline, quantized, distilled).
"""
from fastapi import FastAPI

app = FastAPI(title="FastServe API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/triage")
def triage_ticket(ticket_text: str):
    """
    Classify a support ticket, extract entities, and draft a first response.
    TODO: wire up model backend.
    """
    raise NotImplementedError("Model backend not yet connected.")
EOF

cat > eval/harness.py << 'EOF'
"""
Evaluation harness for FastServe model variants.
Measures task accuracy, latency, throughput, and cost per 1K requests.
"""


def evaluate_accuracy(model, eval_set):
    """Run the model over the labeled eval set and return accuracy/F1."""
    raise NotImplementedError


def evaluate_latency(model, num_requests: int = 100):
    """Measure p50/p95/p99 latency over N requests."""
    raise NotImplementedError
EOF

cat > scripts/quantize.py << 'EOF'
"""
Quantize a baseline model using GPTQ or AWQ.
Run this on a GPU instance (Colab / cloud GPU), not locally.
"""

if __name__ == "__main__":
    print("TODO: implement GPTQ/AWQ quantization pipeline")
EOF

cat > scripts/distill.py << 'EOF'
"""
Knowledge distillation: train a smaller student model
to replicate the baseline model's task performance.
Run this on a GPU instance (Colab / cloud GPU), not locally.
"""

if __name__ == "__main__":
    print("TODO: implement distillation training loop")
EOF

cat > configs/model_config.yaml << 'EOF'
# Model configuration for FastServe variants
baseline:
  name: "meta-llama/Meta-Llama-3-8B"
  precision: "fp16"

quantized:
  method: "gptq"  # or "awq"
  bits: 4

distilled:
  base_size: "1B"
  teacher: "baseline"
EOF

cat > requirements.txt << 'EOF'
fastapi
uvicorn
transformers
huggingface_hub
torch
datasets
mlflow
locust
pyyaml
pytest
EOF

cat > .gitignore << 'EOF'
venv/
__pycache__/
*.pyc
.env
data/raw/*
data/processed/*
models/baseline/*
models/quantized/*
models/distilled/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!models/baseline/.gitkeep
!models/quantized/.gitkeep
!models/distilled/.gitkeep
.mlflow/
mlruns/
EOF

cat > tests/test_api.py << 'EOF'
from fastapi.testclient import TestClient
from serving.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
EOF

echo "Done. Folder structure created:"
find . -maxdepth 2 -type d | sort
