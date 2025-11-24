# Ray + vLLM Multi-Node Batch Inference System

## 🚀 IMPORTANT: Ray Data Batch Inference

This project uses **Ray Data** for distributed batch inference, following to official documentation:
https://docs.ray.io/en/latest/data/batch_inference.html

### Key Pattern:
```python
# Ray Data map_batches for distributed vLLM inference
ds.map_batches(
    vllm_inference,
    num_gpus=1,  # Each actor gets 1 GPU
    concurrency=2,  # 2 parallel actors (one per node)
    batch_size=4   # Optimize batch size for GPU utilization
)
```

This approach provides:
- **Intelligent GPU orchestration** across nodes
- **Automatic load balancing** via Ray scheduler
- **Scalable batch processing** with optimal resource utilization
- **Fault tolerance** with automatic actor recovery

Production-ready batch inference stack for running small language models across multiple GPU
workers with monitoring, load balancing, and SLA tracking.

## Overview
- **Batch Inference:** `app/batch_inference_vllm_fixed.py` provides production-ready
  distributed batch inference using Ray Data + vLLM with 24-hour SLA monitoring.
- **Inference API:** `app/inference_api_server.py` serves single and batch text
  generation, exposes Prometheus metrics on port 8001, and powers the
  client-facing REST interface.
- **Client CLI:** `app/llm_client.py` exercises the API for health checks, single
  prompts, batch prompts, or reading prompts from a file.
- **Monitoring:** Built-in SLA tracking with real-time metrics and optional
  Prometheus/Grafana stack via `config/docker-compose-monitoring.yml`.
- **Ray Cluster Utilities:** `app/ray_cluster_fixed.py` helpers support forming a
  multi-node Ray cluster for distributed workers.

## Prerequisites
- Python 3.10+
- GPU with recent NVIDIA drivers (tested on RTX 3090)
- Docker for containerized deployment
- Access to Hugging Face model `Qwen/Qwen2.5-0.5B`
- SSH access to rented Ubuntu VM (for production deployment)

## Local Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn torch transformers prometheus-client requests ruff pytest
```

## Production Deployment (2-Node Ray Cluster)

### Prerequisites
- SSH access to rented Ubuntu VM with Docker installed
- `michaelsigamani/proj-grounded-telescopes:0.1.0` Docker image

### Quick Start
```bash
# 1. Connect to head node (port 59554)
ssh -p 59554 root@77.104.167.149

# 2. Start Ray cluster (if not running)
docker exec ray-head-new bash -c "cd /app/doubleword-technical && python batch_inference_final.py --mode head"

# 3. Connect worker node (port 55089)  
ssh -p 55089 root@77.104.167.149

# 4. Start worker (if not running)
docker run -d --name ray-worker --gpus all -p 8002:8000 \
  michaelsigamani/proj-grounded-telescopes:0.1.0

# 5. Join worker to cluster
docker exec ray-worker bash -c "cd /app/doubleword-technical && python batch_inference_final.py --mode worker --head-address 77.104.167.149:6379"

# 6. Run batch inference
docker exec ray-head-new bash -c "cd /app/doubleword-technical && python batch_inference_final.py --mode inference"
```

### Cluster Status
```bash
# Check cluster status
docker exec ray-head-new ray status

# Access Ray dashboard (via SSH tunnel)
ssh -p 59554 root@77.104.167.149 \
  -L 8265:localhost:8265
# Then open http://localhost:8265 in browser
```

### Monitoring & SLA
- **Built-in SLA tracking** with 24-hour target
- **Real-time metrics** via built-in Prometheus endpoint (port 8001)
- **Progress tracking** with ETA calculations and alerts
- **Optional Grafana** stack via `config/docker-compose-monitoring.yml`

## Local Development
```bash
python app/inference_api_server.py
```
- REST API available at `http://localhost:8000`
- Prometheus metrics exposed at `http://localhost:8001/metrics`
- Default worker ID `gpu-worker-1`; override via `WORKER_ID` env var

## Client Commands
```bash
python app/llm_client.py --server http://localhost:8000 --health
python app/llm_client.py --server http://localhost:8000 --prompt "What is AI?"
python app/llm_client.py --server http://localhost:8000 --batch
python app/llm_client.py --server http://localhost:8000 --file prompts.txt --batch
```

## Client Commands
```bash
# Test API health
python app/llm_client.py --server http://localhost:8000 --health

# Single inference
python app/llm_client.py --server http://localhost:8000 --prompt "What is AI?"

# Batch inference
python app/llm_client.py --server http://localhost:8000 --batch

# Batch from file
python app/llm_client.py --server http://localhost:8000 --file prompts.txt --batch
```

## Monitoring Stack
```bash
cd config
docker compose -f docker-compose-monitoring.yml up -d
```
- Grafana: `http://localhost:3000` (default creds `admin/admin123`)
- Prometheus: `http://localhost:9090`
- Node exporter: `http://localhost:9100`

## Quick Start
```bash
# Make setup script executable
chmod +x setup_ray_cluster.sh

# Start full cluster
./setup_ray_cluster.sh start-cluster

# Check status
./setup_ray_cluster.sh status

# Run batch inference
./setup_ray_cluster.sh inference

# Setup SSH tunnels (in separate terminals)
./setup_ray_cluster.sh tunnels
```

## Testing & Validation
```bash
# Run SLA validation test
python app/test_sla_validation.py

# Performance testing
python app/test_ray_vllm_minimal.py

# End-to-end tests
python -m pytest tests/test_batch_inference_e2e.py
```

## Multi-Node Ray Cluster
Head node (already running inside the main container):
```bash
docker exec inference-server ray start --head --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 --redis-password=ray123
```
Worker node (second machine):
```bash
docker pull michaelsigamani/proj-grounded-telescopes:0.1.0
docker run -d --name ray-worker --gpus all -p 8002:8000 \
  michaelsigamani/proj-grounded-telescopes:0.1.0
docker exec ray-worker ray start --address='77.104.167.148:6379' \
  --redis-password='ray123'
docker exec ray-worker bash -c "cd /workspace && python ray_cluster_fixed.py \
  worker 77.104.167.148:6379"
```
Dashboard: `http://77.104.167.148:8265`

## SSH Port Forwarding Shortcut
```bash
ssh -p 40195 root@77.104.167.148 \
  -L 8000:localhost:8000 \
  -L 8001:localhost:8001 \
  -L 8080:localhost:8080 \
  -L 3000:localhost:3000 \
  -L 9090:localhost:9090 \
  -L 8265:localhost:8265
```

## Testing & Linting
```bash
python -m pytest                # Full suite
python -m pytest app/tests/...  # Single test
ruff check app                  # Lint
ruff format app                 # Format
```

## Repository Layout
```
app/      FastAPI server, Ray helpers, clients, tests
config/   Monitoring stack, Nginx configuration, setup scripts
docs/     Deployment guides, debugging notes, monitoring references
```

## Useful References
- Detailed monitoring and setup notes live under `docs/`
- Use `app/quick_test.py` to smoke-test the model on a single GPU
- `app/test_api.py` offers a scripted smoke test against a running server
