# Multi-Node LLM Inference Setup Summary

## Overview
This session covered setting up a distributed LLM inference system with load balancing and monitoring capabilities.

## Architecture Implemented

### 1. Single Machine Setup (Completed)
- **Docker Image**: `michaelsigamani/proj-grounded-telescopes:0.1.0`
- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **GPU**: NVIDIA GeForce RTX 3090 (24GB)

### 2. Multiple Workers on Same Node (Completed)
- **Worker 1**: `http://localhost:8000` (gpu-worker-1)
- **Worker 2**: `http://localhost:8001` (gpu-worker-2)
- **Load Balancer**: Nginx on `http://localhost:8080`

### 3. Monitoring Stack (Completed)
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin/admin123)
- **Ray Dashboard**: `http://localhost:8265`

## Key Components

### Inference Server (`inference_api_server.py`)
- FastAPI-based REST API
- Prometheus metrics integration
- Health checks and monitoring
- Support for single and batch inference

### Load Balancer (`nginx.conf`)
- Round-robin load distribution
- Health checks
- Request timeout handling

### Client (`llm_client.py`)
- Command-line interface
- Support for single and batch requests
- Performance metrics reporting

## Current Status

### ✅ Completed
1. **Docker Environment**: Multi-container setup with GPU support
2. **Inference API**: RESTful endpoints for text generation
3. **Load Balancing**: Nginx distributing requests across workers
4. **Monitoring**: Prometheus + Grafana dashboards
5. **Health Checks**: Service availability monitoring

### 🔄 In Progress
1. **Multi-Node Ray Cluster**: Head node configured, worker setup pending

### ⏳ Pending
1. **Ray Serve Integration**: Distributed deployment across nodes
2. **Autoscaling**: Dynamic resource allocation
3. **Advanced Metrics**: KV cache utilization, custom dashboards

## Access Points

### From MacBook Pro (with SSH forwarding)
```bash
ssh -p 40195 root@77.104.167.148 \
  -L 8000:localhost:8000 \
  -L 8001:localhost:8001 \
  -L 8080:localhost:8080 \
  -L 3000:localhost:3000 \
  -L 9090:localhost:9090 \
  -L 8265:localhost:8265
```

### Services
- **Load Balanced API**: `http://localhost:8080`
- **Direct Worker 1**: `http://localhost:8000`
- **Direct Worker 2**: `http://localhost:8001`
- **Grafana Dashboard**: `http://localhost:3000`
- **Prometheus**: `http://localhost:9090`
- **Ray Dashboard**: `http://localhost:8265`

## Usage Examples

### Client Usage
```bash
# Single request through load balancer
python llm_client.py --server http://localhost:8080 --prompt "What is AI?"

# Direct request to specific worker
python llm_client.py --server http://localhost:8000 --prompt "What is machine learning?"

# Health check
python llm_client.py --server http://localhost:8080 --health
```

### API Usage
```bash
# Direct API call
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?", "max_tokens": 50}'
```

## Multi-Node Ray Cluster Setup

### Head Node (Current Machine: 77.104.167.148)
```bash
# Already running
docker exec inference-server ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --redis-password=ray123
```

### Worker Node Setup (Second Machine)
```bash
# 1. Pull Docker image
docker pull michaelsigamani/proj-grounded-telescopes:0.1.0

# 2. Start worker container
docker run -d --name ray-worker --gpus all -p 8002:8000 michaelsigamani/proj-grounded-telescopes:0.1.0

# 3. Connect to Ray cluster
docker exec ray-worker ray start --address='77.104.167.148:6379' --redis-password='ray123'
```

## Performance Metrics

### Available Prometheus Metrics
- `llm_api_requests_total` - Total API requests
- `llm_api_inference_duration_seconds` - Inference timing
- `llm_api_active_connections` - Active connections
- `llm_api_queue_size` - Request queue size

### GPU Monitoring
- GPU Memory Usage: ~0.93GB / 23.56GB
- GPU Utilization: ~4%
- Model Loading Time: ~30 seconds per worker

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Ensure ports 8000, 8001, 8080 are available
2. **GPU Memory**: Monitor with `nvidia-smi` in containers
3. **Ray Connection**: Verify network connectivity between nodes
4. **Model Loading**: Check HuggingFace model access

### Solutions
1. **Port Forwarding**: Update SSH command to include all required ports
2. **Container Restart**: Use `docker restart <container-name>`
3. **Ray Status**: Check with `ray status` in containers
4. **Health Checks**: Use `/health` endpoints for monitoring

## Next Steps

### Immediate
1. **Complete Multi-Node Setup**: Add second machine to Ray cluster
2. **Ray Serve Deployment**: Implement distributed inference service
3. **Autoscaling Configuration**: Set up dynamic resource allocation

### Future Enhancements
1. **Model Versioning**: Support multiple model versions
2. **Request Queuing**: Advanced queue management
3. **Caching**: Response caching for repeated queries
4. **Security**: Authentication and authorization
5. **Performance Optimization**: Tensor parallelism, model sharding

## File Structure
```
/workspace/
├── inference_api_server.py     # FastAPI inference server
├── llm_client.py              # Python client
├── nginx.conf                 # Load balancer config
├── ray_cluster_fixed.py       # Ray cluster setup
├── docker-compose-monitoring.yml # Monitoring stack
├── prometheus.yml             # Prometheus config
└── SETUP_SUMMARY.md          # This file
```

## Commands Reference

### Docker Management
```bash
# Start containers
docker run -d --name inference-server -p 8000:8000 --gpus all michaelsigamani/proj-grounded-telescopes:0.1.0

# Check status
docker ps | grep inference

# View logs
docker logs inference-server

# Stop containers
docker stop inference-server inference-worker-2 nginx-lb
```

### Ray Management
```bash
# Start head node
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# Connect worker
ray start --address='HEAD_IP:6379' --redis-password='ray123'

# Check cluster status
ray status

# Stop Ray
ray stop
```

### Monitoring
```bash
# Start monitoring stack
docker-compose -f docker-compose-monitoring.yml up -d

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Access Grafana
http://localhost:3000 (admin/admin123)
```

---
*Generated on: 2025-11-23*
*Session Duration: ~2 hours*
*Key Achievement: Working multi-container inference system with load balancing and monitoring*