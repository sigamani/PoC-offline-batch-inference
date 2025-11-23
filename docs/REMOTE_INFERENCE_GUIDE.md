# Remote Batch Inference Setup Guide

## 🚀 Server Setup (Linux Machine)

### 1. Start the API Server
```bash
# Get your server IP
hostname -I | awk '{print $1}'

# Start the inference API server
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/workspace:/workspace \
  --network host \
  -e WORKER_ID=gpu-server-1 \
  -e MODEL_NAME=Qwen2.5-0.5B \
  -d \
  michaelsigamani/proj-grounded-telescopes:0.1.0 \
  bash -c "
    pip install fastapi uvicorn pydantic prometheus_client
    python /workspace/inference_api_server.py
  "
```

### 2. Verify Server is Running
```bash
# Check health endpoint
curl http://localhost:8000/health

# Check logs
docker logs $(docker ps -q --filter "ancestor=michaelsigamani/proj-grounded-telescopes:0.1.0")
```

## 💻 Client Setup (MacBook Pro)

### 1. Install Dependencies
```bash
pip install requests
```

### 2. Copy Client Script
Copy `/root/workspace/llm_client.py` to your MacBook Pro

### 3. Test Connection
```bash
# Health check
python llm_client.py --server http://YOUR_SERVER_IP:8000 --health

# Single prompt
python llm_client.py --server http://YOUR_SERVER_IP:8000 --prompt "What is AI?"

# Batch processing
python llm_client.py --server http://YOUR_SERVER_IP:8000 --batch

# From file
python llm_client.py --server http://YOUR_SERVER_IP:8000 --file prompts.txt --batch
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "worker_id": "gpu-server-1",
  "model": "Qwen2.5-0.5B",
  "device": "cuda:0",
  "gpu_memory_used_gb": 0.93,
  "gpu_memory_total_gb": 24.0,
  "gpu_utilization": 3.9,
  "queue_size": 0
}
```

### Single Generation
```bash
POST /generate
Content-Type: application/json

{
  "prompt": "What is artificial intelligence?",
  "max_tokens": 100,
  "temperature": 0.7
}
```

### Batch Generation
```bash
POST /generate_batch
Content-Type: application/json

{
  "prompts": [
    "What is AI?",
    "Explain ML.",
    "Neural networks?"
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

## 🔍 Monitoring

### Grafana Dashboards
- **LLM Inference Worker Monitoring**: http://YOUR_SERVER_IP:3000
- **NVIDIA GPU Monitoring**: http://YOUR_SERVER_IP:3000

### Prometheus Metrics
- **API Metrics**: http://YOUR_SERVER_IP:8001/metrics
- **Prometheus**: http://YOUR_SERVER_IP:9090

### Key Metrics to Monitor
- `llm_api_requests_total` - API request count
- `llm_api_inference_duration_seconds` - Request latency
- `llm_api_active_connections` - Concurrent connections
- `llm_api_queue_size` - Request queue depth

## 🛠️ Troubleshooting

### Connection Issues
1. **Check firewall**: Ensure port 8000 is open
2. **Verify IP**: Use correct server IP address
3. **Network**: Ensure MacBook can reach server

### Performance Issues
1. **GPU Memory**: Monitor GPU utilization in Grafana
2. **Queue Depth**: Check for request bottlenecks
3. **Batch Size**: Adjust batch size for optimal throughput

### Common Errors
- **Connection refused**: Server not running or wrong port
- **Timeout**: Request too large or server overloaded
- **CORS errors**: Check CORS configuration

## 📝 Example Usage

### Python Script on MacBook
```python
import requests

server_url = "http://YOUR_SERVER_IP:8000"

# Health check
health = requests.get(f"{server_url}/health").json()
print(f"Server status: {health['status']}")

# Batch request
prompts = [
    "What is machine learning?",
    "How do neural networks work?",
    "Explain deep learning."
]

response = requests.post(f"{server_url}/generate_batch", json={
    "prompts": prompts,
    "max_tokens": 100,
    "temperature": 0.7
}).json()

for i, result in enumerate(response['results']):
    print(f"{i+1}. {result['text']}")
```

### cURL Examples
```bash
# Health check
curl http://YOUR_SERVER_IP:8000/health

# Single generation
curl -X POST http://YOUR_SERVER_IP:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?", "max_tokens": 50}'
```

## 🔒 Security Considerations

1. **Firewall**: Only expose necessary ports
2. **Authentication**: Add API keys for production
3. **Rate Limiting**: Implement request throttling
4. **CORS**: Restrict to your MacBook's IP in production

Your remote batch inference system is now ready! You can send requests from your MacBook Pro and monitor everything through Grafana.