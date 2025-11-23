# LLM Inference Worker Monitoring Setup

## ✅ Complete Monitoring Stack

### Services Running
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Custom Metrics Exporter**: http://localhost:8000/metrics

### Available Dashboards
1. **NVIDIA GPU Monitoring** - Hardware metrics
2. **LLM Inference Worker Monitoring** - Application metrics

## 🎯 Custom Inference Metrics

### KV Cache Monitoring
- `llm_kv_cache_utilization_percent` - KV cache usage per worker/model
- Tracks memory efficiency of attention mechanisms

### Queue Management
- `llm_queue_depth` - Number of requests waiting per worker
- `llm_active_requests` - Currently processing requests

### Performance Metrics
- `llm_inference_duration_seconds` - Request latency with percentiles
- `llm_tokens_generated_total` - Total output tokens
- `rate(llm_tokens_generated_total[5m])` - Tokens/second rate

### Resource Utilization
- `llm_gpu_memory_utilization_percent` - GPU memory per worker/GPU
- Integration with DCGM hardware metrics

## 🚀 Running Monitored Inference

### Start Metrics Exporter
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/workspace:/workspace \
  --network host \
  -e WORKER_ID=worker-1 \
  -e MODEL_NAME=Qwen2.5-0.5B \
  michaelsigamani/proj-grounded-telescopes:0.1.0 \
  bash -c "
    pip install prometheus_client
    python /workspace/inference_metrics_exporter.py
  "
```

### Multiple Workers
```bash
# Worker 1
WORKER_ID=worker-1 GPU_ID=0 python inference_metrics_exporter.py &

# Worker 2  
WORKER_ID=worker-2 GPU_ID=1 python inference_metrics_exporter.py &
```

## 📊 Key Grafana Panels

### Real-time Monitoring
1. **KV Cache Utilization** - % cache usage (alerts at 70%, 90%)
2. **Queue Depth** - Pending requests per worker
3. **Active Requests** - Concurrent processing
4. **Inference Duration** - P50, P95, P99 latencies
5. **Token Generation Rate** - Throughput monitoring
6. **GPU Memory Utilization** - Per-worker memory usage

### Alerting Thresholds
- **KV Cache**: Warning at 70%, Critical at 90%
- **Queue Depth**: Warning at 5, Critical at 10 requests
- **GPU Memory**: Warning at 80%, Critical at 95%

## 🔍 Prometheus Queries

### Performance Analysis
```promql
# Average inference time
histogram_quantile(0.50, rate(llm_inference_duration_seconds_bucket[5m]))

# 95th percentile latency
histogram_quantile(0.95, rate(llm_inference_duration_seconds_bucket[5m]))

# Throughput per worker
rate(llm_tokens_generated_total[5m]) by (worker_id)

# Queue depth trends
llm_queue_depth by (worker_id)

# KV cache efficiency
llm_kv_cache_utilization_percent by (worker_id, model_name)
```

### Capacity Planning
```promql
# Total tokens per minute
sum(rate(llm_tokens_generated_total[1m])) * 60

# GPU memory utilization
llm_gpu_memory_utilization_percent by (worker_id, gpu_id)

# Concurrent request capacity
sum(llm_active_requests) by (worker_id)
```

## 🛠️ Configuration Files

- `/root/workspace/prometheus.yml` - Prometheus scraping config
- `/root/workspace/docker-compose-monitoring.yml` - Monitoring stack
- `/root/workspace/inference_metrics_exporter.py` - Custom metrics
- `/root/workspace/setup_inference_dashboard.sh` - Grafana datasource + dashboards

## 📈 Scaling Considerations

### Adding Workers
1. Deploy metrics exporter on each worker
2. Set unique `WORKER_ID` environment variable
3. Prometheus auto-discovers new targets
4. Grafana automatically aggregates metrics

### Multi-GPU Setup
- Metrics labeled by `gpu_id` and `worker_id`
- Track per-GPU utilization
- Monitor load balancing across GPUs

Your comprehensive inference monitoring is now ready! The system provides real-time visibility into KV cache utilization, queue depths, and performance metrics across all workers.