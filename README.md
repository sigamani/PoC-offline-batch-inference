# Ray Data + vLLM Batch Inference

## Overview

Build a production-ready offline batch inference server using Ray Data and vLLM that processes 1000+ requests within 24-hour SLA windows with proper authentication, monitoring, and observability.

---

## Architecture Overview

```
┌─────────────────────────────┐
│        Users / Clients      │
│ (Bearer Token Auth)        │
└─────────────┬──────────────┘
              │
              ▼
┌─────────────────────────────┐
│       Head Node (CPU)       │
│                             │
│  FastAPI Gateway            │
│  - JWT Authentication        │
│  - Rate Limiting            │
│  - Job Queue Management      │
│  - SLA & Metrics Tracker      │
│  - Real-time Monitoring       │
│                             │
└─────────────┬──────────────┘
              │
              ▼
┌───────────────────────────────┐
│   Job Dispatcher / Worker   │
│  - Queue Processing           │
│  - Concurrency Control        │
│  - Artifact Storage          │
└─────────────┬──────────────┘
              │
              ▼
┌───────────────────────────────┐
│       GPU Worker Nodes         │
│                               │
│  Ray Workers                  │
│  - vLLM Engine               │
│  - Adaptive Batching          │
│  - Memory Optimization        │
│  - Model Sharding             │
└─────────────┬─────────────────┘
              │
              ▼
┌───────────────────────────────┐
│   Batch Output Storage   │
│  - SHA-based Artifacts      │
│  - Version Control           │
│  - S3 Integration           │
└─────────────────────────────┘
```

---

## Key Features

### **Authentication & Security**
- **JWT-based Authentication**: Bearer tokens with expiration and tier-based rate limiting
- **Rate Limiting**: Per-tier request limits (Free: 100/hr, Basic: 500/hr, Premium: 2000/hr, Enterprise: 10000/hr)
- **Token Management**: Generation, validation, revocation, and blacklisting

### **Batch Processing**
- **Ray Data Integration**: Official `ray.data.llm` API with `vLLMEngineProcessorConfig`
- **Adaptive Batching**: Dynamic batch size calculation based on available memory
- **vLLM Optimization**: Chunked prefill, speculative decoding, KV cache optimization
- **Multimodal Support**: Text, image, audio, and video processing capabilities

### **SLA Management**
- **Tier-based SLAs**: Free (72h), Basic (24h), Premium (12h), Enterprise (6h)
- **Real-time Monitoring**: Progress tracking, ETA calculation, SLA risk assessment
- **Automated Alerting**: SLA breach warnings and notifications

### **Data Management**
- **SHA-based Artifacts**: Immutable data with content and SHA256 hashing
- **Version Control**: Timestamped versions with audit trails
- **Storage Options**: Local filesystem with S3 cloud integration

### **Observability**
- **Prometheus Metrics**: Throughput, tokens/sec, memory usage, error rates
- **Grafana Dashboard**: Real-time monitoring and alerting
- **Structured Logging**: JSON format for Loki integration
- **Health Endpoints**: Service status and component health checks

---

## Performance Characteristics

### **Throughput Targets**
- **Baseline**: 250-300 requests/second (128 batch, 2 concurrency)
- **Optimized**: 500-800 requests/second (256 batch, 4 concurrency)
- **Maximum**: 1000+ requests/second (stress testing)

### **Memory Requirements**
- **Qwen2.5-0.5B**: ~2GB per GPU
- **Qwen2.5-7B**: ~4GB per GPU (requires sharding)
- **Qwen2.5-13B**: ~8GB per GPU (requires sharding)

### **SLA Compliance**
- **24-hour Window**: Standard for all tiers
- **Burst Handling**: Queue-based load balancing
- **Failure Recovery**: Automatic retry with exponential backoff

---

## Quick Start

### **Prerequisites**
```bash
# Clone repository
git clone https://github.com/sigamani/doubleword-technical.git
cd doubleword-technical

# Build and start services
docker-compose up -d

# Check deployment
curl http://localhost:8000/health
```

### **Configuration**
```yaml
# config/config.yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  max_model_len: 32768
  tensor_parallel_size: 2

inference:
  batch_size: 256
  concurrency: 4
  max_tokens: 512
  gpu_memory_utilization: 0.90
  enable_chunked_prefill: true
  enable_speculative_decoding: true

sla:
  target_hours: 24
  buffer_factor: 0.7
  alert_threshold_hours: 20

storage:
  local_path: "/tmp/artifacts"
  s3_bucket: "batch-inference-artifacts"

monitoring:
  log_level: "INFO"
  prometheus_port: 8001
  grafana_enabled: true
```

### **API Usage**
```bash
# Submit batch job
curl -X POST "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/tmp/input.json",
    "output_path": "/tmp/output.json",
    "num_samples": 1000,
    "batch_size": 256,
    "concurrency": 4,
    "sla_tier": "basic"
  }' \
  http://localhost:8000/start

# Check job status
curl http://localhost:8000/jobs/{job_id}

# Get metrics
curl http://localhost:8000/metrics
```

---

## 📈 Monitoring & Alerting

### **Key Metrics**
- **Throughput**: Requests per second, tokens per second
- **Latency**: P50, P95, P99 response times
- **Memory**: GPU utilization, memory usage per request
- **Error Rate**: Failed requests per minute
- **SLA Compliance**: Jobs meeting 24-hour deadline

### **Alert Thresholds**
- **SLA Risk**: ETA > remaining time + 0.1h buffer
- **High Memory**: GPU utilization > 90%
- **Low Throughput**: < 5 requests/second
- **High Error Rate**: > 5% failure rate

---

## Deployment Options

### **Single Node (Development)**
```bash
# Start all services
docker-compose up -d
```

```
```

---

## Testing

### **Test Matrix**
- **Configurations**: 5 batch sizes × 4 concurrency levels × 3 model sizes = 60 permutations
- **Scenarios**: Baseline, high load, stress, SLA validation
- **Automation**: Testing Symbiote agent for comprehensive validation

### **Quality Gates**
- **All Tests Pass**: 100% success rate required for deployment
- **SLA Met**: 99%+ jobs within 24-hour window
- **Performance Benchmarks**: Meet or exceed target throughput
- **Security Validated**: All authentication and rate limiting tests pass

---

*Last Updated: 2025-11-25*
*Status: Production Ready with Comprehensive Architecture*