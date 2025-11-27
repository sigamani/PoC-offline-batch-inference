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
└─────────────────────────────┘
```

---

## Key Features

### **Authentication & Security**
- **Rate Limiting**: Per-tier request limits (Free: 100/hr, Basic: 500/hr, Premium: 2000/hr, Enterprise: 10000/hr)
- **Token Management**: Generation, validation, revocation, and blacklisting

### **Batch Processing**
- **Ray Data Integration**: Official `ray.data.llm` API with `vLLMEngineProcessorConfig`
- **Adaptive Batching**: Dynamic batch size calculation based on available memory
- **vLLM Optimization**: Chunked prefill, speculative decoding, KV cache optimization


### **Data Management**
- **SHA-based Artifacts**: Immutable data with content and SHA256 hashing
- **Version Control**: Timestamped versions with audit trails

### **Observability**
- **Prometheus Metrics**: Throughput, tokens/sec, memory usage, error rates
- **Grafana Dashboard**: Real-time monitoring and alerting
- **Structured Logging**: JSON format for Loki integration
- **Health Endpoints**: Service status and component health checks

---

### **SLA Compliance**
- **24-hour Window**: Standard for all tiers
- **Burst Handling**: Queue-based load balancing
- **Failure Recovery**: Automatic retry with exponential backoff

---

## Quick Start Guide

### Prerequisites
- Docker installed on your system
- Git for cloning the repository

### 1. Build the Docker Image

```bash
# Clone the repository
git clone <repository-url>
cd doubleword-technical

# Build the lightweight development image
docker build -f Dockerfile.dev -t doubleword-technical_app:latest .
```

### 2. Run the Container with Qwen Model

```bash
docker stop ray_vllm_container
docker rm ray_vllm_container
# Run the container with mapped ports
docker run -p 8000:8000 -e MODEL=Qwen/Qwen2.5-0.5B -e ENGINE=vllm --name ray_vllm_container -d ray_vllm_dev python -m uvicorn app.api.routes:app --host 0.0.0.0 --port 8000

```

### 3. Verify the Service is Running

```bash
# Check container status
docker ps | grep ray_vllm_container

# Check API health
curl -s http://localhost:8000/health | jq .
```

Expected health response:
```json
{
  "status": "healthy",
  "service": "ray-data-vllm-authenticated",
  "processor_initialized": false,
  "config_loaded": true,
  "version": "2.0.0"
}
```

### 4. Send a Batch Job

```bash
# Send batch inference request
curl -X POST "http://localhost:8000/generate_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "What is the capital of France?",
      "Explain the concept of machine learning.",
      "Write a short poem about technology.",
      "What are the benefits of renewable energy?",
      "Describe the process of photosynthesis."
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }' | jq .
```

Expected response:
```json
{
  "results": [
    {
      "text": "Processed: What is the capital of France?...",
      "tokens_generated": 6,
      "inference_time": 3.2
    }
  ],
  "total_time": 9.6,
  "total_prompts": 5,
  "throughput": 0.52
}
```

### 5. Monitor the System

#### View Container Logs
```bash
# View real-time logs
docker logs -f doubleword-container

# View last 50 lines
docker logs --tail 50 doubleword-container
```

#### Check System Metrics
```bash
# Get metrics endpoint
curl -s http://localhost:8000/metrics | jq .

# Check Ray Dashboard (if accessible)
curl -s http://localhost:8265 | head -20
```

#### Monitor Resource Usage
```bash
# Check container resource usage
docker stats doubleword-container

# Check system resources
top -p $(docker inspect -f '{{.State.Pid}}' doubleword-container)
```

### 6. View Results

#### Check Batch Job Status
```bash
# Submit an async job (alternative to direct batch)
curl -X POST "http://localhost:8000/start" \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/tmp/input.json",
    "output_path": "/tmp/output.json",
    "num_samples": 100,
    "batch_size": 32,
    "concurrency": 2
  }' | jq .

# Monitor job status (replace JOB_ID with actual ID)
curl -s "http://localhost:8000/jobs/JOB_ID" | jq .
```

#### View Output Files
```bash
# Access container shell
docker exec -it doubleword-container bash

# View output directory
ls -la /tmp/

# View specific output file
cat /tmp/output.json | jq .
```

### 7. Cleanup

```bash
# Stop and remove container
docker stop doubleword-container
docker rm doubleword-container

# Remove Docker image (optional)
docker rmi doubleword-technical_app:latest
```

---

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check Docker daemon
docker version

# Check for port conflicts
netstat -tulpn | grep :8000
```

#### API Not Responding
```bash
# Check if container is running
docker ps

# Check container logs for errors
docker logs doubleword-container

# Restart container
docker restart doubleword-container
```

#### Memory Issues
```bash
# Check available memory
free -h

# Run with increased memory limits
docker run -d --name doubleword-container \
  --memory=8g \
  -p 8000:8000 \
  -p 8265:8265 \
  doubleword-technical_app:latest
```

#### Performance Monitoring
```bash
# Monitor Ray performance
docker exec doubleword-container ray status

# Check GPU usage (if using GPU)
docker exec doubleword-container nvidia-smi

# Monitor CPU and memory
docker stats --no-stream doubleword-container
```

### Debug Mode

```bash
# Run with additional debugging
docker run -d --name doubleword-container \
  -e RAY_BACKEND_LOG_LEVEL=debug \
  -p 8000:8000 \
  -p 8265:8265 \
  doubleword-technical_app:latest

# Access shell for debugging
docker exec -it doubleword-container bash
```

---

## OpenAI-Compatible Batch Interface

The server now provides an OpenAI-compatible batch processing interface with additional features for production use.

### 1. Create a Batch Job

#### Using JSON Input

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "input": [
    {"prompt": "What is AI?"},
    {"prompt": "Explain ML basics"},
    {"prompt": "How do neural networks work?"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

#### Using Python Client

```python
import openai_batch_client as batch

# Create batch job
response = batch.create_batch(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    input=[
        {"prompt": "What is AI?"},
        {"prompt": "Explain ML basics"},
        {"prompt": "How do neural networks work?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(f"Job ID: {response['id']}")
print(f"Status: {response['status']}")
```

#### Using Direct API Call

```bash
curl -X POST "http://localhost:8000/v1/batches" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "input": [
      {"prompt": "What is AI?"},
      {"prompt": "Explain ML basics"},
      {"prompt": "How do neural networks work?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq .
```

### 2. Check Job Status

#### Using Python Client

```python
# Check job status
job_id = response["id"]
status = batch.retrieve_batch(job_id)
print(f"Status: {status['status']}")
print(f"Created: {status['created_at']}")
print(f"Completed: {status.get('completed_at')}")
```

#### Using Direct API Call

```bash
# Get job status
curl -s "http://localhost:8000/v1/batches/JOB_ID" | jq .

# Poll for completion
while true; do
  status=$(curl -s "http://localhost:8000/v1/batches/JOB_ID" | jq -r '.status')
  echo "Status: $status"
  
  if [[ "$status" == "completed" || "$status" == "failed" ]]; then
    break
  fi
  
  sleep 5
done
```

### 3. Retrieve Results

#### Using Python Client

```python
# Wait for completion and get results
final_status = batch.BatchClient().wait_for_completion(job_id)

if final_status["status"] == "completed":
    results = batch.get_batch_results(job_id)
    
    for item in results:
        print(f"Input: {item['input']['prompt']}")
        print(f"Output: {item['output_text']}")
        print(f"Tokens: {item['tokens_generated']}")
        print("---")
```

#### Using Direct API Call

```bash
# Get results
curl -s "http://localhost:8000/v1/batches/JOB_ID/results" | jq '.data[]'

# Format results nicely
curl -s "http://localhost:8000/v1/batches/JOB_ID/results" | \
  jq -r '.data[] | "Input: \(.input.prompt)\nOutput: \(.output_text)\nTokens: \(.tokens_generated)\n---"'
```

### 4. Optional Features

#### Batch Size Validation

```python
# Validate input before submission
validation = batch.BatchClient().validate_input({
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "input": [
        {"prompt": "What is AI?"},
        {"prompt": "Explain ML basics"}
    ],
    "max_tokens": 100
})

if validation['is_valid']:
    print("Input is valid")
else:
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
```

#### Cost Estimation

```python
# Estimate cost before submission
cost_estimate = batch.BatchClient().estimate_cost(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    input=[
        {"prompt": "What is AI?"},
        {"prompt": "Explain ML basics"},
        {"prompt": "How do neural networks work?"}
    ],
    max_tokens=100
)

print(f"Estimated cost: ${cost_estimate['estimated_cost_usd']:.6f}")
print(f"Input tokens: {cost_estimate['input_tokens']}")
print(f"Max output tokens: {cost_estimate['max_output_tokens']}")
```

#### Retry & Failure Handling

The system automatically implements:
- **Max 3 retry attempts** per failed batch
- **Exponential backoff** (1s, 2s, 4s delays)
- **Partial success handling** - failed batches don't stop processing
- **Error tracking** - detailed error information in output

#### Input Validation Features

```bash
# Validate JSON structure
curl -X POST "http://localhost:8000/v1/batches/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "json_data": {
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "input": [
        {"prompt": "What is AI?"},
        {"prompt": "Explain ML basics"}
      ]
    },
    "max_batch_size": 1000
  }' | jq .
```

#### Result Mapping with Indices

For tracking input-output mapping:

```python
# Use indexed input format
from app.models.schemas import BatchInputWithIndex

indexed_input = [
    BatchInputWithIndex(index=0, prompt="What is AI?"),
    BatchInputWithIndex(index=1, prompt="Explain ML basics"),
    BatchInputWithIndex(index=2, prompt="How do neural networks work?")
]
```

### 5. Production Considerations

#### Batch Size Limits
- **Maximum batch size**: 1000 items per request
- **Automatic truncation**: Excess items are automatically truncated
- **Recommended size**: 100-500 items for optimal performance

#### Streaming Output
- **Batch jobs**: No streaming support (asynchronous processing)
- **Real-time API**: Use `/generate_batch` endpoint for streaming

#### Token Cost Management
- **Pre-calculation**: Use cost estimation endpoint
- **Budget tracking**: Monitor cumulative costs
- **Token limits**: Respect max_tokens per request

#### Error Handling Best Practices

```python
# Robust batch processing
try:
    # Create batch
    response = batch.create_batch(model, input_data, max_tokens, temperature)
    job_id = response["id"]
    
    # Wait with timeout
    final_status = batch.BatchClient().wait_for_completion(
        job_id, 
        poll_interval=5.0, 
        timeout=3600.0  # 1 hour
    )
    
    if final_status["status"] == "completed":
        results = batch.get_batch_results(job_id)
        # Process results with error handling
        for item in results:
            try:
                # Process successful result
                print(item["output_text"])
            except Exception as e:
                print(f"Error processing result: {e}")
    
    elif final_status["status"] == "failed":
        print(f"Batch failed: {final_status}")
        
except TimeoutError:
    print("Batch processing timed out")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 6. Advanced Usage Examples

#### Batch Processing with Retry Configuration

```python
from app.models.schemas import BatchRetryConfig, OpenAIBatchCreateRequestWithIndex

retry_config = BatchRetryConfig(
    max_retries=5,
    retry_delay=2.0,
    backoff_factor=1.5
)

request = OpenAIBatchCreateRequestWithIndex(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    input=indexed_input,
    max_tokens=256,
    temperature=0.7,
    retry_config=retry_config,
    validate_input=True
)
```

#### Monitoring Multiple Jobs

```python
import concurrent.futures
import time

def monitor_job(job_id):
    client = batch.BatchClient()
    return client.wait_for_completion(job_id, poll_interval=2.0)

# Monitor multiple jobs concurrently
job_ids = ["job1", "job2", "job3"]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(monitor_job, jid): jid for jid in job_ids}
    
    for future in concurrent.futures.as_completed(futures, timeout=3600):
        job_id = futures[future]
        try:
            result = future.result()
            print(f"Job {job_id} completed: {result['status']}")
        except Exception as e:
            print(f"Job {job_id} failed: {e}")
```

---

## Testing

This section provides comprehensive testing instructions for validating the OpenAI-compatible batch interface functionality.

### Prerequisites for Testing

```bash
# Ensure container is running
docker ps | grep doubleword-container

# Install required Python packages for testing
pip install requests jq

# Set environment variables
export BATCH_API_URL="http://localhost:8000"
export TEST_OUTPUT_DIR="/tmp/test_results"
mkdir -p $TEST_OUTPUT_DIR
```

### 1. Test Basic API Connectivity

```bash
# Test health endpoint
curl -s "$BATCH_API_URL/health" | jq .

# Test OpenAI-compatible endpoint exists
curl -s "$BATCH_API_URL/v1/batches" | head -20

# Expected: Should return method not allowed (405) confirming endpoint exists
```

### 2. Test Batch Job Creation

```bash
# Create test batch job
BATCH_RESPONSE=$(curl -s -X POST "$BATCH_API_URL/v1/batches" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "input": [
      {"prompt": "What is the capital of France?"},
      {"prompt": "Explain machine learning in one sentence"},
      {"prompt": "Write a haiku about technology"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }')

echo "Batch Creation Response:"
echo "$BATCH_RESPONSE" | jq .

# Extract job ID
JOB_ID=$(echo "$BATCH_RESPONSE" | jq -r '.id')
echo "Job ID: $JOB_ID"
```

### 3. Test Job Status Retrieval

```bash
# Wait a moment then check status
sleep 5

# Retrieve job status
STATUS_RESPONSE=$(curl -s "$BATCH_API_URL/v1/batches/$JOB_ID")
echo "Job Status:"
echo "$STATUS_RESPONSE" | jq .

# Check status field
STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
echo "Current Status: $STATUS"
```

### 4. Test Job Completion Polling

```bash
# Poll for completion with timeout
POLL_TIMEOUT=300  # 5 minutes
POLL_INTERVAL=10
ELAPSED=0

echo "Polling for job completion..."
while [ $ELAPSED -lt $POLL_TIMEOUT ]; do
    STATUS_RESPONSE=$(curl -s "$BATCH_API_URL/v1/batches/$JOB_ID")
    STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
    
    echo "[$(date +%H:%M:%S)] Status: $STATUS"
    
    if [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]]; then
        echo "Job finished with status: $STATUS"
        break
    fi
    
    sleep $POLL_INTERVAL
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [ $ELAPSED -ge $POLL_TIMEOUT ]; then
    echo "Polling timed out after $POLL_TIMEOUT seconds"
    exit 1
fi
```

### 5. Test Results Retrieval

```bash
# Get results if job completed
if [[ "$STATUS" == "completed" ]]; then
    RESULTS_RESPONSE=$(curl -s "$BATCH_API_URL/v1/batches/$JOB_ID/results")
    
    echo "Results Response:"
    echo "$RESULTS_RESPONSE" | jq .
    
    # Save results to file
    echo "$RESULTS_RESPONSE" | jq . > "$TEST_OUTPUT_DIR/test_results_$JOB_ID.json"
    echo "Results saved to: $TEST_OUTPUT_DIR/test_results_$JOB_ID.json"
    
    # Extract and display individual results
    echo ""
    echo "Individual Results:"
    echo "$RESULTS_RESPONSE" | jq -r '.data[] | "Input: \(.input.prompt)\nOutput: \(.output_text)\nTokens: \(.tokens_generated)\n---"'
else
    echo "Job did not complete successfully. Status: $STATUS"
fi
```

### 6. Test Input Validation

```bash
# Test valid input
VALID_RESPONSE=$(curl -s -X POST "$BATCH_API_URL/v1/batches/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "json_data": {
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "input": [
        {"prompt": "Test prompt 1"},
        {"prompt": "Test prompt 2"}
      ],
      "max_tokens": 50
    },
    "max_batch_size": 1000
  }')

echo "Valid Input Test:"
echo "$VALID_RESPONSE" | jq .

# Test invalid input
INVALID_RESPONSE=$(curl -s -X POST "$BATCH_API_URL/v1/batches/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "json_data": {
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "input": [
        {"wrong_field": "This should fail"},
        {"prompt": "This one is ok"}
      ]
    },
    "max_batch_size": 1000
  }')

echo "Invalid Input Test:"
echo "$INVALID_RESPONSE" | jq .
```

### 7. Test Cost Estimation

```bash
# Test cost estimation
COST_RESPONSE=$(curl -s -X POST "$BATCH_API_URL/v1/batches/cost-estimate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "input": [
      {"prompt": "What is AI?"},
      {"prompt": "Explain machine learning"},
      {"prompt": "How do neural networks work?"},
      {"prompt": "Write a short poem"},
      {"prompt": "What is the meaning of life?"}
    ],
    "max_tokens": 100
  }')

echo "Cost Estimation:"
echo "$COST_RESPONSE" | jq .

# Extract cost information
ESTIMATED_COST=$(echo "$COST_RESPONSE" | jq -r '.estimated_cost_usd')
INPUT_TOKENS=$(echo "$COST_RESPONSE" | jq -r '.input_tokens')
OUTPUT_TOKENS=$(echo "$COST_RESPONSE" | jq -r '.max_output_tokens')

echo "Estimated Cost: $$ESTIMATED_COST"
echo "Input Tokens: $INPUT_TOKENS"
echo "Max Output Tokens: $OUTPUT_TOKENS"
```

### 8. Test Batch Size Limits

```bash
# Test batch size within limits
NORMAL_BATCH=$(curl -s -X POST "$BATCH_API_URL/v1/batches/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "json_data": {
      "input": []
    },
    "max_batch_size": 1000
  }')

# Add 1000 items to input
ITEMS=""
for i in {1..1000}; do
    ITEMS+=$(printf '{"prompt": "Test item %d"},' $i)
done
ITEMS=${ITEMS%?}  # Remove trailing comma

LARGE_BATCH=$(curl -s -X POST "$BATCH_API_URL/v1/batches/validate" \
  -H "Content-Type: application/json" \
  -d "{
    \"json_data\": {
      \"input\": [$ITEMS]
    },
    \"max_batch_size\": 1000
  }")

echo "Normal Batch Validation:"
echo "$NORMAL_BATCH" | jq '.is_valid, .estimated_items'

echo "Large Batch Validation:"
echo "$LARGE_BATCH" | jq '.is_valid, .estimated_items, .warnings'
```

### 9. Test Python Client Library

```bash
# Test the Python client
cat > /tmp/test_client.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.append('/app')

# Import the client
try:
    import openai_batch_client as batch
    
    print("✅ Client import successful")
    
    # Test batch creation
    print("\n📝 Testing batch creation...")
    response = batch.create_batch(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        input=[
            {"prompt": "What is the capital of France?"},
            {"prompt": "Explain Python in one sentence"}
        ],
        max_tokens=50,
        temperature=0.7
    )
    
    job_id = response["id"]
    print(f"✅ Job created: {job_id}")
    
    # Test status retrieval
    print("\n📊 Testing status retrieval...")
    status = batch.retrieve_batch(job_id)
    print(f"✅ Status retrieved: {status['status']}")
    
    # Test validation
    print("\n✅ Testing input validation...")
    validation = batch.BatchClient().validate_input({
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "input": [{"prompt": "Test validation"}],
        "max_tokens": 50
    })
    print(f"✅ Validation result: {validation['is_valid']}")
    
    # Test cost estimation
    print("\n💰 Testing cost estimation...")
    cost = batch.BatchClient().estimate_cost(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        input=[{"prompt": "Test cost estimation"}],
        max_tokens=50
    )
    print(f"✅ Cost estimate: ${cost['estimated_cost_usd']:.6f}")
    
    print("\n🎉 All client tests passed!")
    
except ImportError as e:
    print(f"❌ Client import failed: {e}")
except Exception as e:
    print(f"❌ Client test failed: {e}")
EOF

# Run the test inside container
docker exec doubleword-container python /tmp/test_client.py
```

### 10. Test Error Handling

```bash
# Test with invalid job ID
echo "Testing invalid job ID..."
curl -s "$BATCH_API_URL/v1/batches/invalid_job_id" | jq .

# Test with malformed JSON
echo "Testing malformed JSON..."
curl -s -X POST "$BATCH_API_URL/v1/batches" \
  -H "Content-Type: application/json" \
  -d '{"invalid": json}' | jq .

# Test with missing required fields
echo "Testing missing required fields..."
curl -s -X POST "$BATCH_API_URL/v1/batches" \
  -H "Content-Type: application/json" \
  -d '{"model": "test"}' | jq .
```

### 11. Performance Testing

```bash
# Test concurrent batch submissions
echo "Testing concurrent submissions..."

CONCURRENT_JOBS=5
PIDS=()

for i in $(seq 1 $CONCURRENT_JOBS); do
    (
        BATCH_RESPONSE=$(curl -s -X POST "$BATCH_API_URL/v1/batches" \
          -H "Content-Type: application/json" \
          -d "{
            \"model\": \"Qwen/Qwen2.5-0.5B-Instruct\",
            \"input\": [
              {\"prompt\": \"Concurrent test $i - prompt 1\"},
              {\"prompt\": \"Concurrent test $i - prompt 2\"}
            ],
            \"max_tokens\": 30
          }")
        
        JOB_ID=$(echo "$BATCH_RESPONSE" | jq -r '.id')
        echo "Started concurrent job $i: $JOB_ID"
    ) &
    PIDS+=($!)
done

# Wait for all background jobs
wait "${PIDS[@]}"

echo "All concurrent jobs submitted"
```

### 12. Integration Test Script

```bash
# Complete integration test
cat > /tmp/integration_test.sh << 'EOF'
#!/bin/bash

set -e

API_URL="${BATCH_API_URL:-http://localhost:8000}"
TEST_DIR="${TEST_OUTPUT_DIR:-/tmp/integration_test}"
mkdir -p "$TEST_DIR"

echo "🧪 Starting Integration Test"
echo "API URL: $API_URL"
echo "Test Directory: $TEST_DIR"

# Step 1: Health Check
echo "1️⃣ Testing health endpoint..."
HEALTH=$(curl -s "$API_URL/health")
if [[ $(echo "$HEALTH" | jq -r '.status') != "healthy" ]]; then
    echo "❌ Health check failed"
    exit 1
fi
echo "✅ Health check passed"

# Step 2: Create Batch
echo "2️⃣ Creating batch job..."
BATCH_CREATE=$(curl -s -X POST "$API_URL/v1/batches" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "input": [
      {"prompt": "Integration test prompt 1"},
      {"prompt": "Integration test prompt 2"}
    ],
    "max_tokens": 50
  }')

JOB_ID=$(echo "$BATCH_CREATE" | jq -r '.id')
echo "✅ Batch created: $JOB_ID"

# Step 3: Wait for Completion
echo "3️⃣ Waiting for completion..."
TIMEOUT=180
ELAPSED=0

while [ $ELAPSED -lt $TIMEOUT ]; do
    STATUS=$(curl -s "$API_URL/v1/batches/$JOB_ID" | jq -r '.status')
    echo "   Status: $STATUS"
    
    if [[ "$STATUS" == "completed" ]]; then
        echo "✅ Job completed successfully"
        break
    elif [[ "$STATUS" == "failed" ]]; then
        echo "❌ Job failed"
        exit 1
    fi
    
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "❌ Job timed out"
    exit 1
fi

# Step 4: Retrieve Results
echo "4️⃣ Retrieving results..."
RESULTS=$(curl -s "$API_URL/v1/batches/$JOB_ID/results")
echo "$RESULTS" | jq . > "$TEST_DIR/integration_results.json"

RESULT_COUNT=$(echo "$RESULTS" | jq '.data | length')
echo "✅ Retrieved $RESULT_COUNT results"

# Step 5: Validate Results
echo "5️⃣ Validating results..."
for i in $(seq 0 $((RESULT_COUNT - 1))); do
    PROMPT=$(echo "$RESULTS" | jq -r ".data[$i].input.prompt")
    OUTPUT=$(echo "$RESULTS" | jq -r ".data[$i].output_text")
    TOKENS=$(echo "$RESULTS" | jq -r ".data[$i].tokens_generated")
    
    if [[ -n "$PROMPT" && -n "$OUTPUT" && "$TOKENS" =~ ^[0-9]+$ ]]; then
        echo "   ✅ Result $i: Valid"
    else
        echo "   ❌ Result $i: Invalid"
    fi
done

echo ""
echo "🎉 Integration test completed successfully!"
echo "Results saved to: $TEST_DIR/integration_results.json"
EOF

# Make executable and run
chmod +x /tmp/integration_test.sh
/tmp/integration_test.sh
```

### 13. Test Results Summary

```bash
# Create test summary
cat > /tmp/test_summary.md << 'EOF'
# Test Results Summary

## Tests Performed
- [x] API Connectivity
- [x] Health Check
- [x] Batch Job Creation
- [x] Job Status Retrieval
- [x] Results Retrieval
- [x] Input Validation
- [x] Cost Estimation
- [x] Python Client Library
- [x] Error Handling
- [x] Performance Testing
- [x] Integration Testing

## Expected Results
- All API endpoints should respond with proper HTTP status codes
- Batch jobs should process within reasonable time
- Results should maintain input-output mapping
- Validation should catch malformed inputs
- Cost estimation should provide reasonable estimates
- Error handling should be graceful

## Troubleshooting
If tests fail:
1. Check container logs: `docker logs doubleword-container`
2. Verify API endpoints: `curl -s http://localhost:8000/health`
3. Check resource usage: `docker stats doubleword-container`
4. Restart container: `docker restart doubleword-container`
EOF

echo "📋 Test summary created: /tmp/test_summary.md"
cat /tmp/test_summary.md
```

### Running All Tests

```bash
# Execute complete test suite
echo "🚀 Running complete test suite..."

# Run individual test components
bash /tmp/integration_test.sh
docker exec doubleword-container python /tmp/test_client.py

# Performance test
echo "Running performance test..."
bash -c 'source /tmp/performance_test.sh'

echo ""
echo "✅ All tests completed!"
echo "📊 Check test outputs in: $TEST_OUTPUT_DIR"
ls -la "$TEST_OUTPUT_DIR"
```

### Test Cleanup

```bash
# Clean up test artifacts
echo "🧹 Cleaning up test artifacts..."
rm -rf /tmp/test_*
rm -f /tmp/test_client.py
rm -f /tmp/integration_test.sh
rm -f /tmp/test_summary.md

echo "✅ Cleanup completed"
```

This comprehensive testing suite validates all aspects of the OpenAI-compatible batch interface including:
- Basic API functionality
- Batch job lifecycle management
- Input validation and error handling
- Performance under load
- Integration testing
- Client library functionality

