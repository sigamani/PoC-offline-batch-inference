#!/usr/bin/env python3
"""
FastAPI server for LLM batch inference with monitoring
Accepts requests from remote clients
"""
import os
import time
import torch
import asyncio
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from prometheus_client import start_http_server, Gauge, Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST, generate_latest, CONTENT_TYPE_LATEST
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import queue

# Initialize FastAPI
app = FastAPI(title="LLM Inference API", version="1.0.0")

# Add CORS middleware for remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your MacBook's IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
inference_requests = Counter(
    'llm_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

inference_duration = Histogram(
    'llm_api_inference_duration_seconds',
    'API inference duration',
    ['worker_id', 'model_name']
)

active_connections = Gauge(
    'llm_api_active_connections',
    'Active API connections'
)

queue_size = Gauge(
    'llm_api_queue_size',
    'Current request queue size'
)

kv_cache_utilization = Gauge(
    'llm_kv_cache_utilization_percent',
    'KV cache utilization percentage'
)

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    worker_id: Optional[str] = None

class BatchInferenceRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 100
    temperature: float = 0.7
    worker_id: Optional[str] = None

class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int
    inference_time: float
    worker_id: str

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse]
    total_time: float
    worker_id: str

class LLMInferenceWorker:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.worker_id = os.environ.get('WORKER_ID', 'gpu-worker-1')
        self.model_name = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B')
        self.request_queue = queue.Queue()
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading {self.model_name} on {self.worker_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Model loaded on device: {next(self.model.parameters()).device}")
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict:
        """Generate text with timing"""
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_response[len(prompt):].strip()
        
        inference_time = time.time() - start_time
        tokens_generated = len(self.tokenizer.encode(generated_text))
        
        return {
            'text': generated_text,
            'tokens_generated': tokens_generated,
            'inference_time': inference_time
        }

# Global worker instance
worker = LLMInferenceWorker()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "worker_id": worker.worker_id,
        "model": worker.model_name,
        "device": str(next(worker.model.parameters()).device)
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    gpu_memory = torch.cuda.memory_allocated() / 1024**3
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        "status": "healthy",
        "worker_id": worker.worker_id,
        "model": worker.model_name,
        "device": str(next(worker.model.parameters()).device),
        "gpu_memory_used_gb": round(gpu_memory, 2),
        "gpu_memory_total_gb": round(gpu_total, 2),
        "gpu_utilization": round((gpu_memory / gpu_total) * 100, 1),
        "queue_size": worker.request_queue.qsize()
    }

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Single text generation endpoint"""
    active_connections.inc()
    queue_size.set(worker.request_queue.qsize())
    
    try:
        start_time = time.time()
        
        # Generate text
        result = worker.generate_text(
            request.prompt,
            request.max_tokens,
            request.temperature
        )
        
        total_time = time.time() - start_time
        
        # Record metrics
        inference_requests.labels(endpoint="/generate", method="POST", status="success").inc()
        inference_duration.labels(
            worker_id=worker.worker_id,
            model_name=worker.model_name
        ).observe(result['inference_time'])
        
        response = InferenceResponse(
            text=result['text'],
            tokens_generated=result['tokens_generated'],
            inference_time=result['inference_time'],
            worker_id=worker.worker_id
        )
        
        return response
        
    except Exception as e:
        inference_requests.labels(endpoint="/generate", method="POST", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_connections.dec()
        queue_size.set(worker.request_queue.qsize())

@app.post("/generate_batch", response_model=BatchInferenceResponse)
async def generate_batch(request: BatchInferenceRequest):
    """Batch text generation endpoint"""
    active_connections.inc()
    queue_size.set(worker.request_queue.qsize())
    
    try:
        start_time = time.time()
        results = []
        
        for i, prompt in enumerate(request.prompts):
            # Generate text for each prompt
            result = worker.generate_text(
                prompt,
                request.max_tokens,
                request.temperature
            )
            
            results.append(InferenceResponse(
                text=result['text'],
                tokens_generated=result['tokens_generated'],
                inference_time=result['inference_time'],
                worker_id=worker.worker_id
            ))
        
        total_time = time.time() - start_time
        
        # Record metrics
        inference_requests.labels(endpoint="/generate_batch", method="POST", status="success").inc()
        for result in results:
            inference_duration.labels(
                worker_id=worker.worker_id,
                model_name=worker.model_name
            ).observe(result.inference_time)
        
        return BatchInferenceResponse(
            results=results,
            total_time=total_time,
            worker_id=worker.worker_id
        )
        
    except Exception as e:
        inference_requests.labels(endpoint="/generate_batch", method="POST", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_connections.dec()
        queue_size.set(worker.request_queue.qsize())

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics server"""
    start_http_server(port)
    print(f"Metrics server started on port {port}")

if __name__ == "__main__":
    import socket
    
    # Get host IP
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    
    # Start metrics server
    start_metrics_server()
    
    print(f"\n🚀 LLM Inference API Server")
    print(f"📍 Server IP: {host_ip}")
    print(f"🔗 Local: http://localhost:8000")
    print(f"🌐 Remote: http://{host_ip}:8000")
    print(f"📊 Metrics: http://{host_ip}:8001/metrics")
    print(f"👥 Worker: {worker.worker_id}")
    print(f"🤖 Model: {worker.model_name}")
    print(f"\n📝 API Endpoints:")
    print(f"  GET  /health - Health check")
    print(f"  POST /generate - Single inference")
    print(f"  POST /generate_batch - Batch inference")
    print(f"\n🔍 Monitoring:")
    print(f"  Grafana: http://localhost:3000")
    print(f"  Prometheus: http://localhost:9090")
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )