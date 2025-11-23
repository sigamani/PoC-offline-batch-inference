#!/usr/bin/env python3
"""
Ray Data + vLLM distributed batch inference orchestration
Using Ray Data's built-in vLLM integration following official documentation:
https://docs.ray.io/en/latest/data/batch_inference.html
"""
import os
import time
import sys
import torch
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ray
from ray import data
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
import numpy as np
from prometheus_client import start_http_server, Gauge, Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
import threading

# Prometheus metrics
inference_requests = Counter(
    'ray_data_requests_total',
    'Total Ray Data requests',
    ['method', 'status']
)

inference_duration = Histogram(
    'ray_data_inference_duration_seconds',
    'Ray Data inference duration',
    ['node_id', 'model_name']
)

active_batches = Gauge(
    'ray_data_active_batches',
    'Active Ray Data batches'
)

gpu_utilization = Gauge(
    'ray_data_gpu_utilization_percent',
    'GPU utilization per node',
    ['node_id', 'gpu_id']
)

# Pydantic models
class BatchInferenceRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 100
    temperature: float = 0.7
    batch_size: int = 64  # Ray Data batch size for vLLM

class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int
    inference_time: float
    node_id: str

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse]
    total_time: float
    total_prompts: int
    nodes_used: int
    throughput: float  # prompts per second

# FastAPI app
app = FastAPI(title="Ray Data vLLM Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor and actors
llm_processor = None
vllm_actors = []

def initialize_ray_cluster():
    """Initialize Ray cluster and create vLLM processor"""
    global llm_processor, vllm_actors
    
    # Initialize Ray
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # Worker mode
        head_address = sys.argv[2] if len(sys.argv) > 2 else "localhost:6379"
        ray.init(address=head_address, _redis_password="ray123")
        print(f"✅ Worker connected to {head_address}")
    else:
        # Head mode
        ray.init(
            address="local",
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            _redis_password="ray123"
        )
        print("🚀 Ray head node started")
    
    # Configure vLLM processor using Ray Data's built-in integration
    model_name = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')
    
    config = vLLMEngineProcessorConfig(
        model=model_name,
        engine_kwargs={
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "max_model_len": 16384,
        },
        concurrency=1,  # Number of GPUs per actor
        batch_size=64,  # Optimize for vLLM throughput
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": row["item"]}
            ]
        ),
        sampling_params=dict(
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        ),
        postprocess=lambda row: dict(
            answer=row["generated_text"]
        )
    )
    
    # Build the LLM processor
    llm_processor = build_llm_processor(config)
    print(f"✅ vLLM processor configured for model: {model_name}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ray-data-vllm",
        "ray_version": ray.__version__,
        "ray_nodes": len(ray.nodes()),
        "vllm_processor": llm_processor is not None
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        nodes = ray.nodes()
        gpu_nodes = [node for node in nodes if node.get('Resources', {}).get('GPU', 0) > 0]
        
        return {
            "status": "healthy",
            "service": "ray-data-vllm",
            "ray_nodes": len(nodes),
            "gpu_nodes": len(gpu_nodes),
            "vllm_processor": llm_processor is not None,
            "model": os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/generate_batch", response_model=BatchInferenceResponse)
async def generate_batch(request: BatchInferenceRequest):
    """Distributed batch inference using Ray Data vLLM processor"""
    if not llm_processor:
        raise HTTPException(status_code=503, detail="vLLM processor not initialized")
    
    active_batches.inc()
    
    try:
        start_time = time.time()
        
        # Create Ray Dataset from prompts
        ds = data.from_items([{"item": prompt} for prompt in request.prompts])
        
        # Use Ray Data's built-in vLLM processor for distributed processing
        # This automatically handles GPU allocation and batch optimization
        processed_ds = llm_processor(ds)
        
        # Collect results
        results = []
        nodes_used = set()
        
        # Iterate through processed batches
        for batch in processed_ds.iter_batches():
            for row in batch:
                results.append(InferenceResponse(
                    text=row["answer"],
                    tokens_generated=len(row["answer"].split()),  # Approximate token count
                    inference_time=0,  # Will be calculated below
                    node_id="ray-cluster"  # Ray Data handles node allocation
                ))
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        throughput = len(request.prompts) / total_time if total_time > 0 else 0
        
        # Record metrics
        inference_requests.labels(
            method="generate_batch",
            status="success"
        ).inc(len(request.prompts))
        
        inference_duration.labels(
            node_id="ray-cluster",
            model_name=os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')
        ).observe(total_time)
        
        return BatchInferenceResponse(
            results=results,
            total_time=total_time,
            total_prompts=len(request.prompts),
            nodes_used=len(nodes_used),
            throughput=throughput
        )
        
    except Exception as e:
        inference_requests.labels(
            method="generate_batch",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_batches.dec()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    # Initialize Ray cluster and vLLM processor
    initialize_ray_cluster()
    
    # Start FastAPI server
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )