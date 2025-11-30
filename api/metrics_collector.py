"""Metrics collector service for monitoring batch inference performance."""

import sys
import os
import time
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Metrics Collector",
    version="1.0.0",
    description="Metrics collection service for batch inference"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "healthy", "service": "metrics-collector", "version": "1.0.0"}

@app.get("/metrics")
async def get_metrics():
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        process = psutil.Process()
        process_memory = process.memory_info()
        
        llm_metrics = {}
        try:
            vllm_response = requests.get("http://vllm:8001/metrics", timeout=2)
            if vllm_response.status_code == 200:
                llm_metrics["vllm_available"] = True
                metrics_text = vllm_response.text
                kv_cache_usage = 0
                gpu_utilization = 0
                request_queue_size = 0
                
                for line in metrics_text.split('\n'):
                    if 'vllm:gpu_cache_usage' in line:
                        kv_cache_usage = float(line.split()[-1])
                    elif 'vllm:gpu_utilization' in line:
                        gpu_utilization = float(line.split()[-1])
                    elif 'vllm:waiting_requests' in line:
                        request_queue_size = int(line.split()[-1])
                
                llm_metrics.update({
                    "kv_cache_usage_percent": kv_cache_usage * 100,
                    "gpu_utilization_percent": gpu_utilization * 100,
                    "vllm_queue_size": request_queue_size
                })
            else:
                llm_metrics["vllm_available"] = False
                
        except Exception as e:
            llm_metrics = {"vllm_available": False, "error": str(e)}
        
        queue_metrics = {}
        try:
            import requests
            queue_response = requests.get("http://api:8000/queue/stats", timeout=2)
            if queue_response.status_code == 200:
                queue_metrics = queue_response.json()
        except Exception as e:
            queue_metrics = {"error": str(e)}
        
        gpu_pool_metrics = {}
        try:
            import requests
            pool_response = requests.get("http://api:8000/debug/gpu-pools", timeout=2)
            if pool_response.status_code == 200:
                gpu_pool_metrics = pool_response.json()
        except Exception as e:
            gpu_pool_metrics = {"error": str(e)}
        
        metrics = {
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                }
            },
            "process": {
                "pid": process.pid,
                "memory_rss": process_memory.rss,
                "memory_vms": process_memory.vms,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time()
            },
            "llm_inference": llm_metrics,
            "queue": queue_metrics,
            "gpu_pools": gpu_pool_metrics
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect metrics")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}