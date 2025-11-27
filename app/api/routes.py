import logging
import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, Info
except ImportError:
    generate_latest = None
    CONTENT_TYPE_LATEST = None
    Counter = None
    Histogram = None
    Gauge = None
    Info = None

from app.core.context import get_context, set_context
from app.core.config import get_config
from app.core.simple_processor import SimpleInferencePipeline, VLLMProcessor
from app.core.job_manager import job_manager, JobStatus
from app.models.schemas import (
    BatchInferenceRequest, BatchInferenceResponse,
    AuthBatchJobRequest, BatchJobResponse,
    InferenceResponse,
    # OpenAI-compatible schemas
    OpenAIBatchCreateRequest, OpenAIBatchCreateResponse,
    OpenAIBatchRetrieveResponse, OpenAIBatchResultsResponse,
    BatchInputItem, BatchResultItem, BatchCostEstimate,
    BatchValidationRequest, BatchValidationResponse,
    BatchInputWithIndex, OpenAIBatchCreateRequestWithIndex,
    JobStatus as SchemaJobStatus
)
from app.core.job_manager import JobStatus

logger = logging.getLogger(__name__)

# Initialize components
config = get_config()
inference_pipeline = SimpleInferencePipeline()

# Start background job manager
job_manager.start_worker()

# FastAPI app
app = FastAPI(
    title="Ray Data vLLM Batch Inference",
    version="2.0.0",
    description="Production-ready batch inference with authentication"
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ray-data-vllm-authenticated",
        "version": "2.0.0"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    context = get_context()
    return {
        "status": "healthy",
        "service": "ray-data-vllm-authenticated",
        "processor_initialized": context.processor is not None,
        "config_loaded": bool(context.config),
        "version": "2.0.0"
    }

@app.post("/generate_batch", response_model=BatchInferenceResponse)
async def generate_batch(
    request: BatchInferenceRequest
):
    """Authenticated batch inference"""
    try:
        context = get_context()
        if not context.processor:
            # Initialize processor on first request
            context.processor = VLLMProcessor("Qwen/Qwen2.5-0.5B-Instruct")
            set_context(context)
        
        # Execute inference
        results, total_time = inference_pipeline.execute_batch(
            request.prompts,
            context.processor
        )
        
        # Create response
        response_results = [
            InferenceResponse(
                text=result.get("response", ""),
                tokens_generated=result.get("tokens", 0),
                inference_time=total_time / len(request.prompts)
            )
            for result in results
        ]
        
        return BatchInferenceResponse(
            results=response_results,
            total_time=total_time,
            total_prompts=len(request.prompts),
            throughput=len(request.prompts) / total_time if total_time > 0 else 0,
        )
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start", response_model=BatchJobResponse)
async def start_batch_job(
    request: AuthBatchJobRequest
):
    """Start asynchronous batch job with SLA tracking"""
    try:
        # Submit job to background manager
        job_id = job_manager.submit_job({
            "input_path": request.input_path,
            "output_path": request.output_path,
            "num_samples": request.num_samples,
            "batch_size": request.batch_size,
            "concurrency": request.concurrency,

        })
        
        return BatchJobResponse(
            job_id=job_id,
            status="queued",
            message=f"Batch job {job_id} queued with {request.num_samples} samples",
            estimated_completion_hours=0.1  # Rough estimate
        )
        
    except Exception as e:
        logger.error(f"Batch job submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status by ID"""
    job = job_manager.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "samples_processed": job.samples_processed,
        "total_samples": job.num_samples,
        "progress": job.samples_processed / job.num_samples if job.num_samples > 0 else 0,
        "error_message": job.error_message,
        "output_path": job.output_path
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if generate_latest is None:
        return {"error": "Prometheus client not available"}
    
    # Simple metrics for now
    metrics_data = f"""# HELP ray_vllm_active_jobs Number of active batch jobs
# TYPE ray_vllm_active_jobs gauge
ray_vllm_active_jobs {len([j for j in job_manager.jobs.values() if j.status.value in ['queued', 'running']])}

# HELP ray_vllm_total_jobs Total number of jobs
# TYPE ray_vllm_total_jobs counter
ray_vllm_total_jobs {len(job_manager.jobs)}

# HELP ray_vllm_service_info Service information
# TYPE ray_vllm_service_info gauge
ray_vllm_service_info{{version="2.0.0",service="ray-data-vllm-batch-inference"}} 1
"""
    
    return Response(metrics_data, media_type="text/plain")

# OpenAI-compatible batch endpoints
@app.post("/v1/batches", response_model=OpenAIBatchCreateResponse)
async def create_openai_batch(request: OpenAIBatchCreateRequest):
    """Create OpenAI-compatible batch job"""
    try:
        # Convert to internal format
        prompts = [item.prompt for item in request.input]
        
        # Submit job
        job_id = job_manager.submit_job({
            "input_path": f"/tmp/batch_input.json",
            "output_path": f"/tmp/batch_output.json",
            "num_samples": len(prompts),
            "batch_size": 32,  # Default batch size
            "concurrency": 2
        })
        
        # Save input data
        import json
        input_data = {
            "model": request.model,
            "input": prompts,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        with open(f"/tmp/batch_{job_id}_input.json", 'w') as f:
            json.dump(input_data, f, indent=2)
        
        return OpenAIBatchCreateResponse(
            id=job_id,
            object="batch",
            created_at=int(time.time()),
            status=SchemaJobStatus.QUEUED
        )
        
    except Exception as e:
        logger.error(f"Batch creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/batches/{job_id}", response_model=OpenAIBatchRetrieveResponse)
async def retrieve_openai_batch(job_id: str):
    """Retrieve OpenAI-compatible batch job status"""
    try:
        job = job_manager.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        # Convert job status to schema status
        status_map = {
            JobStatus.QUEUED: SchemaJobStatus.QUEUED,
            JobStatus.RUNNING: SchemaJobStatus.RUNNING,
            JobStatus.COMPLETED: SchemaJobStatus.COMPLETED,
            JobStatus.FAILED: SchemaJobStatus.FAILED
        }
        
        return OpenAIBatchRetrieveResponse(
            id=job.job_id,
            object="batch",
            created_at=int(job.created_at),
            completed_at=int(job.completed_at) if job.completed_at else None,
            status=status_map.get(job.status, SchemaJobStatus.FAILED),
            results_file=f"/tmp/batch_{job_id}_output.json" if job.status == JobStatus.COMPLETED else None,
            error_file=f"/tmp/batch_{job_id}_errors.json" if job.status == JobStatus.FAILED else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/batches/{job_id}/results", response_model=OpenAIBatchResultsResponse)
async def get_openai_batch_results(job_id: str):
    """Get OpenAI-compatible batch results"""
    try:
        job = job_manager.get_job_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Batch not completed yet")
        
        # Load results from file
        import json
        with open(f"/tmp/batch_{job_id}_output.json", 'r') as f:
            output_data = json.load(f)
        
        # Convert to OpenAI format
        batch_results = []
        for i, result in enumerate(output_data.get("results", [])):
            batch_results.append(BatchResultItem(
                id=f"{job_id}_{i}",
                input=BatchInputItem(prompt=result.get("prompt", "")),
                output_text=result.get("response", ""),
                tokens_generated=result.get("tokens", 0)
            ))
        
        return OpenAIBatchResultsResponse(
            object="list",
            data=batch_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Results retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/batches/validate", response_model=BatchValidationResponse)
async def validate_batch_input(request: BatchValidationRequest):
    """Validate batch input JSON structure"""
    try:
        errors = []
        warnings = []
        
        # Check if it's a dict with required fields
        if not isinstance(request.json_data, dict):
            errors.append("Input must be a JSON object")
            return BatchValidationResponse(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                estimated_items=0
            )
        
        # Check for required fields
        if "input" not in request.json_data:
            errors.append("Missing required field: 'input'")
        
        if "input" in request.json_data:
            input_items = request.json_data["input"]
            if not isinstance(input_items, list):
                errors.append("'input' must be a list")
            else:
                # Check batch size limit
                if len(input_items) > request.max_batch_size:
                    warnings.append(f"Batch size {len(input_items)} exceeds recommended limit {request.max_batch_size}")
                
                # Validate each input item
                for i, item in enumerate(input_items):
                    if not isinstance(item, dict) or "prompt" not in item:
                        errors.append(f"Input item {i} missing 'prompt' field")
                    elif not isinstance(item["prompt"], str):
                        errors.append(f"Input item {i} 'prompt' must be a string")
        
        return BatchValidationResponse(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            estimated_items=len(request.json_data.get("input", []))
        )
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/batches/cost-estimate", response_model=BatchCostEstimate)
async def estimate_batch_cost(request: OpenAIBatchCreateRequest):
    """Estimate cost for batch processing"""
    try:
        # Simple cost estimation (adjust rates as needed)
        input_tokens_per_prompt = 50  # Average estimate
        total_input_tokens = len(request.input) * input_tokens_per_prompt
        max_output_tokens = len(request.input) * (request.max_tokens or 256)
        
        # Example cost calculation (adjust rates based on actual pricing)
        input_cost_per_1k = 0.0001  # $0.10 per 1M input tokens
        output_cost_per_1k = 0.0002  # $0.20 per 1M output tokens
        
        input_cost = (total_input_tokens / 1000) * input_cost_per_1k
        output_cost = (max_output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        
        return BatchCostEstimate(
            input_tokens=total_input_tokens,
            max_output_tokens=max_output_tokens,
            estimated_cost_usd=total_cost
        )
        
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))