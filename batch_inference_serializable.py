#!/usr/bin/env python3
"""
Ray Data + vLLM batch inference - Serialization-safe implementation
Ray 2.49.1 + vLLM 0.10.0
"""

import os
import time
import sys
import yaml
import logging
from typing import Dict, Any
from dataclasses import dataclass, asdict

import ray
from ray import data
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Track batch inference metrics"""
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    start_time: float = None
    tokens_processed: int = 0
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
    
    def throughput_per_sec(self) -> float:
        elapsed = time.time() - self.start_time
        return self.completed_requests / elapsed if elapsed > 0 else 0
    
    def tokens_per_sec(self) -> float:
        elapsed = time.time() - self.start_time
        return self.tokens_processed / elapsed if elapsed > 0 else 0
    
    def eta_hours(self) -> float:
        throughput = self.throughput_per_sec()
        if throughput == 0:
            return float('inf')
        remaining = self.total_requests - self.completed_requests
        return (remaining / throughput) / 3600
    
    def progress_pct(self) -> float:
        return (self.completed_requests / self.total_requests) * 100


# Global metrics (shared across workers)
GLOBAL_METRICS = None


def load_config(config_path: str = "/config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        return {
            "model": {
                "name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
                "max_model_len": 32768,
                "tensor_parallel_size": 1,
            },
            "inference": {
                "batch_size": 128,
                "concurrency": 2,
                "gpu_memory_utilization": 0.90,
                "temperature": 0.7,
                "max_tokens": 512,
            },
            "data": {
                "input_path": "/data/input/sharegpt_sample.json",
                "output_path": "/data/output/",
                "num_samples": 1000,
            },
            "sla": {
                "target_hours": 24,
                "buffer_factor": 0.7,
            }
        }


def load_sharegpt_data(config: Dict[str, Any]) -> data.Dataset:
    """Load and format ShareGPT dataset"""
    input_path = config["data"]["input_path"]
    num_samples = config["data"]["num_samples"]
    
    logger.info(f"Loading data from {input_path}")
    
    if os.path.exists(input_path):
        ds = data.read_json(input_path)
    else:
        logger.info("Loading ShareGPT from HuggingFace")
        from datasets import load_dataset
        hf_ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered", 
            split=f"train[:{num_samples}]"
        )
        ds = data.from_huggingface(hf_ds)
    
    def format_sharegpt(row):
        messages = row.get("conversations", [])
        prompt = next((m["value"] for m in messages if m["from"] == "human"), "Hello")
        return {"prompt": prompt}
    
    ds = ds.map(format_sharegpt)
    logger.info(f"Loaded {ds.count()} samples")
    return ds


# Simple serializable functions (no class dependencies)
def preprocess_row(row: Dict, config: Dict) -> Dict:
    """Preprocess function - must be top-level for serialization"""
    return {
        "prompt": row["prompt"],
        "sampling_params": {
            "temperature": config["inference"]["temperature"],
            "max_tokens": config["inference"]["max_tokens"],
            "top_p": 0.9,
        }
    }


def postprocess_row(row: Dict) -> Dict:
    """Postprocess function - must be top-level for serialization"""
    generated_text = row.get("generated_text", "")
    
    # Estimate tokens
    tokens = len(generated_text.split()) * 1.3
    
    return {
        "prompt": row.get("prompt", ""),
        "response": generated_text,
        "tokens": int(tokens),
    }


def log_progress(metrics: BatchMetrics, sla_hours: float):
    """Log progress and check SLA"""
    logger.info(
        f"Progress: {metrics.progress_pct():.1f}% | "
        f"Completed: {metrics.completed_requests}/{metrics.total_requests} | "
        f"Throughput: {metrics.throughput_per_sec():.2f} req/s | "
        f"Tokens/sec: {metrics.tokens_per_sec():.2f} | "
        f"ETA: {metrics.eta_hours():.2f}h | "
        f"Failed: {metrics.failed_requests}"
    )
    
    # Check SLA
    eta = metrics.eta_hours()
    elapsed_hours = (time.time() - metrics.start_time) / 3600
    remaining_hours = sla_hours - elapsed_hours
    
    if eta > remaining_hours:
        logger.warning(f"SLA AT RISK! ETA {eta:.2f}h > Remaining {remaining_hours:.2f}h")


def run_batch_inference(config: Dict[str, Any]):
    """Run distributed batch inference with Ray Data + vLLM"""
    
    # Load dataset
    ds = load_sharegpt_data(config)
    
    # Initialize metrics
    metrics = BatchMetrics(total_requests=ds.count())
    
    logger.info("=" * 80)
    logger.info("Starting Ray Data + vLLM Batch Inference")
    logger.info("=" * 80)
    logger.info(f"Ray version: {ray.__version__}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Total requests: {metrics.total_requests}")
    logger.info(f"Batch size: {config['inference']['batch_size']}")
    logger.info(f"Concurrency: {config['inference']['concurrency']}")
    logger.info(f"Target SLA: {config['sla']['target_hours']} hours")
    logger.info("=" * 80)
    
    # Configure vLLM processor
    vllm_config = vLLMEngineProcessorConfig(
        model=config["model"]["name"],
        concurrency=config["inference"]["concurrency"],
        batch_size=config["inference"]["batch_size"],
        engine_kwargs={
            "max_model_len": config["model"]["max_model_len"],
            "gpu_memory_utilization": config["inference"]["gpu_memory_utilization"],
            "tensor_parallel_size": config["model"]["tensor_parallel_size"],
            "trust_remote_code": True,
            "enable_chunked_prefill": True,
        }
    )
    
    # Create simple lambda functions with config baked in (serializable)
    preprocess_fn = lambda row: preprocess_row(row, config)
    postprocess_fn = lambda row: postprocess_row(row)
    
    # Build processor using official Ray Data API
    logger.info("Building vLLM processor...")
    processor = build_llm_processor(
        vllm_config,
        preprocess=preprocess_fn,
        postprocess=postprocess_fn,
    )
    
    # Run inference
    logger.info("Starting distributed inference...")
    start_time = time.time()
    
    result_ds = processor(ds)
    
    # Write results incrementally with progress tracking
    output_path = config["data"]["output_path"]
    logger.info(f"Writing results to {output_path}")
    
    # Track progress during write
    processed_count = 0
    log_interval = 100
    
    def track_batch(batch):
        """Track progress on each batch written"""
        nonlocal processed_count
        batch_size = len(batch["prompt"])
        total_tokens = sum(batch["tokens"])
        
        processed_count += batch_size
        metrics.completed_requests = processed_count
        metrics.tokens_processed += total_tokens
        
        if processed_count % log_interval == 0 or processed_count >= metrics.total_requests:
            log_progress(metrics, config["sla"]["target_hours"])
        
        return batch
    
    # Add progress tracking to pipeline
    result_ds = result_ds.map_batches(track_batch, batch_format="pandas")
    
    # Write to parquet
    result_ds.write_parquet(
        output_path,
        try_create_dir=True,
    )
    
    # Final report
    total_time = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("Batch Inference Complete!")
    logger.info("=" * 80)
    log_progress(metrics, config["sla"]["target_hours"])
    logger.info(f"Total time: {total_time / 3600:.2f} hours")
    logger.info(f"Average throughput: {metrics.throughput_per_sec():.2f} req/s")
    logger.info(f"Average tokens/sec: {metrics.tokens_per_sec():.2f}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # Worker mode - just connect and wait
        head_address = sys.argv[2] if len(sys.argv) > 2 else "ray-head:6379"
        ray.init(address=head_address, _redis_password="ray123")
        logger.info(f"Ray worker connected to {head_address}")
        
        # Workers wait for tasks from head node
        while True:
            time.sleep(60)
    else:
        # Head mode - run the batch inference
        ray.init(address="local", _redis_password="ray123")
        logger.info("Ray head initialized")
        
        # Load config and run inference
        config = load_config()
        run_batch_inference(config)


if __name__ == "__main__":
    main()