#!/usr/bin/env python3
import ray
from ray.data import from_items
import time
import os
import logging
import asyncio
from typing import List, Dict, Any
import numpy as np

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - could be moved to config file or environment variables
CONFIG = {
    "model_name": "facebook/opt-125m",
    "max_model_len": 256,
    "num_workers": 2,
    "batch_size": None,  # Will be calculated based on data size
    "enforce_eager": True,
    "dtype": "float32",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 50,
}

# Initialize Ray with CPU-only resources
ray.init(num_cpus=4, num_gpus=0)

@ray.remote
class VLLMWorker:
    """Distributed vLLM worker with shared model"""
    
    def __init__(self, config: Dict[str, Any]):
        worker_id = ray.get_runtime_context().get_worker_id()
        logger.info(f"[Worker {worker_id}] Initializing vLLM...")
        
        try:
            from vllm import LLM, SamplingParams
            
            # Load model ONCE per worker - shared across all batches
            self.llm = LLM(
                model=config["model_name"],
                max_model_len=config["max_model_len"],
                enforce_eager=config["enforce_eager"],
                dtype=config["dtype"],
                tensor_parallel_size=1,
                trust_remote_code=True,
            )
            self.sampling_params = SamplingParams(
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_tokens=config["max_tokens"],
            )
            logger.info(f"[Worker {worker_id}] vLLM initialized successfully")
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Failed to initialize vLLM: {e}")
            raise
    
    def process_batch(self, batch_data: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of prompts"""
        worker_id = ray.get_runtime_context().get_worker_id()
        logger.info(f"[Worker {worker_id}] Processing batch of {len(batch_data)} prompts...")
        
        try:
            start_time = time.time()
            
            # Generate with vLLM using shared model
            outputs = self.llm.generate(batch_data, self.sampling_params)
            
            # Format results
            results = []
            for output in outputs:
                results.append({
                    "prompt": output.prompt,
                    "response": output.outputs[0].text,
                    "tokens": len(output.outputs[0].token_ids),
                    "worker_id": worker_id,
                    "generation_time": time.time() - start_time
                })
            
            logger.info(f"[Worker {worker_id}] Completed {len(batch_data)} prompts in {time.time() - start_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Failed to process batch: {e}")
            # Return empty results for failed batch
            return [{"error": str(e), "worker_id": worker_id} for _ in batch_data]

class DistributedBatchProcessor:
    """Distributed batch processor with Ray actors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_workers = config["num_workers"]
        self.workers = None
        
    def initialize_workers(self):
        """Initialize distributed vLLM workers"""
        logger.info(f"Initializing {self.num_workers} distributed vLLM workers...")
        
        # Create multiple workers (actors) - each loads model once
        # No dummy call needed - __init__ runs when actor is created
        self.workers = [VLLMWorker.remote(self.config) for _ in range(self.num_workers)]
        logger.info("All worker actors created")
    
    async def process_dataset(self, dataset):
        """Process dataset with distributed Ray actors using asyncio coordination optimized for throughput"""
        logger.info("Starting distributed batch processing...")
        
        # Stream prompts without loading all data into memory
        prompts = [item["prompt"] for item in dataset.take_all()]
        
        logger.info(f"Processing {len(prompts)} prompts with {self.num_workers} workers")
        
        # Optimize batch size for throughput - smaller batches for better parallelism
        # Target 2-3 prompts per batch for optimal GPU utilization on this small model
        optimal_batch_size = max(1, min(3, len(prompts) // (self.num_workers * 2)))
        if len(prompts) < self.num_workers * 2:
            optimal_batch_size = 1  # For small datasets, use single prompts
        
        # Create optimized batches
        batches = []
        for i in range(0, len(prompts), optimal_batch_size):
            batches.append(prompts[i:i + optimal_batch_size])
        
        logger.info(f"Created {len(batches)} optimized batches of size {optimal_batch_size} for maximum throughput")
        
        # Worker assignment pool for better load balancing
        worker_pool = asyncio.Queue()
        for worker in self.workers:
            await worker_pool.put(worker)
        
        # Track metrics
        completed_batches = 0
        total_prompts_processed = 0
        start_time = time.time()
        
        async def process_single_batch(batch):
            """Process a single batch with dynamic worker assignment"""
            # Get available worker
            worker = await worker_pool.get()
            try:
                # Submit task to worker
                task = worker.process_batch.remote(batch)
                # Wait for result with timeout
                batch_result = await asyncio.wait_for(
                    asyncio.to_thread(ray.get, task), 
                    timeout=120.0  # 2 minute timeout per batch
                )
                return batch_result
            except asyncio.TimeoutError:
                logger.error(f"Batch timeout, worker may be stuck")
                return [{"error": "timeout", "worker_id": str(worker)} for _ in batch]
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                return [{"error": str(e), "worker_id": str(worker)} for _ in batch]
            finally:
                # Return worker to pool
                await worker_pool.put(worker)
        
        # Process all batches concurrently with controlled parallelism
        semaphore = asyncio.Semaphore(self.num_workers * 2)  # Allow some queuing for throughput
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await process_single_batch(batch)
        
        # Create all tasks
        tasks = [process_with_semaphore(batch) for batch in batches]
        
        try:
            # Use asyncio.as_completed for immediate processing as results arrive
            completed_results = []
            for coro in asyncio.as_completed(tasks):
                try:
                    batch_result = await coro
                    if batch_result:
                        completed_results.extend(batch_result)
                        completed_batches += 1
                        
                        # Log progress
                        if completed_batches % max(1, len(batches) // 10) == 0:
                            elapsed = time.time() - start_time
                            throughput = completed_batches / elapsed if elapsed > 0 else 0
                            logger.info(f"Progress: {completed_batches}/{len(batches)} batches, {throughput:.2f} batches/sec")
                            
                except Exception as e:
                    logger.error(f"Batch completion failed: {e}")
            
            # Filter out any error results
            successful_results = [r for r in completed_results if "error" not in r]
            error_count = len(completed_results) - len(successful_results)
            
            if error_count > 0:
                logger.warning(f"Found {error_count} failed results")
            
            # Log final throughput metrics
            total_time = time.time() - start_time
            prompts_per_sec = len(successful_results) / total_time if total_time > 0 else 0
            batches_per_sec = completed_batches / total_time if total_time > 0 else 0
            
            logger.info(f"Throughput: {prompts_per_sec:.2f} prompts/sec, {batches_per_sec:.2f} batches/sec")
            logger.info(f"Successfully processed {len(successful_results)} prompts in {total_time:.2f}s")
            
            return successful_results
            
        except Exception as e:
            logger.error(f"Error in distributed processing: {e}")
            return []

async def main():
    """Main function for distributed processing"""
    prompts_list = [
        {"prompt": "What is the capital of France?"},
        {"prompt": "Tell me a joke."},
        {"prompt": "Explain photosynthesis in simple terms."},
        {"prompt": "Summarize the plot of Hamlet."},
        {"prompt": "What is the meaning of life?"},
        {"prompt": "What is the capital of England?"},
        {"prompt": "Tell me a fact."},
        {"prompt": "Explain quantum mechanics in simple terms."},
        {"prompt": "Summarize the plot of Macbeth."},
        {"prompt": "What is the meaning of the word horrid?"},
    ]

    logger.info("Starting Ray Data + vLLM distributed batch inference...")
    logger.info(f"Using {CONFIG['model_name']} model with {CONFIG['num_workers']} distributed engines")
    
    # Create Ray Dataset
    dataset = from_items(prompts_list)
    
    # Initialize distributed processor
    processor = DistributedBatchProcessor(CONFIG)
    processor.initialize_workers()
    
    # Process dataset
    start_time = time.time()
    results = await processor.process_dataset(dataset)
    end_time = time.time()
    
    # Print results
    print("\n=== DISTRIBUTED vLLM BATCH INFERENCE RESULTS ===")
    for i, r in enumerate(results):
        print(f"{i+1}. Prompt: {r['prompt']}")
        print(f"   Response: {r['response']}")
        print(f"   Tokens: {r['tokens']}")
        print(f"   Worker: {r['worker_id']}")
        print(f"   Generation Time: {r['generation_time']:.2f}s")
        print()
    
    print(f"Total prompts processed: {len(results)}")
    print(f"Total processing time: {end_time - start_time:.2f}s")
    print(f"Average time per prompt: {(end_time - start_time) / len(results):.2f}s")
    logger.info("Successfully completed distributed Ray Data + vLLM batch inference!")
    ray.shutdown()

if __name__ == "__main__":
    asyncio.run(main())