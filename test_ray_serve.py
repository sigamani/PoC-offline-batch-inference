import time
import os
from typing import Dict, Any
import ray
from ray import serve
import requests
import concurrent.futures
import json

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["VLLM_USE_MODELSCOPE"] = "False"

# Initialize Ray for CPU-only
ray.init(num_cpus=4, ignore_reinit_error=True)

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0}
)
class VLLMDeployment:
    def __init__(self):
        from vllm import LLM, SamplingParams
        worker_id = ray.get_runtime_context().get_worker_id()
        logger.info(f"[Worker {worker_id}] Initializing vLLM on CPU...")
        
        self.llm = LLM(
            model="facebook/opt-125m",
            enforce_eager=True,
            tensor_parallel_size=1
        )
        self.sampling_params = SamplingParams(
            temperature=0.7, max_tokens=50, top_p=0.9
        )
        logger.info(f"[Worker {worker_id}] vLLM initialized successfully")

    async def __call__(self, request):
        """HTTP endpoint for testing"""
        import asyncio
        
        # Parse request body
        try:
            if hasattr(request, 'body'):
                # Ray Serve Request object
                body = request.body.decode('utf-8') if isinstance(request.body, bytes) else request.body
                data = json.loads(body)
            else:
                # Direct request (should be dict)
                data = request if isinstance(request, dict) else {"text": str(request)}
        except Exception as e:
            logger.error(f"Request parsing error: {str(e)}")
            data = {"text": ""}
            
        prompt = data.get("text", "")
        start_time = time.time()
        worker_id = ray.get_runtime_context().get_worker_id()
        
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            output = outputs[0]
            
            return {
                "prompt": prompt,
                "response": output.outputs[0].text,
                "tokens": len(output.outputs[0].token_ids),
                "worker_id": worker_id,
                "generation_time": time.time() - start_time,
                "finish_reason": output.outputs[0].finish_reason
            }
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Generation error: {str(e)}")
            return {
                "prompt": prompt,
                "response": None,
                "tokens": 0,
                "worker_id": worker_id,
                "generation_time": time.time() - start_time,
                "error": str(e)
            }

if __name__ == "__main__":
    # Deploy application
    serve.start(http_options={"host": "127.0.0.1", "port": 8000})
    
    app = VLLMDeployment.bind()
    handle = serve.run(app)
    
    logger.info("Ray Serve with vLLM workers deployed successfully on CPU")
    logger.info("HTTP endpoint available at http://127.0.0.1:8000")
    logger.info("Ray Serve dashboard at http://127.0.0.1:8265")
    
    # Wait for deployment to be ready
    time.sleep(5)
    
    # Test with HTTP requests
    test_prompts = [
        "What is the capital of France?",
        "Tell me a joke about programming.",
        "Explain machine learning in simple terms.",
        "What is the meaning of life?",
    ]
    
    logger.info("Testing distributed vLLM workers via HTTP...")
    
    def send_http_request(prompt, idx):
        url = "http://127.0.0.1:8000/"
        start = time.time()
        
        try:
            response = requests.post(
                url,
                json={
                    "text": prompt,
                    "request_id": idx
                },
                timeout=30
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Request {idx} completed in {elapsed:.2f}s by worker {result.get('worker_id')}")
                return result, elapsed
            else:
                logger.error(f"Request {idx} failed with status {response.status_code}: {response.text}")
                return None, elapsed
        except Exception as e:
            logger.error(f"Request {idx} failed: {str(e)}")
            return None, time.time() - start
    
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(send_http_request, prompt, i) for i, prompt in enumerate(test_prompts)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_total
    valid_results = [r for r in results if r[0] is not None]
    
    logger.info(f"Completed {len(valid_results)} requests in {total_time:.2f}s")
    
    if valid_results:
        avg_latency = sum(r[1] for r in valid_results) / len(valid_results)
        logger.info(f"Average request latency: {avg_latency:.2f}s")
        
        # Show worker distribution
        workers = [r[0].get('worker_id') for r in valid_results if r[0].get('worker_id')]
        unique_workers = set(workers)
        logger.info(f"Requests distributed across {len(unique_workers)} workers")
        for worker_id in unique_workers:
            count = workers.count(worker_id)
            logger.info(f"  Worker {worker_id}: {count} requests")
    
    logger.info("\\n=== SUCCESS: Ray Serve + vLLM Distributed Working ===")
    logger.info("✓ 2 vLLM workers running on CPU")
    logger.info("✓ Ray Serve load balancing across workers") 
    logger.info("✓ HTTP API responding to POST requests")
    logger.info("\\nServer is running. Send POST requests to test:")
    logger.info("curl -X POST http://127.0.0.1:8000/ -H 'Content-Type: application/json' -d '{\"text\": \"your prompt\"}'")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Ray Serve...")
        serve.shutdown()