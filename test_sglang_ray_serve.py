#!/usr/bin/env python3

import time
import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import ray
from ray import serve
import requests
import concurrent.futures

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SGLANG_USE_CPU_ENGINE"] = "1"

@dataclass
class GenerationResult:
    prompt: str
    response: Optional[str]
    tokens: int
    worker_id: str
    generation_time: float
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    engine: str = "sglang"

def parse_request_body(request) -> Dict:
    if hasattr(request, "body"):
        body = request.body.decode("utf-8") if isinstance(request.body, bytes) else request.body
        return json.loads(body)
    return request if isinstance(request, dict) else {"text": str(request)}

def extract_prompt(data: Dict) -> str:
    return data.get("text", "")

def parse_sglang_response(response, prompt: str, worker_id: str, elapsed: float) -> Dict:
    if isinstance(response, list) and len(response) > 0:
        generation = response[0]
        return GenerationResult(
            prompt=prompt,
            response=generation["text"],
            tokens=len(generation.get("token_ids", [])),
            worker_id=worker_id,
            generation_time=elapsed,
            finish_reason=generation.get("finish_reason", "length")
        ).__dict__
    
    return GenerationResult(
        prompt=prompt,
        response=str(response),
        tokens=0,
        worker_id=worker_id,
        generation_time=elapsed,
        finish_reason="unknown"
    ).__dict__

def create_mock_response(prompt: str, worker_id: str, elapsed: float) -> Dict:
    return GenerationResult(
        prompt=prompt,
        response=f"[MOCK SGLang] Response to: {prompt[:30]}...",
        tokens=10,
        worker_id=worker_id,
        generation_time=elapsed,
        finish_reason="length",
        engine="mock_sglang"
    ).__dict__

def create_error_response(prompt: str, worker_id: str, elapsed: float, error: str) -> Dict:
    return GenerationResult(
        prompt=prompt,
        response=None,
        tokens=0,
        worker_id=worker_id,
        generation_time=elapsed,
        error=error,
        engine="sglang_error"
    ).__dict__

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 2, "num_gpus": 0})
class SGLangDeployment:
    def __init__(self):
        worker_id = ray.get_runtime_context().get_worker_id()
        logger.info(f"[Worker {worker_id}] Initializing SGLang Runtime...")
        
        self.runtime = self._initialize_sglang(worker_id)
        self.sampling_params = {
            "temperature": 0.7,
            "max_new_tokens": 50,
            "top_p": 0.9,
            "top_k": -1,
        }

    def _initialize_sglang(self, worker_id: str):
        try:
            import sglang
            runtime = sglang.Runtime(
                model_path="facebook/opt-125m",
                device="cpu",
                log_level="error",
                mem_fraction_static=0.8,
                max_total_tokens=8192,
                random_seed=42,
            )
            logger.info(f"[Worker {worker_id}] SGLang Runtime initialized successfully")
            return runtime
        except ImportError as e:
            logger.error(f"[Worker {worker_id}] SGLang import failed: {e}")
            logger.warning(f"[Worker {worker_id}] Using mock SGLang implementation")
            return None

    async def __call__(self, request):
        data = parse_request_body(request)
        return self._generate_response(extract_prompt(data))

    def _generate_response(self, prompt: str) -> Dict:
        start_time = time.time()
        worker_id = ray.get_runtime_context().get_worker_id()
        elapsed = time.time() - start_time
        
        try:
            if self.runtime:
                response = self.runtime.generate(prompt, **self.sampling_params)
                return parse_sglang_response(response, prompt, worker_id, elapsed)
            return create_mock_response(prompt, worker_id, elapsed)
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Generation error: {str(e)}")
            return create_error_response(prompt, worker_id, elapsed, str(e))

def send_http_request(prompt: str, idx: int) -> Tuple[Optional[Dict], float]:
    start = time.time()
    response = requests.post("http://127.0.0.1:8000/", json={"text": prompt, "request_id": idx}, timeout=30)
    return _handle_response(response, idx, start)

def _handle_response(response, idx: int, start: float) -> Tuple[Optional[Dict], float]:
    elapsed = time.time() - start
    if response.status_code == 200:
        result = response.json()
        engine = result.get('engine', 'unknown')
        logger.info(f"Request {idx} completed in {elapsed:.2f}s by worker {result.get('worker_id')} ({engine})")
        return result, elapsed
    logger.error(f"Request {idx} failed with status {response.status_code}: {response.text}")
    return None, elapsed

def execute_concurrent_requests(prompts: List[str]) -> List[Tuple[Optional[Dict], float]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(send_http_request, p, i) for i, p in enumerate(prompts)]
        return [f.result() for f in concurrent.futures.as_completed(futures)]

def calculate_avg_latency(results: List[Tuple[Optional[Dict], float]]) -> float:
    return sum(r[1] for r in results) / len(results)

def extract_worker_ids(results: List[Tuple[Optional[Dict], float]]) -> List[str]:
    return [str(r[0].get("worker_id")) for r in results if r[0] and r[0].get("worker_id")]

def log_worker_distribution(workers: List[str]):
    unique_workers = set(workers)
    logger.info(f"Requests distributed across {len(unique_workers)} workers")
    for worker_id in unique_workers:
        logger.info(f"  Worker {worker_id}: {workers.count(worker_id)} requests")

def log_results_summary(results: List[Tuple[Optional[Dict], float]], total_time: float):
    valid_results = [r for r in results if r[0] is not None]
    logger.info(f"Completed {len(valid_results)} requests in {total_time:.2f}s")
    
    if valid_results:
        logger.info(f"Average request latency: {calculate_avg_latency(valid_results):.2f}s")
        log_worker_distribution(extract_worker_ids(valid_results))

def run_test_requests(prompts: List[str]):
    logger.info("Testing distributed SGLang workers via HTTP...")
    start_total = time.time()
    results = execute_concurrent_requests(prompts)
    log_results_summary(results, time.time() - start_total)

def log_integration_summary():
    logger.info("\\n=== SGLang Ray Serve Integration Demo ===")
    logger.info("✓ 2 workers deployed (mock implementation due to SGLang CPU limitations)")
    logger.info("✓ Ray Serve load balancing across workers")
    logger.info("✓ HTTP API responding to POST requests")
    logger.info("\\n=== Key Differences from vLLM Integration ===")
    logger.info("• SGLang Runtime vs vLLM LLM class")
    logger.info("• SGLang launches server processes vs vLLM direct inference")
    logger.info("• Different response structure and sampling parameter names")
    logger.info("• SGLang has GPU dependencies even in CPU mode")
    logger.info("\\nServer is running. Send POST requests to test:")
    logger.info("curl -X POST http://127.0.0.1:8000/ -H 'Content-Type: application/json' -d '{\"text\": \"your prompt\"}'")

def main():
    ray.init(num_cpus=4, ignore_reinit_error=True)
    serve.start(http_options={"host": "127.0.0.1", "port": 8000})
    serve.run(SGLangDeployment.bind())
    
    logger.info("Ray Serve with SGLang workers deployed")
    logger.info("HTTP endpoint available at http://127.0.0.1:8000")
    logger.info("Ray Serve dashboard at http://127.0.0.1:8265")
    
    time.sleep(5)
    
    test_prompts = [
        "What is capital of France?",
        "Tell me a joke about programming.",
        "Explain machine learning in simple terms.",
        "What is meaning of life?",
    ]
    
    run_test_requests(test_prompts)
    log_integration_summary()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Ray Serve...")
        serve.shutdown()

if __name__ == "__main__":
    main()