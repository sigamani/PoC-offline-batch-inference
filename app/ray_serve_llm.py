#!/usr/bin/env python3
"""
Ray Serve deployment for LLM inference with built-in load balancing
"""

import ray
from ray import serve
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import asyncio
from fastapi import Request

# Initialize Ray
ray.init(address="auto")

# Start Ray Serve
serve.start(http_options={"host": "0.0.0.0", "port": 8002})


@serve.deployment(
    num_replicas=2,  # This will create 2 replicas for load balancing
    max_concurrent_requests=10,
)
class LLMInference:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        print(f"Loading {self.model_name} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Model loaded on device: {next(self.model.parameters()).device}")

    async def generate(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.7
    ):
        """Generate text with timing"""
        start_time = time.time()

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_response[len(prompt) :].strip()

        inference_time = time.time() - start_time
        tokens_generated = len(self.tokenizer.encode(generated_text))

        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "inference_time": inference_time,
            "worker_id": serve.get_replica_context().replica_id,
        }


# Deploy the inference service
LLMInference.deploy()


@serve.deployment(route_prefix="/")
class APIGateway:
    async def __call__(self, request: Request):
        """API Gateway that routes to LLM inference"""
        if request.url.path == "/health":
            return {
                "status": "healthy",
                "service": "ray-serve-llm",
                "replicas": serve.get_deployment_status("LLMInference").replica_states,
            }

        elif request.url.path == "/generate":
            data = await request.json()
            inference_handle = LLMInference.get_handle()
            result = await inference_handle.generate.remote(
                data.get("prompt", ""),
                data.get("max_tokens", 100),
                data.get("temperature", 0.7),
            )
            return result

        elif request.url.path == "/generate_batch":
            data = await request.json()
            inference_handle = LLMInference.get_handle()
            prompts = data.get("prompts", [])

            # Process batch in parallel
            tasks = [
                inference_handle.generate.remote(
                    prompt, data.get("max_tokens", 100), data.get("temperature", 0.7)
                )
                for prompt in prompts
            ]

            results = await asyncio.gather(*tasks)
            return {
                "results": results,
                "total_time": sum(r["inference_time"] for r in results),
            }

        return {
            "message": "LLM Inference Service",
            "endpoints": ["/health", "/generate", "/generate_batch"],
        }


# Deploy the API gateway
APIGateway.deploy()

print("\n🚀 Ray Serve LLM Inference Started")
print("📍 Endpoints:")
print("  GET  /health - Health check")
print("  POST /generate - Single inference")
print("  POST /generate_batch - Batch inference")
print("🌐 Access: http://localhost:8002")
print("📊 Dashboard: http://localhost:8265")
