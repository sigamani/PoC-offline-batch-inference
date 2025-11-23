#!/usr/bin/env python3
"""
Ray Serve deployment for LLM inference with built-in load balancing
"""
import ray
from ray import serve
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from fastapi import Request

# Initialize Ray
ray.init(address="auto")

# Start Ray Serve on port 8002
serve.start(http_options={"host": "0.0.0.0", "port": 8002})

@serve.deployment(num_replicas=2)
class LLMInference:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        print(f"Loading {self.model_name} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Model loaded on device: {next(self.model.parameters()).device}")
    
    def __call__(self, request: Request):
        """Handle inference requests"""
        data = request.json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 100)
        temperature = data.get("temperature", 0.7)
        
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
            'inference_time': inference_time,
            'worker_id': serve.get_replica_context().replica_id
        }

# Deploy the inference service
llm_app = LLMInference.bind()

print("\n🚀 Ray Serve LLM Inference Started")
print("📍 Endpoints:")
print("  POST / - Single inference")  
print(f"🌐 Access: http://localhost:8002")
print(f"📊 Dashboard: http://localhost:8265")