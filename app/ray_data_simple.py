#!/usr/bin/env python3
"""
Pure Ray Data + vLLM distributed batch inference
No FastAPI/uvicorn needed - Ray Data handles everything
Following official documentation: https://docs.ray.io/en/latest/data/batch_inference.html
"""
import os
import sys
import time
import argparse
from typing import List, Dict, Optional
import ray
from ray import data
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
    ['model_name']
)

active_batches = Gauge(
    'ray_data_active_batches',
    'Active Ray Data batches'
)

def setup_ray_cluster():
    """Initialize Ray cluster"""
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

@ray.remote
class VLLMWorker:
    """Ray actor for vLLM inference"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.node_id = ray.get_runtime_context().get_node_id()
        self.load_model()
    
    def load_model(self):
        """Initialize vLLM with GPU"""
        try:
            from vllm import LLM, SamplingParams
            
            print(f"Loading {self.model_name} on {self.node_id}...")
            
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                trust_remote_code=True
            )
            
            self.use_vllm = True
            print(f"✅ vLLM model loaded on {self.node_id}")
            
        except Exception as e:
            print(f"⚠ vLLM failed, falling back to transformers: {e}")
            self.load_transformers_fallback()
    
    def load_transformers_fallback(self):
        """Fallback to transformers"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"Loading {self.model_name} with transformers on {self.node_id}...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='auto',
                dtype=torch.float16,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.use_vllm = False
            print(f"✅ Transformers model loaded on {self.node_id}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def infer_batch(self, prompts: List[str], max_tokens: int, temperature: float) -> List[Dict]:
        """Process a batch of prompts"""
        start_time = time.time()
        
        if self.use_vllm:
            results = self._infer_with_vllm(prompts, max_tokens, temperature)
        else:
            results = self._infer_with_transformers(prompts, max_tokens, temperature)
        
        inference_time = time.time() - start_time
        
        # Add node info to results
        for result in results:
            result['node_id'] = self.node_id
            result['inference_time'] = inference_time / len(prompts)  # Average time per prompt
        
        return results
    
    def _infer_with_vllm(self, prompts: List[str], max_tokens: int, temperature: float) -> List[Dict]:
        """Inference using vLLM"""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append({
                'text': generated_text,
                'tokens_generated': len(output.outputs[0].token_ids),
                'inference_time': 0  # Will be set by caller
            })
        
        return results
    
    def _infer_with_transformers(self, prompts: List[str], max_tokens: int, temperature: float) -> List[Dict]:
        """Inference using transformers"""
        import torch
        
        results = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_response[len(prompt):].strip()
            
            results.append({
                'text': generated_text,
                'tokens_generated': len(self.tokenizer.encode(generated_text)),
                'inference_time': 0  # Will be set by caller
            })
        
        return results

def vllm_inference(batch):
    """Ray Data map_batches function"""
    # batch is a list of prompts
    prompts = list(batch)
    
    # Get current actor (this runs on actor itself)
    current_actor = ray.get_runtime_context().current_actor
    
    # Use the actor to process the batch
    return ray.get(current_actor.infer_batch.remote(prompts, 100, 0.7))

def run_batch_inference(prompts: List[str], max_tokens: int = 100, temperature: float = 0.7):
    """Run distributed batch inference using Ray Data vLLM"""
    print(f"🔄 Processing {len(prompts)} prompts with Ray Data vLLM...")
    
    start_time = time.time()
    
    try:
        model_name = os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-0.5B-Instruct')
        
        # Create vLLM workers (one per GPU node)
        nodes = ray.nodes()
        gpu_nodes = [node for node in nodes if node.get('Resources', {}).get('GPU', 0) > 0]
        
        print(f"Found {len(gpu_nodes)} GPU nodes")
        
        # Create actors on GPU nodes
        actors = []
        for node in gpu_nodes:
            actor = VLLMActor.options(
                num_gpus=1,
                resources={f"node:{node['NodeID']}": 1}
            ).remote(model_name)
            actors.append(actor)
            print(f"Created VLLM actor on node {node['NodeID']}")
        
        # Create Ray Dataset from prompts
        ds = data.from_items(prompts)
        
        # Use map_batches for distributed processing
        # This automatically distributes batches across available actors
        batch_results = ds.map_batches(
            vllm_inference,
            batch_size=4,   # Optimize for vLLM throughput
            num_gpus=1,   # Each actor gets 1 GPU
            concurrency=2     # 2 parallel actors (one per node)
        )
        
        # Collect results
        results = []
        for batch in batch_results.iter_batches():
            results.extend(batch)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        throughput = len(prompts) / total_time if total_time > 0 else 0
        nodes_used = len(set(result['node_id'] for result in results))
        
        # Record metrics
        inference_requests.labels(
            method="batch_inference",
            status="success"
        ).inc(len(prompts))
        
        inference_duration.labels(
            model_name=model_name
        ).observe(total_time)
        
        print(f"✅ Processed {len(prompts)} prompts in {total_time:.2f}s")
        print(f"📊 Throughput: {throughput:.2f} prompts/second")
        print(f"🖥 Nodes used: {nodes_used}")
        
        return {
            'results': results,
            'total_time': total_time,
            'total_prompts': len(prompts),
            'nodes_used': nodes_used,
            'throughput': throughput,
            'model_name': model_name
        }
        
    except Exception as e:
        inference_requests.labels(
            method="batch_inference",
            status="error"
        ).inc(len(prompts))
        print(f"❌ Batch inference failed: {e}")
        raise

def start_metrics_server():
    """Start Prometheus metrics server"""
    try:
        start_http_server(8001)
        print("📊 Prometheus metrics server started on port 8001")
    except Exception as e:
        print(f"⚠️ Metrics server failed to start: {e}")

def main():
    """Main function - pure Ray Data processing"""
    parser = argparse.ArgumentParser(description='Ray Data vLLM Batch Inference')
    parser.add_argument('--mode', choices=['head', 'worker', 'inference'], 
                       default='inference', help='Running mode')
    parser.add_argument('--head-address', default='localhost:6379', 
                       help='Ray head address (for worker mode)')
    parser.add_argument('--prompts', nargs='+', default=['Hello, how are you?', 'What is AI?'],
                       help='Prompts to process')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Generation temperature')
    parser.add_argument('--metrics', action='store_true',
                       help='Start metrics server')
    
    args = parser.parse_args()
    
    # Setup Ray cluster
    setup_ray_cluster()
    
    # Start metrics server if requested
    if args.metrics:
        start_metrics_server()
    
    if args.mode == 'inference':
        # Run batch inference
        results = run_batch_inference(
            prompts=args.prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Print results
        print("\n📋 Results:")
        print("=" * 50)
        for i, result in enumerate(results['results']):
            print(f"\n{i+1}. Prompt: {args.prompts[i] if i < len(args.prompts) else 'Unknown'}")
            print(f"   Response: {result['text']}")
            print(f"   Tokens: {result['tokens_generated']}")
            print(f"   Node: {result['node_id']}")
        
        print(f"\n📊 Summary:")
        print(f"   Total prompts: {results['total_prompts']}")
        print(f"   Total time: {results['total_time']:.2f}s")
        print(f"   Throughput: {results['throughput']:.2f} prompts/second")
        print(f"   Nodes used: {results['nodes_used']}")
        print(f"   Model: {results['model_name']}")
        
    elif args.mode == 'head':
        print("🚀 Ray head node running. Use --inference mode to process prompts.")
        print("📊 Dashboard: http://localhost:8265")
        
        # Keep head node alive
        try:
            while True:
                time.sleep(60)
                nodes = ray.nodes()
                gpu_nodes = [node for node in nodes if node.get('Resources', {}).get('GPU', 0) > 0]
                print(f"📈 Cluster status: {len(nodes)} nodes, {len(gpu_nodes)} GPU nodes")
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Ray cluster...")
            ray.shutdown()
    
    elif args.mode == 'worker':
        print(f"🔧 Worker node running, connected to {args.head_address}")
        print("📊 Ready for batch inference tasks")
        
        # Keep worker alive
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down worker...")
            ray.shutdown()

if __name__ == "__main__":
    main()