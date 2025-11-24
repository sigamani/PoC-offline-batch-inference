#!/usr/bin/env python3
"""
Pure Ray Data + vLLM distributed batch inference
Using official Ray Data vLLM integration: https://docs.ray.io/en/latest/data/batch_inference.html
"""

import os
import sys
import time
import argparse
from typing import List
import ray
from ray import data
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from prometheus_client import (
    start_http_server,
    Gauge,
    Histogram,
    Counter,
)

# Prometheus metrics
inference_requests = Counter(
    "ray_data_requests_total", "Total Ray Data requests", ["method", "status"]
)

inference_duration = Histogram(
    "ray_data_inference_duration_seconds", "Ray Data inference duration", ["model_name"]
)

active_batches = Gauge("ray_data_active_batches", "Active Ray Data batches")


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
            _redis_password="ray123",
        )
        print("🚀 Ray head node started")


def run_batch_inference(
    prompts: List[str], max_tokens: int = 100, temperature: float = 0.7
):
    """Run distributed batch inference using Ray Data vLLM processor"""
    print(f"🔄 Processing {len(prompts)} prompts with Ray Data vLLM...")

    start_time = time.time()

    try:
        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

        # Configure vLLM processor using official Ray Data API
        config = vLLMEngineProcessorConfig(
            model=model_name,
            engine_kwargs={
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 4096,
                "max_model_len": 16384,
            },
            concurrency=2,  # 2 parallel actors (one per node)
            batch_size=64,  # Optimized for vLLM throughput
            model_source="huggingface",
            preprocess=lambda row: dict(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": row["item"]},
                ]
            ),
            sampling_params=dict(
                temperature=temperature, max_tokens=max_tokens, top_p=0.9
            ),
            postprocess=lambda row: dict(
                answer=row["generated_text"],
                prompt=row["item"],
                tokens_generated=len(row["generated_text"].split()),
                inference_time=0,  # Will be calculated below
            ),
        )

        # Build vLLM processor
        processor = build_llm_processor(config)
        print(f"✅ vLLM processor configured for model: {model_name}")

        # Create Ray Dataset from prompts
        ds = data.from_items([{"item": prompt} for prompt in prompts])

        # Process using Ray Data map_batches - this handles all orchestration
        processed_ds = processor(ds)

        # Collect results
        results = []
        for batch in processed_ds.iter_batches():
            for row in batch:
                results.append(
                    {
                        "text": row["answer"],
                        "tokens_generated": row["tokens_generated"],
                        "prompt": row["prompt"],
                        "node_id": "ray-cluster",  # Ray Data handles node allocation
                    }
                )

        total_time = time.time() - start_time

        # Calculate metrics
        throughput = len(prompts) / total_time if total_time > 0 else 0

        # Record metrics
        inference_requests.labels(method="batch_inference", status="success").inc(
            len(prompts)
        )

        inference_duration.labels(model_name=model_name).observe(total_time)

        print(f"✅ Processed {len(prompts)} prompts in {total_time:.2f}s")
        print(f"📊 Throughput: {throughput:.2f} prompts/second")

        return {
            "results": results,
            "total_time": total_time,
            "total_prompts": len(prompts),
            "throughput": throughput,
            "model_name": model_name,
        }

    except Exception as e:
        inference_requests.labels(method="batch_inference", status="error").inc(
            len(prompts)
        )
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
    parser = argparse.ArgumentParser(description="Ray Data vLLM Batch Inference")
    parser.add_argument(
        "--mode",
        choices=["head", "worker", "inference"],
        default="inference",
        help="Running mode",
    )
    parser.add_argument(
        "--head-address",
        default="localhost:6379",
        help="Ray head address (for worker mode)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["Hello, how are you?", "What is AI?"],
        help="Prompts to process",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature"
    )
    parser.add_argument("--metrics", action="store_true", help="Start metrics server")

    args = parser.parse_args()

    # Setup Ray cluster
    setup_ray_cluster()

    # Start metrics server if requested
    if args.metrics:
        start_metrics_server()

    if args.mode == "inference":
        # Run batch inference
        results = run_batch_inference(
            prompts=args.prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Print results
        print("\n📋 Results:")
        print("=" * 50)
        for i, result in enumerate(results["results"]):
            print(
                f"\n{i + 1}. Prompt: {args.prompts[i] if i < len(args.prompts) else 'Unknown'}"
            )
            print(f"   Response: {result['text']}")
            print(f"   Tokens: {result['tokens_generated']}")
            print(f"   Node: {result['node_id']}")

        print("\n📊 Summary:")
        print(f"   Total prompts: {results['total_prompts']}")
        print(f"   Total time: {results['total_time']:.2f}s")
        print(f"   Throughput: {results['throughput']:.2f} prompts/second")
        print(f"   Model: {results['model_name']}")

    elif args.mode == "head":
        print("🚀 Ray head node running. Use --inference mode to process prompts.")
        print("📊 Dashboard: http://localhost:8265")

        # Keep head node alive
        try:
            while True:
                time.sleep(60)
                nodes = ray.nodes()
                gpu_nodes = [
                    node
                    for node in nodes
                    if node.get("Resources", {}).get("GPU", 0) > 0
                ]
                print(
                    f"📈 Cluster status: {len(nodes)} nodes, {len(gpu_nodes)} GPU nodes"
                )
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Ray cluster...")
            ray.shutdown()

    elif args.mode == "worker":
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
