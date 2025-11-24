#!/usr/bin/env python3
"""
Batch inference script for Qwen2.5-0.5B model with GPU support
"""

import os

os.environ["HF_HOME"] = "/root/.cache/huggingface"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from typing import List, Dict


def load_model():
    """Load the Qwen model with GPU support"""
    print("Loading Qwen2.5-0.5B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", device_map="auto", dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    print(f"Model loaded on device: {next(model.parameters()).device}")
    return model, tokenizer


def batch_inference(
    model, tokenizer, prompts: List[str], max_new_tokens: int = 100
) -> List[Dict]:
    """
    Run batch inference on a list of prompts

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        List of dictionaries containing prompt, response, and timing info
    """
    results = []

    print(f"Running batch inference on {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i + 1}/{len(prompts)}: {prompt[:50]}...")

        start_time = time.time()

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        end_time = time.time()
        inference_time = end_time - start_time

        # Extract only the generated part
        generated_text = response[len(prompt) :].strip()

        results.append(
            {
                "prompt": prompt,
                "response": generated_text,
                "full_response": response,
                "inference_time": inference_time,
                "tokens_generated": len(tokenizer.encode(generated_text)),
            }
        )

        print(
            f"Generated {len(tokenizer.encode(generated_text))} tokens in {inference_time:.2f}s"
        )

    return results


def main():
    """Main function to demonstrate batch inference"""

    # Load model
    model, tokenizer = load_model()

    # Example batch prompts
    batch_prompts = [
        "What is artificial intelligence?",
        "Explain the concept of machine learning.",
        "How do neural networks work?",
        "What is the difference between AI and ML?",
        "Describe the process of training a deep learning model.",
    ]

    # Run batch inference
    results = batch_inference(model, tokenizer, batch_prompts, max_new_tokens=150)

    # Print results
    print("\n" + "=" * 80)
    print("BATCH INFERENCE RESULTS")
    print("=" * 80)

    total_time = 0
    total_tokens = 0

    for i, result in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(f"Prompt: {result['prompt']}")
        print(f"Response: {result['response']}")
        print(f"Time: {result['inference_time']:.2f}s")
        print(f"Tokens: {result['tokens_generated']}")
        print(
            f"Tokens/sec: {result['tokens_generated'] / result['inference_time']:.1f}"
        )

        total_time += result["inference_time"]
        total_tokens += result["tokens_generated"]

    print("\n--- Summary ---")
    print(f"Total prompts: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens/sec: {total_tokens / total_time:.1f}")
    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")


if __name__ == "__main__":
    main()
