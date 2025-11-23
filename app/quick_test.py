#!/usr/bin/env python3
"""
Quick batch inference test for Qwen2.5-0.5B model
"""
import os
os.environ['HF_HOME'] = '/root/.cache/huggingface'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def main():
    print("Loading Qwen2.5-0.5B model...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', 
        device_map='auto', 
        dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    # Simple batch test
    prompts = [
        "What is AI?",
        "Explain ML.",
        "Neural networks?"
    ]
    
    print(f"\nRunning batch inference on {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip()
        
        end_time = time.time()
        print(f"Response: {generated}")
        print(f"Time: {end_time - start_time:.2f}s")
    
    print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

if __name__ == "__main__":
    main()