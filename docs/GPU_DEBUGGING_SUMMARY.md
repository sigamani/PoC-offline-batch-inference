# Docker GPU Debugging Summary

## Problem
Docker container could not access NVIDIA GPUs, getting error: "could not select device driver with capabilities: [[gpu]]"

## Root Cause
NVIDIA Container Toolkit was not properly installed or configured for Docker.

## Solution Steps

1. **Install NVIDIA Container Toolkit:**
   ```bash
   apt update && apt install -y nvidia-container-toolkit
   ```

2. **Configure Docker for NVIDIA runtime:**
   ```bash
   nvidia-ctk runtime configure --runtime=docker
   systemctl restart docker
   ```

3. **Verify GPU access in Docker:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

## Working Command for Your Use Case

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/workspace:/workspace \
  michaelsigamani/proj-grounded-telescopes:0.1.0 \
  python /workspace/your_script.py
```

## Key Parameters Explained

- `--gpus all`: Grants access to all NVIDIA GPUs
- `--ipc=host`: Enables shared memory for better performance
- `--ulimit memlock=-1`: Removes memory lock limits
- `--ulimit stack=67108864`: Increases stack size for large models
- `-v /root/.cache/huggingface:/root/.cache/huggingface`: Mounts HuggingFace model cache
- `-v /root/workspace:/workspace`: Mounts your code directory

## Performance Results

- Model: Qwen2.5-0.5B successfully loaded on RTX 3090
- GPU Memory Usage: ~0.93GB
- Inference Speed: ~2-4 seconds per 50-token generation
- Batch processing working efficiently

## Model Location

Your Qwen2.5-0.5B model is located at:
```
/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/
```

The model can be loaded using the HuggingFace cache with model ID: `Qwen/Qwen2.5-0.5B`