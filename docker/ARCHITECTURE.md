# Docker Setup for vLLM Batch Inference

This directory contains Docker configurations for running the vLLM batch inference system with proper ARM/M1 compatibility.

## Files

- `Dockerfile.api` - API service container (FastAPI + Ray Data)
- `Dockerfile.vllm` - vLLM engine container (direct vLLM integration)
- `docker-compose.yml` - Complete environment setup

## Quick Start

### API Service Only
```bash
docker-compose -f docker-compose.yml up api
```

### vLLM Engine (Apple Silicon)
```bash
docker-compose -f docker-compose.yml up vllm
```

### Development (Both Services)
```bash
docker-compose -f docker-compose.yml up
```

## Environment Variables

- `RAY_BACKEND=log` - Enable Ray logging
- `VLLM_USE_CPU=1` - Force CPU mode (for ARM compatibility)
- `VLLM_CPU_KV_CACHE_SPACE=4` - Set CPU cache size
- `VLLM_ATTENTION_BACKEND=FLASHINFER` - Use Flash Attention on ARM
- `VLLM_USE_TRITON=0` - Disable Triton on ARM

## Profiles

- `silicon` - Apple Silicon optimization profile
- `linux` - Linux x86_64 profile

## Notes

- The vLLM container uses CPU-optimized settings for Apple Silicon compatibility
- API service runs on standard Python image
- Both services can run simultaneously for development