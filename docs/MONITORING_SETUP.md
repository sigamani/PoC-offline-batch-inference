# GPU Monitoring Setup Complete

## Services Running

### Grafana
- **URL**: http://localhost:3000
- **Login**: admin / admin123
- **Dashboard**: NVIDIA GPU Monitoring

### Prometheus
- **URL**: http://localhost:9090
- **Targets**: 
  - Prometheus (localhost:9090)
  - Node Exporter (node-exporter:9100)
  - DCGM Exporter (dcgm-exporter:9400)

### Exporters
- **Node Exporter**: System metrics (port 9100)
- **DCGM Exporter**: NVIDIA GPU metrics (port 9400)

## Available GPU Metrics

The DCGM exporter provides these key metrics:

- `DCGM_FI_DEV_GPU_UTIL` - GPU utilization (%)
- `DCGM_FI_DEV_GPU_TEMP` - GPU temperature (°C)
- `DCGM_FI_DEV_POWER_USAGE` - Power consumption (W)
- `DCGM_FI_DEV_MEMORY_USED` - Memory used (MB)
- `DCGM_FI_DEV_MEMORY_TOTAL` - Total memory (MB)
- `DCGM_FI_DEV_SM_CLOCK` - SM clock speed (MHz)
- `DCGM_FI_DEV_MEMORY_CLOCK` - Memory clock speed (MHz)

## Monitoring Your Inference Workloads

To monitor your batch inference:

1. **Run your inference container**:
   ```bash
   docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
     -v /root/.cache/huggingface:/root/.cache/huggingface \
     michaelsigamani/proj-grounded-telescopes:0.1.0 \
     python your_inference_script.py
   ```

2. **Watch metrics in Grafana**:
   - Open http://localhost:3000
   - Navigate to "NVIDIA GPU Monitoring" dashboard
   - Monitor GPU utilization, temperature, and memory in real-time

3. **Query metrics in Prometheus**:
   - Open http://localhost:9090
   - Example queries:
     - `DCGM_FI_DEV_GPU_UTIL` - Current GPU utilization
     - `rate(DCGM_FI_DEV_GPU_UTIL[5m])` - 5-minute average utilization
     - `DCGM_FI_DEV_MEMORY_USED / DCGM_FI_DEV_MEMORY_TOTAL * 100` - Memory usage percentage

## Container Management

Start monitoring stack:
```bash
docker compose -f /root/workspace/docker-compose-monitoring.yml up -d
```

Stop monitoring stack:
```bash
docker compose -f /root/workspace/docker-compose-monitoring.yml down
```

Check status:
```bash
docker ps | grep -E "(prometheus|grafana|node-exporter|dcgm)"
```

## Performance Monitoring Tips

1. **GPU Utilization**: Should be high (>80%) during inference
2. **Temperature**: Keep below 80°C for optimal performance
3. **Memory Usage**: Monitor for memory leaks in long-running jobs
4. **Power Consumption**: Track efficiency metrics

The monitoring is now ready for your batch inference workloads!