#!/bin/bash

set -e

# Wait for Grafana to be reachable before provisioning assets
echo "Waiting for Grafana to start..."
until curl -s http://localhost:3000/api/health | grep -q "ok"; do
    sleep 2
done

echo "Grafana is ready. Ensuring Prometheus datasource..."

# Create or update the Prometheus datasource so dashboards can query metrics
curl -s -o /dev/null -w "%{http_code}" -X POST \
  http://localhost:3000/api/datasources \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Basic YWRtaW46YWRtaW4xMjM=' \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy",
    "isDefault": true
  }' | grep -E "(200|202)" >/dev/null || echo "Datasource already exists"

echo "Provisioning NVIDIA GPU dashboard..."

# Import GPU monitoring dashboard (legacy functionality retained)
curl -X POST \
  http://localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Basic YWRtaW46YWRtaW4xMjM=' \
  -d '{
    "dashboard": {
      "id": null,
      "title": "NVIDIA GPU Monitoring",
      "tags": ["gpu", "nvidia"],
      "timezone": "browser",
      "panels": [
        {
          "id": 1,
          "title": "GPU Utilization",
          "type": "stat",
          "targets": [
            {
              "expr": "dcgm_gpu_utilization",
              "legendFormat": "GPU {{gpu}}"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "unit": "percent",
              "min": 0,
              "max": 100
            }
          }
        },
        {
          "id": 2,
          "title": "GPU Memory Usage",
          "type": "graph",
          "targets": [
            {
              "expr": "dcgm_memory_used",
              "legendFormat": "Used - GPU {{gpu}}"
            },
            {
              "expr": "dcgm_memory_total",
              "legendFormat": "Total - GPU {{gpu}}"
            }
          ]
        },
        {
          "id": 3,
          "title": "GPU Temperature",
          "type": "graph",
          "targets": [
            {
              "expr": "dcgm_gpu_temperature",
              "legendFormat": "GPU {{gpu}}"
            }
          ]
        },
        {
          "id": 4,
          "title": "GPU Power Usage",
          "type": "graph",
          "targets": [
            {
              "expr": "dcgm_power_usage",
              "legendFormat": "GPU {{gpu}}"
            }
          ]
        }
      ],
      "time": {
        "from": "now-1h",
        "to": "now"
      },
      "refresh": "5s"
    },
    "overwrite": true
  }'

echo "Provisioning LLM inference dashboard..."

# Import comprehensive LLM inference monitoring dashboard
curl -X POST \
  http://localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Basic YWRtaW46YWRtaW4xMjM=' \
  -d '{
    "dashboard": {
      "id": null,
      "title": "LLM Inference Worker Monitoring",
      "tags": ["llm", "inference", "kv-cache"],
      "timezone": "browser",
      "panels": [
        {
          "id": 1,
          "title": "KV Cache Utilization",
          "type": "stat",
          "targets": [
            {
              "expr": "llm_kv_cache_utilization_percent",
              "legendFormat": "{{worker_id}} - {{model_name}}"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "unit": "percent",
              "min": 0,
              "max": 100,
              "thresholds": {
                "steps": [
                  {"color": "green", "value": 0},
                  {"color": "yellow", "value": 70},
                  {"color": "red", "value": 90}
                ]
              }
            }
          },
          "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
        },
        {
          "id": 2,
          "title": "Queue Depth",
          "type": "stat",
          "targets": [
            {
              "expr": "llm_queue_depth",
              "legendFormat": "{{worker_id}}"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "unit": "short",
              "thresholds": {
                "steps": [
                  {"color": "green", "value": 0},
                  {"color": "yellow", "value": 5},
                  {"color": "red", "value": 10}
                ]
              }
            }
          },
          "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
        },
        {
          "id": 3,
          "title": "Active Requests",
          "type": "graph",
          "targets": [
            {
              "expr": "llm_active_requests",
              "legendFormat": "{{worker_id}}"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "unit": "short"
            }
          },
          "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
        },
        {
          "id": 4,
          "title": "Inference Duration",
          "type": "graph",
          "targets": [
            {
              "expr": "histogram_quantile(0.50, rate(llm_inference_duration_seconds_bucket[5m]))",
              "legendFormat": "50th percentile - {{worker_id}}"
            },
            {
              "expr": "histogram_quantile(0.95, rate(llm_inference_duration_seconds_bucket[5m]))",
              "legendFormat": "95th percentile - {{worker_id}}"
            },
            {
              "expr": "histogram_quantile(0.99, rate(llm_inference_duration_seconds_bucket[5m]))",
              "legendFormat": "99th percentile - {{worker_id}}"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "unit": "s"
            }
          },
          "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
        },
        {
          "id": 5,
          "title": "Tokens Generated Rate",
          "type": "graph",
          "targets": [
            {
              "expr": "rate(llm_tokens_generated_total[5m])",
              "legendFormat": "{{worker_id}} - {{model_name}}"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "unit": "tokens/sec"
            }
          },
          "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
        },
        {
          "id": 6,
          "title": "GPU Memory Utilization",
          "type": "graph",
          "targets": [
            {
              "expr": "llm_gpu_memory_utilization_percent",
              "legendFormat": "{{worker_id}} - GPU {{gpu_id}}"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "unit": "percent",
              "min": 0,
              "max": 100
            }
          },
          "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
        },
        {
          "id": 7,
          "title": "Total Tokens Generated",
          "type": "stat",
          "targets": [
            {
              "expr": "sum(llm_tokens_generated_total)",
              "legendFormat": "Total"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "unit": "short"
            }
          },
          "gridPos": {"h": 4, "w": 6, "x": 0, "y": 24}
        },
        {
          "id": 8,
          "title": "Average Inference Time",
          "type": "stat",
          "targets": [
            {
              "expr": "histogram_quantile(0.50, rate(llm_inference_duration_seconds_bucket[5m]))",
              "legendFormat": "Median"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "unit": "s"
            }
          },
          "gridPos": {"h": 4, "w": 6, "x": 6, "y": 24}
        }
      ],
      "time": {
        "from": "now-1h",
        "to": "now"
      },
      "refresh": "5s"
    },
    "overwrite": true
  }'

echo "Grafana provisioning complete!"
echo "GPU dashboard: http://localhost:3000/d/NVIDIA_GPU_Monitoring"
echo "Inference dashboard: http://localhost:3000/d/LLM_Inference_Worker_Monitoring"
echo "Login with: admin / admin123"
