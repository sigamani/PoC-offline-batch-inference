#!/bin/bash

# Create comprehensive inference monitoring dashboard
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