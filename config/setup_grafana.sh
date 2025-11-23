#!/bin/bash

# Wait for Grafana to be ready
echo "Waiting for Grafana to start..."
until curl -s http://localhost:3000/api/health | grep -q "ok"; do
    sleep 2
done

echo "Grafana is ready. Setting up datasources and dashboards..."

# Add Prometheus datasource
curl -X POST \
  http://localhost:3000/api/datasources \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Basic YWRtaW46YWRtaW4xMjM=' \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy",
    "isDefault": true
  }'

echo "Datasource added. Importing GPU dashboard..."

# Import NVIDIA GPU dashboard
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

echo "Setup complete!"
echo "Access Grafana at: http://localhost:3000"
echo "Login with: admin / admin123"
echo "Access Prometheus at: http://localhost:9090"