import time
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CostMetrics:
    gpu_cost_per_hour: float = 2.0
    cpu_cost_per_hour: float = 0.1
    memory_cost_per_gb: float = 0.05
    storage_cost_per_gb: float = 0.01

@dataclass
class DerivedMetrics:
    tokens_per_dollar: float = 0.0
    requests_per_dollar: float = 0.0
    hourly_cost: float = 0.0

class DerivedMetricsCalculator:
    def __init__(self, cost_metrics: CostMetrics = None):
        self.cost_metrics = cost_metrics or CostMetrics()

    def calculate(self, raw_metrics: Dict[str, Any], context: Dict[str, Any] = None) -> DerivedMetrics:
        context = context or {}
        derived = DerivedMetrics()
        derived.hourly_cost = self._calculate_hourly_cost(context)
        
        tokens_per_hour = raw_metrics.get("tokens_per_sec", 0) * 3600
        requests_per_hour = raw_metrics.get("throughput_req_per_sec", 0) * 3600
        
        derived.tokens_per_dollar = tokens_per_hour / derived.hourly_cost if derived.hourly_cost else 0
        derived.requests_per_dollar = requests_per_hour / derived.hourly_cost if derived.hourly_cost else 0
        
        return derived

    def _calculate_hourly_cost(self, context: Dict[str, Any]) -> float:
        gpu_count = context.get("gpu_count", 1)
        cpu_count = context.get("cpu_count", 4)
        memory_gb = context.get("memory_gb", 16)
        storage_gb = context.get("storage_gb", 100)

        return (
            gpu_count * self.cost_metrics.gpu_cost_per_hour +
            cpu_count * self.cost_metrics.cpu_cost_per_hour +
            memory_gb * self.cost_metrics.memory_cost_per_gb +
            storage_gb * self.cost_metrics.storage_cost_per_gb / (30 * 24)
        )

# Usage example
if __name__ == "__main__":
    calculator = DerivedMetricsCalculator()
    metrics = calculator.calculate({"tokens_per_sec": 100, "throughput_req_per_sec": 5})
    print(metrics)
