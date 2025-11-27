import os
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    """Model configuration with validation"""
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_model_len: int = 32768
    tensor_parallel_size: int = 2
    enable_chunked_prefill: bool = False
    device: str = "cpu"
    trust_remote_code: bool = True

class InferenceConfig(BaseModel):
    """Inference parameters with validation"""
    batch_size: int = 128
    concurrency: int = 2
    temperature: float = 0.7
    max_tokens: int = 512
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.90

class StorageConfig(BaseModel):
    """Storage configuration"""
    local_path: str = "/tmp/artifacts"
    s3_bucket: str = "batch-inference-artifacts"

class SLAConfig(BaseModel):
    """SLA configuration with tier support"""
    target_hours: float = 24.0
    buffer_factor: float = 0.7
    alert_threshold_hours: float = 20.0
    tier: str = "basic"

class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    prometheus_port: int = 8001
    grafana_enabled: bool = False
    loki_enabled: bool = False

class AppConfig(BaseModel):
    """Main application configuration"""
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()
    storage: StorageConfig = StorageConfig()
    sla: SLAConfig = SLAConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load configuration from YAML with validation"""
        import yaml
        
        try:
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Override with environment variables
            return cls(**config_data)
            
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return cls()  # Return defaults

def get_config() -> AppConfig:
    """Get application configuration"""
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    return AppConfig.from_yaml(config_path)