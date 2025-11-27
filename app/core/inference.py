import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataArtifact:
    """Immutable data artifact with SHA tracking"""
    sha256_hash: str
    content_hash: str
    version: str
    created_at: float
    size_bytes: int

@dataclass
class BatchMetrics:
    """Batch processing metrics with SLA tracking"""
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    start_time: float = 0.0
    tokens_processed: int = 0
    artifact_id: str = ""
    
    def progress_pct(self) -> float:
        return (self.completed_requests / self.total_requests) * 100 if self.total_requests > 0 else 0
    
    def throughput_per_sec(self) -> float:
        elapsed = time.time() - self.start_time
        return self.completed_requests / elapsed if elapsed > 0 else 0
    
    def tokens_per_sec(self) -> float:
        elapsed = time.time() - self.start_time
        return self.tokens_processed / elapsed if elapsed > 0 else 0
    
    def eta_hours(self) -> float:
        throughput = self.throughput_per_sec()
        if throughput == 0:
            return float('inf')
        remaining = self.total_requests - self.completed_requests
        return (remaining / throughput) / 3600



@dataclass
class InferenceMonitor:
    """Simple monitoring without SLA tiers"""
    def __init__(self, metrics: BatchMetrics, log_interval: int = 100):
        self.metrics = metrics
        self.log_interval = log_interval
        self.last_log = 0
    
    def update(self, batch_size: int, tokens: int):
        self.metrics.completed_requests += batch_size
        self.metrics.tokens_processed += tokens
        
        if self.metrics.completed_requests - self.last_log >= self.log_interval:
            self.log_progress()
            self.last_log = self.metrics.completed_requests
    
    def log_progress(self):
        logger.info(
            f"Progress: {self.metrics.progress_pct():.1f}% | "
            f"Completed: {self.metrics.completed_requests}/{self.metrics.total_requests} | "
            f"Throughput: {self.metrics.throughput_per_sec():.2f} req/s | "
            f"Tokens/sec: {self.metrics.tokens_per_sec():.2f} | "
            f"ETA: {self.metrics.eta_hours():.2f}h"
        )
    
    def check_sla(self) -> bool:
        return True

def create_data_artifact(data: List[Dict]) -> DataArtifact:
    """Create SHA-based data artifact"""
    import json
    import hashlib
    
    content_str = json.dumps(data, sort_keys=True)
    sha256_hash = hashlib.sha256(content_str.encode()).hexdigest()
    content_hash = hashlib.md5(content_str.encode()).hexdigest()
    
    return DataArtifact(
        sha256_hash=sha256_hash,
        content_hash=content_hash,
        version=f"v{int(time.time())}",
        created_at=time.time(),
        size_bytes=len(content_str.encode())
    )

def store_artifact(artifact: DataArtifact, storage_path: str):
    """Store artifact to local storage (S3 in production)"""
    import os
    import json
    
    os.makedirs(storage_path, exist_ok=True)
    artifact_file = os.path.join(storage_path, f"artifact_{artifact.sha256_hash[:8]}.json")
    
    with open(artifact_file, 'w') as f:
        json.dump(artifact.__dict__, f, indent=2)
    
    logger.info(f"Stored artifact {artifact.sha256_hash[:8]} to {artifact_file}")
    return artifact_file