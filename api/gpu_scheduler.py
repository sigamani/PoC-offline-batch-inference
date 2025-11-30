"""
Mock GPU Scheduler for testing GPU allocation logic without real GPU resources.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
logger = logging.getLogger(__name__)

class PoolType(Enum):
    SPOT = "spot"
    DEDICATED = "dedicated"

@dataclass
class GPUResource:
    pool_type: PoolType
    capacity: int
    available: int
    cost_per_hour: float

@dataclass
class AllocationResult:
    pool_type: PoolType
    allocated: bool
    reason: str
    cost_estimate: float = 0.0

class MockGPUScheduler:
    def __init__(self):
        self.pools: Dict[PoolType, GPUResource] = {
            PoolType.SPOT: GPUResource(
                pool_type=PoolType.SPOT,
                capacity=2,
                available=1,  
                cost_per_hour=0.10
            ),
            PoolType.DEDICATED: GPUResource(
                pool_type=PoolType.DEDICATED,
                capacity=1,
                available=1,  
                cost_per_hour=0.50
            )
        }
        self.allocations: Dict[str, PoolType] = {} 
        
    def allocate_gpu(self, job_id: str, priority_level: int = 1) -> AllocationResult:
        spot_pool = self.pools[PoolType.SPOT]
        if spot_pool.available > 0:
            spot_pool.available -= 1
            self.allocations[job_id] = PoolType.SPOT
            logger.info(f"Allocated SPOT GPU for job {job_id}")
            return AllocationResult(
                pool_type=PoolType.SPOT,
                allocated=True,
                reason="Spot instance available (cost-effective)",
                cost_estimate=spot_pool.cost_per_hour
            )
        
        dedicated_pool = self.pools[PoolType.DEDICATED]
        if dedicated_pool.available > 0:
            dedicated_pool.available -= 1
            self.allocations[job_id] = PoolType.DEDICATED
            logger.info(f"Allocated DEDICATED GPU for job {job_id} (spot unavailable)")
            return AllocationResult(
                pool_type=PoolType.DEDICATED,
                allocated=True,
                reason="Spot unavailable, using dedicated instance",
                cost_estimate=dedicated_pool.cost_per_hour
            )
        
        logger.warning(f"No GPU resources available for job {job_id}")
        return AllocationResult(
            pool_type=PoolType.SPOT,
            allocated=False,
            reason="No GPU resources available in either pool"
        )
    
    def release_gpu(self, job_id: str) -> None:
        if job_id not in self.allocations:
            logger.warning(f"Job {job_id} not found in allocations")
            return
            
        pool_type = self.allocations.pop(job_id)
        pool = self.pools[pool_type]
        pool.available += 1
        logger.info(f"Released {pool_type.value} GPU for job {job_id}")
    
    def get_pool_status(self) -> Dict:
        return {
            pool_type.value: {
                "capacity": pool.capacity,
                "available": pool.available,
                "utilized": pool.capacity - pool.available,
                "cost_per_hour": pool.cost_per_hour
            }
            for pool_type, pool in self.pools.items()
        }
    
    def get_job_allocation(self, job_id: str) -> Optional[PoolType]:
        return self.allocations.get(job_id)