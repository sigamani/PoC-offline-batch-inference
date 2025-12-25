"""Test mock GPU scheduling simulation logic"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from api.gpu_scheduler import MockGPUScheduler, PoolType
from api.models import priorityLevels


class TestMockGPUScheduler:
    def setup_method(self):
        self.scheduler = MockGPUScheduler(spot_capacity=2, dedicated_capacity=1)

    def test_initial_state(self):
        status = self.scheduler.get_pool_status()
        assert status["spot"]["capacity"] == 2
        assert status["spot"]["available"] == 2
        assert status["dedicated"]["capacity"] == 1
        assert status["dedicated"]["available"] == 1

    def test_basic_spot_allocation(self):
        result = self.scheduler.allocate_gpu("job1", priority_level=priorityLevels.LOW)

        assert result.allocated == True
        assert result.pool_type == PoolType.SPOT
        assert result.cost_estimate == 0.10
        assert "Spot instance available" in result.reason

        status = self.scheduler.get_pool_status()
        assert status["spot"]["available"] == 1

    def test_high_priority_gets_dedicated(self):
        result = self.scheduler.allocate_gpu("job1", priority_level=priorityLevels.HIGH)

        assert result.allocated == True
        assert result.pool_type == PoolType.DEDICATED
        assert "dedicated instance" in result.reason.lower()

    def test_resource_exhaustion(self):
        results = [
            self.scheduler.allocate_gpu("job1", priority_level=priorityLevels.LOW),
            self.scheduler.allocate_gpu("job2", priority_level=priorityLevels.LOW),
            self.scheduler.allocate_gpu("job3", priority_level=priorityLevels.LOW),
        ]

        for i, result in enumerate(results):
            assert result.allocated == True, f"Job {i + 1} should be allocated"

        result = self.scheduler.allocate_gpu("job4", priority_level=priorityLevels.LOW)
        assert result.allocated == False
        assert result.queue_position == 1
        assert "queued at position" in result.reason

    def test_priority_queue_ordering(self):
        self.scheduler.allocate_gpu("job1", priority_level=priorityLevels.LOW)
        self.scheduler.allocate_gpu("job2", priority_level=priorityLevels.LOW)
        self.scheduler.allocate_gpu("job3", priority_level=priorityLevels.LOW)

        result1 = self.scheduler.allocate_gpu(
            "normal_job", priority_level=priorityLevels.LOW
        )
        assert result1.queue_position == 1

        result2 = self.scheduler.allocate_gpu(
            "priority_job", priority_level=priorityLevels.HIGH
        )
        assert result2.queue_position == 1

        assert len(self.scheduler.waiting_queue) == 2
        assert self.scheduler.waiting_queue[0] == "priority_job"

    def test_release_and_reallocate(self):
        job1_id = "job1"
        self.scheduler.allocate_gpu(job1_id, priority_level=priorityLevels.LOW)

        self.scheduler.release_gpu(job1_id)

        status = self.scheduler.get_pool_status()
        assert status["spot"]["available"] == 2

        result = self.scheduler.allocate_gpu("job4", 1)
        assert result.allocated == True

    def test_metrics_tracking(self):
        self.scheduler.allocate_gpu("job1", priority_level=priorityLevels.LOW)
        self.scheduler.allocate_gpu("job2", priority_level=priorityLevels.HIGH)

        metrics = self.scheduler.get_metrics()
        assert metrics["total_allocations"] == 2
        assert metrics["total_capacity"] == 3
        assert 0 <= metrics["utilization_rate"] <= 1


class TestStressScenarios:
    def setup_method(self):
        self.scheduler = MockGPUScheduler(spot_capacity=3, dedicated_capacity=2)

    def test_burst_load(self):
        job_ids = []
        results = []

        for i in range(10):
            job_id = f"burst_job_{i}"
            result = self.scheduler.allocate_gpu(
                job_id, priority_level=priorityLevels.LOW
            )
            job_ids.append(job_id)
            results.append(result)

        allocated_results = [r for r in results[:5] if r.allocated]
        assert len(allocated_results) == 5

        queued_results = [r for r in results[5:] if not r.allocated]
        assert len(queued_results) == 5
        assert len(self.scheduler.waiting_queue) == 5

    def test_mixed_priority_load(self):
        normal_jobs = []
        high_jobs = []
        normal_results = []
        high_results = []

        for i in range(8):
            if i % 3 == 0:
                job_id = f"high_job_{i}"
                result = self.scheduler.allocate_gpu(
                    job_id, priority_level=priorityLevels.HIGH
                )
                high_jobs.append(job_id)
                high_results.append(result)
            else:
                job_id = f"normal_job_{i}"
                result = self.scheduler.allocate_gpu(
                    job_id, priority_level=priorityLevels.LOW
                )
                normal_jobs.append(job_id)
                normal_results.append(result)

        total_allocated = sum(1 for r in high_results + normal_results if r.allocated)
        total_queued = sum(1 for r in high_results + normal_results if not r.allocated)

        assert total_allocated == 5, f"Expected 5 allocated, got {total_allocated}"
        assert total_queued == 3, f"Expected 3 queued, got {total_queued}"

        queued_results = [r for r in high_results + normal_results if not r.allocated]
        queue_positions = [r.queue_position for r in queued_results]

        assert queue_positions[0] == 1, "First queued should be position 1"
