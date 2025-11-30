"""Testing race condition and thread safety guardrails """

import pytest
import time
import threading
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from api.job_queue import SimpleQueue
from api.gpu_scheduler import MockGPUScheduler
from api.worker import BatchWorker
from api.models import priorityLevels
class TestThreadSafety:
    
    @pytest.fixture
    def scheduler(self):
        return MockGPUScheduler(spot_capacity=2, dedicated_capacity=1)
    
    @pytest.fixture
    def queue(self):
        return SimpleQueue()
    
    @pytest.fixture
    def worker(self, queue):
        import tempfile
        temp_dir = tempfile.mkdtemp()
        worker = BatchWorker(queue, batch_dir=temp_dir)
        yield worker
        worker.stop()
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_queue_concurrent_enqueue(self, queue):
        results = []
        errors = []
        
        def enqueue_job(job_id):
            try:
                msg_id = queue.enqueue({"job_id": job_id}, priorityLevels.LOW)
                results.append(msg_id)
            except Exception as e:
                errors.append(e)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(enqueue_job, f"job_{i}") for i in range(10)]
            for future in as_completed(futures):
                future.result()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert len(set(results)) == 10, "All message IDs should be unique"
        assert queue.get_depth() == 10, f"Queue depth should be 10, got {queue.get_depth()}"
    
    @pytest.fixture
    def queue(self):
        return SimpleQueue()
    
    def test_shared_resource_race(self):
        import tempfile
        import threading
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        temp_file.close()  
        
        results = []
        errors = []
        
        def write_to_file(thread_id):
            try:
                with open(temp_file.name, 'a') as f:
                    f.write(f"Thread {thread_id} data\n")
                    time.sleep(0.01)
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_to_file, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=5)
        
        assert len(errors) == 0, f"File write errors: {errors}"
        assert len(results) == 5, f"Expected 5 completions, got {len(results)}"
        
        os.unlink(temp_file.name)

    def test_queue_concurrent_enqueue_dequeue(self, queue):
        enqueued = []
        dequeued = []
        errors = []
        
        def producer():
            for i in range(5):
                try:
                    msg_id = queue.enqueue({"job_id": f"prod_job_{i}"}, priorityLevels.LOW)
                    enqueued.append(msg_id)
                    time.sleep(0.01)  
                except Exception as e:
                    errors.append(f"Producer error: {e}")
        
        def consumer():
            time.sleep(0.02) 
            for i in range(5):
                try:
                    messages = queue.dequeue(count=1)
                    if messages:
                        dequeued.extend(messages)
                    time.sleep(0.01)
                except Exception as e:
                    errors.append(f"Consumer error: {e}")
        
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        producer_thread.join(timeout=5)
        consumer_thread.join(timeout=5)
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(enqueued) == 5, f"Expected 5 enqueued, got {len(enqueued)}"
        assert len(dequeued) <= 5, f"Expected <=5 dequeued, got {len(dequeued)}"
    
    def test_scheduler_concurrent_allocation(self, scheduler):
        allocation_results = []
        errors = []
        
        def allocate_gpu(job_id, priority):
            try:
                result = scheduler.allocate_gpu(job_id, priority)
                allocation_results.append((job_id, result))
                return result
            except Exception as e:
                errors.append(f"Allocation error for {job_id}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(allocate_gpu, f"job_{i}", priorityLevels.LOW)
                for i in range(6)
            ]
            for future in as_completed(futures):
                future.result()
        
        assert len(errors) == 0, f"Allocation errors: {errors}"
        assert len(allocation_results) == 6, f"Expected 6 allocation attempts, got {len(allocation_results)}"
        
        successful_allocations = sum(1 for _, result in allocation_results if result.allocated)
        assert successful_allocations <= 3, f"Expected <=3 successful allocations, got {successful_allocations}"
        
        final_status = scheduler.get_pool_status()
        total_available = final_status["spot"]["available"] + final_status["dedicated"]["available"]
        total_capacity = final_status["spot"]["capacity"] + final_status["dedicated"]["capacity"]
        assert total_available >= 0, "Available resources should not be negative"
        assert total_available <= total_capacity, "Available should not exceed capacity"
    
    def test_worker_thread_safety(self, worker, queue, scheduler):
        worker.gpu_scheduler = scheduler
        
        job_ids = []
        for i in range(3):
            job_data = {
                "job_id": f"thread_test_job_{i}",
                "input_file": f"/tmp/test_input_{i}.jsonl",
                "output_file": f"/tmp/test_output_{i}.jsonl",
                "error_file": f"/tmp/test_error_{i}.jsonl"
            }
            
            with open(job_data["input_file"], 'w') as f:
                f.write(json.dumps({"prompt": f"Test prompt {i}"}) + "\n")
            
            queue.enqueue({"job_id": job_data["job_id"]}, priorityLevels.LOW)
            job_ids.append(job_data["job_id"])
        
        worker.start()
        time.sleep(0.5)  
        
        worker.stop()
        
        assert worker.running == False, "Worker should be stopped"
        
        for i in range(3):
            for suffix in ["input", "output", "error"]:
                test_file = f"/tmp/test_{suffix}_{i}.jsonl"
                if os.path.exists(test_file):
                    os.remove(test_file)
    
    def test_concurrent_api_calls(self):
        import requests
        import subprocess
        
        env = os.environ.copy()
        env["ENVIRONMENT"] = "DEV"
        env["GPU_AVAILABLE"] = "false"
        
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "api.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8000"
        ], cwd=project_root, env=env)
        
        try:
            time.sleep(3)  
            
            def create_batch(batch_id):
                response = requests.post(
                    "http://127.0.0.1:8000/v1/batches",
                    json={
                        "model": "Qwen/Qwen2.5-0.5B-Instruct",
                        "input": [{"prompt": f"Concurrent test {batch_id}"}],
                        "max_tokens": 50
                    },
                    timeout=10
                )
                return response
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(create_batch, i) for i in range(5)]
                responses = [future.result() for future in as_completed(futures)]
            
            success_count = sum(1 for r in responses if r.status_code == 200)
            assert success_count == 5, f"Expected 5 successful requests, got {success_count}"
            
            batch_ids = [r.json().get("id") for r in responses if r.status_code == 200]
            assert len(set(batch_ids)) == 5, "All batch IDs should be unique"
            
        finally:
            server_process.terminate()
            server_process.wait(timeout=5)

class TestRaceConditions:
    @pytest.fixture
    def queue(self):
        """Fresh queue for each test."""
        return SimpleQueue()
    
    def test_queue_depth_race(self, queue):
        depth_results = []
        
        def check_depth():
            depth_results.append(queue.get_depth())
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=check_depth)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=2)
        
        unique_depths = set(depth_results)
        assert len(unique_depths) <= 2, f"Depth readings too inconsistent: {unique_depths}"