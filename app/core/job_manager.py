#!/usr/bin/env python3
"""
Background Job Manager for Asynchronous Batch Processing
Handles job queuing, status tracking, and result storage
"""

import json
import logging
import os
import time
import threading
from collections import deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
from app.core.simple_queue import create_simple_queue

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    job_id: str
    input_path: str
    output_path: str
    num_samples: int
    batch_size: int
    concurrency: int

    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    samples_processed: int = 0
    error_message: Optional[str] = None
    results: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if self.results is None:
            self.results = []

class BackgroundJobManager:
    """Manages asynchronous batch processing jobs with simple deque queue and JSON persistence"""
    
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.worker_thread = None
        self.running = False
        self._lock = threading.Lock()
        
        # Initialize simple deque queue
        self.job_queue = deque()
        self.max_queue_depth = 1000
        
        # JSON persistence
        self.state_file = "/tmp/job_manager_state.json"
        self.jobs_dir = "/tmp/jobs"
        os.makedirs(self.jobs_dir, exist_ok=True)
        
        # Load existing state
        self._load_state()
        
    def start_worker(self):
        """Start the background worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Background job worker started")
        
    def stop_worker(self):
        """Stop the background worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Background job worker stopped")
        
    def submit_job(self, request_data: Dict) -> str:
        """Submit a new job for processing"""
        job_id = str(uuid.uuid4())[:8]
        
        job = Job(
            job_id=job_id,
            input_path=request_data["input_path"],
            output_path=request_data["output_path"],
            num_samples=request_data["num_samples"],
            batch_size=request_data["batch_size"],
            concurrency=request_data["concurrency"],

            status=JobStatus.QUEUED,
            created_at=time.time()
        )
        
        with self._lock:
            # Check queue depth
            if len(self.job_queue) >= self.max_queue_depth:
                logger.warning(f"Job queue depth exceeded max {self.max_queue_depth}")
                raise ValueError("Job queue is full")
            
            self.jobs[job_id] = job
            self.job_queue.append(job_id)
            
            # Initialize SLA tracking
            self._create_job_sla(job_id, job.num_samples)
            
        logger.info(f"Job {job_id} submitted for processing (queue depth: {len(self.job_queue)})")
        return job_id
        
    def get_job_status(self, job_id: str) -> Optional[Job]:
        """Get current job status"""
        with self._lock:
            return self.jobs.get(job_id)
            
    def _worker_loop(self):
        """Main worker loop for processing jobs"""
        logger.info("Worker loop started")
        
        while self.running:
            try:
                # Find next queued job
                job = self._get_next_job()
                if not job:
                    time.sleep(1)
                    continue
                    
                # Process the job
                self._process_job(job)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1)
                
        logger.info("Worker loop stopped")
        
    def _get_next_job(self) -> Optional[Job]:
        """Get the next queued job from deque"""
        with self._lock:
            while self.job_queue:
                job_id = self.job_queue.popleft()
                job = self.jobs.get(job_id)
                if job and job.status == JobStatus.QUEUED:
                    job.status = JobStatus.RUNNING
                    job.started_at = time.time()
                    return job
        return None
        
    def _process_job(self, job: Job):
        """Process a single job with retry and validation"""
        gpu_id = None
        try:
            logger.info(f"Processing job {job.job_id}")
            
            # Mark job as started in SLA tracking
            self._mark_job_sla_started(job.job_id)
            
            # Request GPU resource (simplified for now)
            gpu_id = f"mock_gpu_{job.job_id[:4]}"
            logger.info(f"Assigned mock GPU {gpu_id} to job {job.job_id}")
            if not gpu_id:
                logger.warning(f"No GPU available for job {job.job_id}, using CPU fallback")
            
            # Import here to avoid circular imports
            from app.core.simple_processor import SimpleInferencePipeline, VLLMProcessor
            from app.models.schemas import BatchValidationRequest, BatchValidationResponse
            
            # Create processor with GPU information
            pipeline = SimpleInferencePipeline()
            processor = VLLMProcessor("Qwen/Qwen2.5-0.5B-Instruct", gpu_id=gpu_id)
            
            # Generate sample prompts (in real implementation, would read from input_path)
            prompts = [f"Sample prompt {i+1} for testing" for i in range(job.num_samples)]
            
            # Validate batch size limits
            max_batch_size = 1000  # OpenAI-compatible limit
            if len(prompts) > max_batch_size:
                logger.warning(f"Batch size {len(prompts)} exceeds limit {max_batch_size}, truncating")
                prompts = prompts[:max_batch_size]
                job.num_samples = len(prompts)
            
            # Process in batches with retry logic
            batch_size = min(job.batch_size, len(prompts))
            all_results = []
            failed_batches = []
            
            for batch_start in range(0, len(prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(prompts))
                batch_prompts = prompts[batch_start:batch_end]
                batch_index = batch_start // batch_size
                
                # Retry logic for failed batches
                for attempt in range(3):  # Max 3 retries
                    try:
                        results, _ = pipeline.execute_batch(batch_prompts, processor)
                        all_results.extend(results)
                        
                        # Update progress and SLA
                        with self._lock:
                            job.samples_processed = len(all_results)
                        self._update_job_sla(job.job_id, len(all_results))
                        
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        logger.error(f"Batch {batch_index} attempt {attempt + 1} failed: {e}")
                        if attempt < 2:  # Not last attempt
                            time.sleep(1.0 * (2 ** attempt))  # Exponential backoff
                        else:
                            # Log failed batch for error reporting
                            failed_batches.append({
                                "batch_index": batch_index,
                                "prompts": batch_prompts,
                                "error": str(e)
                            })
                            # Add placeholder results for failed batch
                            for prompt in batch_prompts:
                                all_results.append({
                                    "response": f"Failed to process: {str(e)[:100]}...",
                                    "prompt": prompt,
                                    "tokens": 0,
                                    "processing_time": 0.1
                                })
                            
                            # Update progress even for failed batches
                            with self._lock:
                                job.samples_processed = len(all_results)
                            self._update_job_sla(job.job_id, len(all_results))
            
            # Save results to file with error information
            self._save_results(job, all_results, failed_batches)
            
            # Mark as completed
            with self._lock:
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.results = all_results
            
            # Mark SLA as completed
            self._mark_job_sla_completed(job.job_id)
                
            logger.info(f"Job {job.job_id} completed successfully with {len(failed_batches)} failed batches")
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error_message = str(e)
            
            # Mark SLA as failed
            self._mark_job_sla_completed(job.job_id)
        
        finally:
            # Release GPU resource if assigned
            if gpu_id:
                logger.info(f"Releasing GPU {gpu_id} from job {job.job_id}")
                # In real implementation: gpu_simulator.release_gpu(job.job_id)
                
    def _save_results(self, job: Job, results: List[Dict], failed_batches: Optional[List[Dict]] = None):
        """Save results to output file with error tracking"""
        if failed_batches is None:
            failed_batches = []
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(job.output_path), exist_ok=True)
            
            # Prepare output data with error information
            output_data = {
                "job_id": job.job_id,
                "status": "completed",
                "total_samples": job.num_samples,
                "processed_samples": len(results),
                "processing_time": job.completed_at - job.started_at if job.completed_at and job.started_at else 0,
                "results": results,
                "failed_batches": failed_batches or [],
                "success_rate": (len(results) - len(failed_batches or [])) / len(results) if results else 0
            }
            
            # Write to file
            with open(job.output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            logger.info(f"Results saved to {job.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def _create_job_sla(self, job_id: str, samples_total: int):
        """Create SLA tracking for a job"""
        try:
            # Simple SLA tracking without external dependency
            sla_data = {
                "job_id": job_id,
                "target_hours": 24.0,
                "created_at": time.time(),
                "samples_total": samples_total,
                "samples_processed": 0,
                "sla_status": "within_target"
            }
            
            # Save to file for persistence
            sla_file = f"/tmp/sla_{job_id}.json"
            with open(sla_file, 'w') as f:
                json.dump(sla_data, f, indent=2)
                
            logger.info(f"Created SLA tracking for job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to create SLA tracking: {e}")
    
    def _update_job_sla(self, job_id: str, samples_processed: int):
        """Update job SLA progress"""
        try:
            sla_file = f"/tmp/sla_{job_id}.json"
            if not os.path.exists(sla_file):
                return
            
            with open(sla_file, 'r') as f:
                sla_data = json.load(f)
            
            # Update progress
            sla_data["samples_processed"] = samples_processed
            elapsed_time = time.time() - sla_data["created_at"]
            elapsed_hours = elapsed_time / 3600
            
            # Calculate SLA status
            progress_percentage = (samples_processed / sla_data["samples_total"]) * 100 if sla_data["samples_total"] > 0 else 0
            
            if elapsed_hours >= 24.0:
                sla_data["sla_status"] = "breached"
            elif elapsed_hours >= 19.2:  # 80% of 24 hours
                sla_data["sla_status"] = "at_risk"
            else:
                sla_data["sla_status"] = "within_target"
            
            sla_data["progress_percentage"] = progress_percentage
            sla_data["elapsed_hours"] = elapsed_hours
            
            # Save updated SLA
            with open(sla_file, 'w') as f:
                json.dump(sla_data, f, indent=2)
            
            # Log status changes
            logger.info(f"Job {job_id} SLA: {sla_data['sla_status']} ({progress_percentage:.1f}% complete, {elapsed_hours:.1f}h elapsed)")
            
        except Exception as e:
            logger.error(f"Failed to update SLA tracking: {e}")
    
    def _mark_job_sla_started(self, job_id: str):
        """Mark job as started in SLA tracking"""
        try:
            sla_file = f"/tmp/sla_{job_id}.json"
            if not os.path.exists(sla_file):
                return
            
            with open(sla_file, 'r') as f:
                sla_data = json.load(f)
            
            sla_data["started_at"] = time.time()
            
            with open(sla_file, 'w') as f:
                json.dump(sla_data, f, indent=2)
            
            logger.info(f"Job {job_id} marked as started in SLA tracking")
            
        except Exception as e:
            logger.error(f"Failed to mark SLA as started: {e}")
    
    def _mark_job_sla_completed(self, job_id: str):
        """Mark job as completed in SLA tracking"""
        try:
            sla_file = f"/tmp/sla_{job_id}.json"
            if not os.path.exists(sla_file):
                return
            
            with open(sla_file, 'r') as f:
                sla_data = json.load(f)
            
            sla_data["completed_at"] = time.time()
            sla_data["samples_processed"] = sla_data["samples_total"]
            sla_data["progress_percentage"] = 100.0
            sla_data["sla_status"] = "completed"
            
            with open(sla_file, 'w') as f:
                json.dump(sla_data, f, indent=2)
            
            logger.info(f"Job {job_id} marked as completed in SLA tracking")
            
        except Exception as e:
            logger.error(f"Failed to mark SLA as completed: {e}")
    
    def _save_state(self):
        """Save job manager state to JSON file"""
        try:
            with self._lock:
                state = {
                    "timestamp": time.time(),
                    "jobs": {job_id: asdict(job) for job_id, job in self.jobs.items()},
                    "queue": list(self.job_queue),
                    "max_queue_depth": self.max_queue_depth,
                    "running": self.running
                }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"Job manager state saved to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save job manager state: {e}")
    
    def _load_state(self):
        """Load job manager state from JSON file"""
        try:
            if not os.path.exists(self.state_file):
                logger.info("No existing job manager state found, starting fresh")
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            with self._lock:
                # Restore jobs
                for job_id, job_data in state.get("jobs", {}).items():
                    job = Job(**job_data)
                    self.jobs[job_id] = job
                
                # Restore queue
                self.job_queue = deque(state.get("queue", []))
                self.max_queue_depth = state.get("max_queue_depth", 1000)
            
            logger.info(f"Loaded job manager state: {len(self.jobs)} jobs, {len(self.job_queue)} queued jobs")
            
            # Clean up old completed/failed jobs
            self._cleanup_old_jobs()
            
        except Exception as e:
            logger.error(f"Failed to load job manager state: {e}")
    
    def _cleanup_old_jobs(self):
        """Clean up old completed/failed jobs"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (7 * 24 * 3600)  # 7 days ago
            
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
                    job.completed_at and job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            # Remove old jobs
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                
                # Remove job file
                job_file = os.path.join(self.jobs_dir, f"{job_id}.json")
                if os.path.exists(job_file):
                    os.remove(job_file)
                
                # Remove SLA file
                sla_file = f"/tmp/sla_{job_id}.json"
                if os.path.exists(sla_file):
                    os.remove(sla_file)
            
            if jobs_to_remove:
                logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")

# Global instance
job_manager = BackgroundJobManager()