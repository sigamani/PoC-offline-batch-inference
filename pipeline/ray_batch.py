""" Ray Batch Processor for handling batch inference with vLLM or mock processing. """

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
import logging
import time
from typing import List, Dict, Any
from config import EnvironmentConfig, ModelConfig
logger = logging.getLogger(__name__)

from config import EnvironmentConfig, ModelConfig
from pipeline.inference import create_dataset, create_mock_result

class RayBatchProcessor:
    def __init__(self, model_config: ModelConfig, env_config: EnvironmentConfig):
        self.model_config = model_config
        self.env_config = env_config
        self.vllm_engine = None
        
        if env_config.is_gpu_available and not env_config.is_dev:
            self._init_vllm_engine()
            logger.info("STAGE: Using real vLLM engine")
        else:
            self.processor = None
            logger.info("DEV: Using mock Ray Data processor")
    
    def _init_vllm_engine(self):
        try:
            import requests
            
            # Use vLLM HTTP API instead of direct import
            self.vllm_api_url = "http://vllm:8001/v1/completions"
            self.vllm_engine = True  # Flag to indicate HTTP mode
            
            logger.info(f"vLLM HTTP client initialized for API: {self.vllm_api_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM HTTP client: {e}")
            raise
    
    def process_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            if hasattr(self, 'vllm_api_url'):
                results = self._execute_vllm_batch(prompts)
            else:
                results = self._execute_batch_processing(prompts)
            self._log_completion(start_time)
            return results
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return self._fallback_process(prompts)
    
    def _execute_vllm_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Execute batch using real vLLM HTTP API"""
        logger.info(f"Processing {len(prompts)} prompts with vLLM HTTP API")
        
        results = []
        import requests
        import time
        
        for i, prompt in enumerate(prompts):
            start_time = time.time()
            try:
                response = requests.post(
                    self.vllm_api_url,
                    json={
                        "model": self.model_config.model_name,
                        "prompt": prompt,
                        "max_tokens": self.model_config.max_tokens,
                        "temperature": self.model_config.temperature
                    },
                    timeout=10,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data["choices"][0]["text"]
                    tokens = len(data["choices"][0].get("logprobs", {}).get("token_ids", []))
                else:
                    response_text = f"Error: HTTP {response.status_code}"
                    tokens = 0
                
                processing_time = time.time() - start_time
                
                results.append({
                    "prompt": prompt,
                    "response": response_text,
                    "tokens": tokens,
                    "processing_time": processing_time
                })
                
            except Exception as e:
                logger.error(f"Failed to process prompt {i}: {e}")
                results.append({
                    "prompt": prompt,
                    "response": f"Error: {str(e)}",
                    "tokens": 0,
                    "processing_time": 0.001
                })
        
        return results
    
    def _execute_batch_processing(self, prompts: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"Processing {len(prompts)} prompts with mock inference")
        # Skip Ray Data for now and use direct mock processing
        return self._fallback_process(prompts)
    
    def _process_with_mock(self, ds) -> List[Dict[str, Any]]:
        logger.info("Using Ray Data map_batches with mock inference")
        def process_batch_with_mock(batch, is_dev):
            results = []
            if isinstance(batch, dict) and 'prompt' in batch:
                prompts = batch['prompt']
                if hasattr(prompts, '__iter__') and not isinstance(prompts, str):
                    for prompt in prompts:
                        result = create_mock_result(str(prompt), is_dev)
                        results.append(result.to_dict())
                else:
                    result = create_mock_result(str(prompts), is_dev)
                    results.append(result.to_dict())
            else:
                for item in batch:
                    if hasattr(item, 'get'):
                        prompt = item.get('prompt', str(item))
                    else:
                        prompt = str(item)
                    result = create_mock_result(prompt, is_dev)
                    results.append(result.to_dict())
            return {"results": results}
        
        batch_fn = lambda batch: process_batch_with_mock(batch, self.env_config.is_dev)
        processed_ds = ds.map_batches(batch_fn, batch_size=self.model_config.batch_size)
        batches = processed_ds.take_all()
        
        all_results = []
        for batch in batches:
            if isinstance(batch, dict) and 'results' in batch:
                all_results.extend(batch['results'])
            elif isinstance(batch, list):
                all_results.extend(batch)
            else:
                all_results.append(batch)
        return all_results
    
    def _fallback_process(self, prompts: List[str]) -> List[Dict[str, Any]]:
        results = [create_mock_result(p, self.env_config.is_dev) for p in prompts]
        return [r.to_dict() for r in results]
    
    def _log_completion(self, start_time: float):
        duration = time.time() - start_time
        logger.info(f"Batch processing completed in {duration:.2f} seconds")