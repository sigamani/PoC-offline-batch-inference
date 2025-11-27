import logging
from typing import Any, Dict, List
from dataclasses import dataclass

from app.core.config import get_config
from app.core.context import get_context, set_context
from app.core.inference import BatchMetrics, InferenceMonitor, create_data_artifact, store_artifact

logger = logging.getLogger(__name__)

@dataclass
class VLLMProcessorConfig:
    """vLLM processor configuration builder"""
    model_name: str
    batch_size: int = 128
    concurrency: int = 2
    max_tokens: int = 512
    temperature: float = 0.7
    enable_chunked_prefill: bool = True
    chunked_prefill_size: int = 8192
    enable_speculative_decoding: bool = True
    num_speculative_tokens: int = 5
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 2
    max_model_len: int = 32768

class VLLMProcessorFactory:
    """Factory for creating vLLM processors with proper configuration"""
    
    def __init__(self):
        self.config = get_config()
    
    def _build_fallback_processor(self, processor_config: VLLMProcessorConfig, metrics: BatchMetrics, monitor: InferenceMonitor):
        """Build fallback processor when Ray Data LLM is not available"""
        def fallback_processor(ds):
            """Simple fallback processor for testing"""
            results = []
            for batch in ds.iter_batches(batch_size=processor_config.batch_size):
                batch_results = []
                for row in batch:
                    # Simple mock inference
                    response = f"Mock response for: {row.get('prompt', '')[:50]}..."
                    tokens = len(response.split())
                    
                    monitor.update(batch_size=1, tokens=tokens)
                    
                    batch_results.append({
                        "response": response,
                        "prompt": row.get("prompt", ""),
                        "tokens": tokens,
                        "processing_time": 0.001
                    })
                results.extend(batch_results)
            
            # Return as a simple dataset-like object
            class SimpleDataset:
                def __init__(self, data):
                    self.data = data
                def take_all(self):
                    return self.data
            return SimpleDataset(results)
        
        return fallback_processor
    
    def create_processor_config(self, processor_config: VLLMProcessorConfig) -> Dict[str, Any]:
        """Create vLLM engine processor configuration"""
        return {
            "model_source": processor_config.model_name,
            "concurrency": processor_config.concurrency,
            "batch_size": processor_config.batch_size,
            "engine_kwargs": {
                "max_num_batched_tokens": processor_config.max_num_batched_tokens,
                "max_model_len": processor_config.max_model_len,
                "gpu_memory_utilization": processor_config.gpu_memory_utilization,
                "tensor_parallel_size": processor_config.tensor_parallel_size,
                "enable_chunked_prefill": processor_config.enable_chunked_prefill,
                "chunked_prefill_size": processor_config.chunked_prefill_size,
                "enable_speculative_decoding": processor_config.enable_speculative_decoding,
                "num_speculative_tokens": processor_config.num_speculative_tokens,
                "speculative_draft_tensor_parallel_size": 1,
                "trust_remote_code": True,
            }
        }
    
    def create_preprocess_fn(self, config):
        """Create preprocessing function"""
        def preprocess(row: Dict) -> Dict:
            return {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": row.get("prompt", "")}
                ],
                "sampling_params": {
                    "temperature": config.inference.temperature,
                    "max_tokens": config.inference.max_tokens,
                }
            }
        return preprocess
    
    def create_postprocess_fn(self, metrics: BatchMetrics, monitor: InferenceMonitor):
        """Create postprocessing function with metrics tracking"""
        def postprocess(row: Dict) -> Dict:
            # Estimate tokens
            generated_text = row.get("generated_text", "")
            tokens = len(generated_text.split()) * 1.3
            
            # Update metrics
            monitor.update(batch_size=1, tokens=int(tokens))
            
            return {
                "response": generated_text,
                "prompt": row.get("prompt", ""),
                "tokens": int(tokens),
                "processing_time": 0.001
            }
        return postprocess
    
    def build_processor(self, processor_config: VLLMProcessorConfig, metrics: BatchMetrics, monitor: InferenceMonitor):
        """Build complete vLLM processor"""
        try:
            from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
            
            vllm_config = self.create_processor_config(processor_config)
            preprocess_fn = self.create_preprocess_fn(self.config)
            postprocess_fn = self.create_postprocess_fn(metrics, monitor)
            
            logger.info(f"Building vLLM processor for {processor_config.model_name}")
            
            return build_llm_processor(
                vllm_config,
                preprocess=preprocess_fn,
                postprocess=postprocess_fn
            )
        except ImportError:
            logger.warning("Ray Data LLM module not available, using fallback processor")
            return self._build_fallback_processor(processor_config, metrics, monitor)

class InferencePipeline:
    """High-level inference pipeline with artifact management"""
    
    def __init__(self, processor_factory: VLLMProcessorFactory):
        self.factory = processor_factory
    
    def execute_batch(self, prompts: List[str], processor_config: VLLMProcessorConfig) -> List[Dict]:
        """Execute batch inference with full pipeline"""
        try:
            import ray
            import time
            
            # Create metrics and monitor
            from app.core.inference import BatchMetrics, InferenceMonitor, SLATier
            
            config = get_config()
            sla_tier = getattr(SLATier, config.sla.tier.upper(), SLATier.BASIC)
            
            metrics = BatchMetrics(total_requests=len(prompts))
            monitor = InferenceMonitor(metrics, sla_tier)
            
            # Create dataset
            ds = ray.data.from_items([{"prompt": prompt} for prompt in prompts])
            logger.info(f"Created dataset with {ds.count()} samples")
            
            # Build processor
            processor = self.factory.build_processor(processor_config, metrics, monitor)
            
            # Execute inference using map_batches
            logger.info("Starting batch inference...")
            start_time = time.time()
            
            result_ds = ds.map_batches(
                processor,
                batch_size=processor_config.batch_size,
                concurrency=processor_config.concurrency
            )
            results = result_ds.take_all()
            
            inference_time = time.time() - start_time
            logger.info(f"Batch inference completed in {inference_time:.2f} seconds")
            
            # Create and store artifact
            artifact = create_data_artifact(results)
            storage_path = config.storage.local_path
            store_artifact(artifact, storage_path)
            
            return results
            
        except ImportError:
            logger.warning("Ray not available, using fallback execution")
            return self._execute_fallback(prompts, processor_config)
    
    def _execute_fallback(self, prompts: List[str], processor_config: VLLMProcessorConfig) -> List[Dict]:
        """Fallback execution without Ray"""
        import time
        from app.core.inference import BatchMetrics, InferenceMonitor, SLATier
        
        config = get_config()
        sla_tier = getattr(SLATier, config.sla.tier.upper(), SLATier.BASIC)
        
        metrics = BatchMetrics(total_requests=len(prompts))
        monitor = InferenceMonitor(metrics, sla_tier)
        
        # Simple mock processing
        results = []
        start_time = time.time()
        
        for prompt in prompts:
            response = f"Mock response for: {prompt[:50]}..."
            tokens = len(response.split())
            
            monitor.update(batch_size=1, tokens=tokens)
            
            results.append({
                "response": response,
                "prompt": prompt,
                "tokens": tokens,
                "processing_time": 0.001
            })
        
        inference_time = time.time() - start_time
        logger.info(f"Fallback batch inference completed in {inference_time:.2f} seconds")
        
        # Create and store artifact
        artifact = create_data_artifact(results)
        storage_path = config.storage.local_path
        store_artifact(artifact, storage_path)
        
        return results