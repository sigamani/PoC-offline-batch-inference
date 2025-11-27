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
        def fallback_processor(row):
            """Simple fallback processor for testing - Ray Data passes individual rows"""
            # Handle individual row format from Ray Data
            prompt = row.get("prompt", "")
            
            # Simple mock inference
            response = f"Mock response for: {str(prompt)[:50]}..."
            tokens = len(response.split())
            
            monitor.update(batch_size=1, tokens=tokens)
            
            return {
                "response": response,
                "prompt": str(prompt),
                "tokens": tokens,
                "processing_time": 0.001
            }
        
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
        """Create preprocessing function for Ray Data"""
        def preprocess(batch: Dict) -> Dict:
            # Ray Data build_llm_processor passes batches as dicts with arrays
            prompts = batch.get("prompt", [])
            logger.info(f"[PREPROCESS] Processing batch of {len(prompts)} prompts")
            
            # For Instruct models, use messages format (OpenAI chat format)
            model_name = config.model.name
            if "Instruct" in model_name:
                # Use OpenAI chat format for vLLM
                messages_list = []
                for prompt in prompts:
                    messages_list.append([
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": str(prompt)}
                    ])
                
                result = {
                    "messages": messages_list,
                    "sampling_params": {
                        "temperature": config.inference.temperature,
                        "max_tokens": config.inference.max_tokens,
                    }
                }
            else:
                # Raw prompt for base models
                result = {
                    "prompt": [str(p) for p in prompts],
                    "sampling_params": {
                        "temperature": config.inference.temperature,
                        "max_tokens": config.inference.max_tokens,
                    }
                }
            
            logger.info(f"[PREPROCESS] Output keys: {result.keys()}")
            return result
        return preprocess
    
    def create_postprocess_fn(self, metrics: BatchMetrics, monitor: InferenceMonitor):
        """Create postprocessing function with metrics tracking"""
        def postprocess(batch: Dict) -> Dict:
            # Debug logging to understand vLLM output format
            logger.info(f"[POSTPROCESS] Input keys: {batch.keys()}")
            
            # vLLM returns generated_text field as array
            generated_texts = batch.get("generated_text", [])
            original_prompts = batch.get("prompt", [])
            
            results = []
            total_tokens = 0
            
            for i, generated_text in enumerate(generated_texts):
                # Estimate tokens more accurately
                tokens = len(str(generated_text).split()) if generated_text else 0
                total_tokens += tokens
                
                result = {
                    "response": generated_text,
                    "prompt": original_prompts[i] if i < len(original_prompts) else "",
                    "tokens": tokens,
                    "processing_time": 0.001
                }
                results.append(result)
            
            # Update metrics for the entire batch
            monitor.update(batch_size=len(results), tokens=total_tokens)
            
            logger.info(f"[POSTPROCESS] Processed batch of {len(results)} results")
            # Convert list of results back to batch format for Ray Data
            return {
                "response": [r["response"] for r in results],
                "prompt": [r["prompt"] for r in results],
                "tokens": [r["tokens"] for r in results],
                "processing_time": [r["processing_time"] for r in results]
            }
        return postprocess
    
    def build_processor(self, processor_config: VLLMProcessorConfig, metrics: BatchMetrics, monitor: InferenceMonitor):
        """Build complete vLLM processor using mandatory Ray Data LLM APIs"""
        try:
            from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
            
            # Create vLLM engine processor configuration
            vllm_engine_config = vLLMEngineProcessorConfig(
                model_source=processor_config.model_name,
                concurrency=processor_config.concurrency,
                batch_size=processor_config.batch_size,
                engine_kwargs={
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
                    "device": "cpu",  # Force CPU mode
                }
            )
            
            preprocess_fn = self.create_preprocess_fn(self.config)
            postprocess_fn = self.create_postprocess_fn(metrics, monitor)
            
            logger.info(f"Building vLLM processor for {processor_config.model_name}")
            
            # Use mandatory build_llm_processor API
            return build_llm_processor(
                vllm_engine_config,
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
        """Execute batch inference with full pipeline using mandatory Ray Data APIs"""
        try:
            import ray
            import time
            
            # Create metrics and monitor
            from app.core.inference import BatchMetrics, InferenceMonitor
            
            config = get_config()
            metrics = BatchMetrics(total_requests=len(prompts))
            monitor = InferenceMonitor(metrics)
            
            # Create Ray dataset from prompts
            ds = ray.data.from_items([{"prompt": prompt} for prompt in prompts])
            logger.info(f"Created Ray dataset with {ds.count()} samples")
            
            # Build processor using mandatory Ray Data LLM APIs
            processor = self.factory.build_processor(processor_config, metrics, monitor)
            
            # Execute inference using the processor directly
            logger.info("Starting batch inference with Ray Data LLM processor")
            start_time = time.time()
            
            # The build_llm_processor already handles map_batches internally
            result_ds = processor(ds)
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
        from app.core.inference import BatchMetrics, InferenceMonitor
        
        config = get_config()
        
        metrics = BatchMetrics(total_requests=len(prompts))
        monitor = InferenceMonitor(metrics)
        
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