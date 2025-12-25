import time
import os
import logging
from typing import Dict, Any
import ray
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - could be moved to config file or environment variables
CONFIG = {
    "model_name": "facebook/opt-125m",
    "max_model_len": 256,
    "num_workers": 2,
    "batch_size": None,  # Will be calculated based on data size
    "enforce_eager": True,
    "dtype": "float32",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 50,
}


llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="facebook/opt-125m",
        model_source="facebook/opt-125m",
    ),
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=4),
        request_router_config=dict(
            request_router_class="ray.serve.llm.request_router.PrefixCacheAffinityRouter"
        ),
    ),
)


@serve.deployment
class VLLMDeployment:

    def __init__(self, config: Dict[str, Any]):
        worker_id = ray.get_runtime_context().get_worker_id()
        logger.info(f"[Worker {worker_id}] Initializing vLLM...")

    async def __call__(self, request: str, request_id) -> Dict[str, Any]:
        output = self.llm.generate(request_id, self.sampling_params)
        worker_id = ray.get_runtime_context.get_worker_id()
        start_time = time.time()

        return {
            "prompt": output.prompt,
            "response": output.outputs[0].text,
            "tokens": len(output.outputs[0].token_ids),
            "worker_id": worker_id,
            "generation_time": time.time() - start_time,
        }

  

if __name__ == "__main__":
    app = build_openai_app({"llm_configs": [llm_config]})
