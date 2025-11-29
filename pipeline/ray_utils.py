import ray
from ray.data import Dataset
from typing import List, Dict, Any
import pandas as pd
from .inference import create_mock_result


def create_dataset(prompts: List[str]) -> Dataset:
    return ray.data.from_items([{"prompt": p} for p in prompts])


def extract_prompts_from_batch(batch: Dict[str, Any]) -> List[str]:
    if isinstance(batch, dict) and 'prompt' in batch:
        return list(batch['prompt'])
    return [str(item) for item in batch]


def process_batch_with_mock(batch: Dict[str, Any], is_dev: bool) -> pd.DataFrame:
    prompts = extract_prompts_from_batch(batch)
    results = [create_mock_result(p, is_dev) for p in prompts]
    return pd.DataFrame([r.to_dict() for r in results])


def collect_results_from_batches(batches: List[Any]) -> List[Dict[str, Any]]:
    results = []
    for batch in batches:
        results.extend(_extract_batch_results(batch))
    return results


def _extract_batch_results(batch: Any) -> List[Dict[str, Any]]:
    if hasattr(batch, 'to_dict'):
        return batch.to_dict('records')
    if isinstance(batch, list):
        return batch
    if isinstance(batch, dict) and _is_single_result(batch):
        return [batch]
    if isinstance(batch, dict):
        return list(batch.values())
    return [batch]


def _is_single_result(batch: Dict[str, Any]) -> bool:
    return 'prompt' in batch and 'response' in batch