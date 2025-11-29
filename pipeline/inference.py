import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from pipeline.models import InferenceResult


def generate_mock_response(prompt: str, is_dev: bool) -> str:
    if is_dev:
        return _generate_dev_response(prompt)
    return f"STAGE-Qwen2.5-0.5B: {prompt[:50]}..."


def _generate_dev_response(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "what is" in prompt_lower:
        return f"DEV-Qwen2.5-0.5B: {prompt.replace('What is', 'This is')}."
    if "explain" in prompt_lower:
        return f"DEV-Qwen2.5-0.5B: {prompt.replace('Explain', 'Let me explain')}."
    if "hello" in prompt_lower:
        return "DEV-Qwen2.5-0.5B: Hello! How can I help you today?"
    return f"DEV-Qwen2.5-0.5B: I understand you're asking about '{prompt[:30]}...'"


def create_mock_result(prompt: str, is_dev: bool) -> InferenceResult:
    response = generate_mock_response(prompt, is_dev)
    return InferenceResult(
        prompt=prompt,
        response=response,
        tokens=len(response.split()),
        processing_time=0.001
    )