""" Data models for batch processing pipeline. """

from dataclasses import dataclass
from pydantic import BaseModel
from typing import Dict, List

@dataclass
class InferenceRequest:
    prompt: str


@dataclass
class InferenceResult:
    prompt: str
    response: str
    tokens: int
    processing_time: float
    
    def to_dict(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "tokens": self.tokens,
            "processing_time": self.processing_time
        }
    
class OpenAIBatchRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    input: List[Dict[str, str]] = None 
    input_file_id: str = None  
    max_tokens: int = 256
    temperature: float = 0.7