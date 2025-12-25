"""End to end test for API endpoints of Ray Batch processing service with vllm mocked"""

import os
import sys
import json
import time
import requests
import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="module")
def server_url():
    import subprocess

    env = os.environ.copy()
    env["ENVIRONMENT"] = "DEV"
    env["GPU_AVAILABLE"] = "false"

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]

    server_process = subprocess.Popen(
        cmd, cwd=project_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    time.sleep(3)

    yield "http://127.0.0.1:8000"

    server_process.terminate()
    server_process.wait(timeout=5)


def test_batch_creation(server_url):
    # Check initial queue depth
    queue_response = requests.get(f"{server_url}/queue/stats")
    initial_depth = 0
    if queue_response.status_code == 200:
        initial_depth = queue_response.json().get("depth", 0)

    test_data = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "input": [{"prompt": "What is 2+2?"}, {"prompt": "Hello world"}],
        "max_tokens": 50,
    }

    response = requests.post(
        f"{server_url}/v1/batches",
        json=test_data,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    result = response.json()

    assert "id" in result
    assert result["object"] == "batch"
    assert result["status"] == "queued"
    assert result["total_prompts"] == 2
    assert result["model"] == "Qwen/Qwen2.5-0.5B-Instruct"

    batch_id = result["id"]

    job_file = f"/tmp/job_{batch_id}.json"
    assert os.path.exists(job_file), f"Job file not found: {job_file}"

    with open(job_file, "r") as f:
        job_data = json.load(f)

    assert job_data["id"] == batch_id
    assert job_data["status"] == "queued"
    assert job_data["num_prompts"] == 2

    input_file = f"/tmp/{batch_id}_input.jsonl"
    assert os.path.exists(input_file), f"Input file not found: {input_file}"

    with open(input_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2
    prompts = [json.loads(line)["prompt"] for line in lines]
    assert "What is 2+2?" in prompts
    assert "Hello world" in prompts

    # Verify queue depth increased
    queue_response = requests.get(f"{server_url}/queue/stats")
    if queue_response.status_code == 200:
        final_depth = queue_response.json().get("depth", 0)
        assert final_depth == initial_depth + 1, (
            f"Queue depth should increase by 1, was {initial_depth}, now {final_depth}"
        )


def test_batch_list(server_url):
    response = requests.get(f"{server_url}/v1/batches")
    assert response.status_code == 200

    result = response.json()
    assert result["object"] == "list"
    assert "data" in result
    assert isinstance(result["data"], list)


def test_health_check(server_url):
    response = requests.get(f"{server_url}/")
    assert response.status_code == 200

    result = response.json()
    assert result["status"] == "healthy"
    assert "service" in result
    assert "version" in result
