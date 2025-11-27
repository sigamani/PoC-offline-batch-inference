# Agent Guide

## Environment Setup

1. **Verify Docker**: Run `docker ps` to check running containers
2. **Build and run lightweight dev Image**: Dockerfile.dev this is made to be lightweight for faster iteration and testing in github codespace. Use this for development and testing before moving to the full image. No GPU acceleration. 
   ```bash
   docker build -f Dockerfile.dev -t .
   docker run -it --rm -v $(pwd):/app <name> /bin/bash
   ```
4. **Container Rule**: DO NOT install python packgages outside of the docker container unless needed to keep the space light. We only have 30 GiG to test. All packages will be build inside the Dockerfile.dev container.
5. The github repository for this project that you are in a git clone of is located here: [gh repo clone sigamani/proj-grounded-telescopes](https://github.com/sigamani/doubleword-technical)

---

## System Diagram

                       ┌─────────────────────────────┐
                       │        Users / Clients      │
                       │ (submit batch requests via │
                       │       HTTP POST /start)     │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌─────────────────────────────┐
                       │       Head Node (CPU)       │
                       │                             │
                       │  FastAPI Gateway            │
                       │  - Receives requests        │
                       │  - Assigns job_id           │
                       │                             │
                       │  Redis DB                   │
                       │  - batch_job_queue          │
                       │  - batch_job_status         │
                       │  - concurrency counter      │
                       │                             │
                       │  SLA & Metrics Tracker      │
                       │  - Tracks ETA per job       │
                       │  - Throughput & tokens/sec  │
                       │  - Alerts if ETA > SLA     │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌─────────────────────────────┐
                       │   Job Dispatcher / Worker   │
                       │  Threads (CPU)             │
                       │  - Dequeue jobs from Redis │
                       │  - Respect concurrency      │
                       │  - Submit batch inference   │
                       │    tasks to GPU nodes       │
                       │  - Update SLA & metrics     │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                 ┌───────────────────────────────┐
                 │       GPU Worker Nodes         │
                 │                               │
                 │  Ray Workers                  │
                 │  - vLLM engine (Qwen2.5-0.5B) │
                 │  - Execute batch inference    │
                 │  - Return results to head    │
                 │  - Report tokens processed   │
                 └─────────────┬─────────────────┘
                               │
                               ▼
                       ┌─────────────────────────┐
                       │   Batch Output Storage   │
                       │  (S3, local disk, etc.) │
                       └─────────────────────────┘


## Project Requirements

### Functional Requirements
- Build an offline batch inference server using Ray Data and vLLM
- Use the official `ray.data.llm` module with `vLLMEngineProcessorConfig` and `build_llm_processor`
- Use `ds.map_batches` for distributed processing

### Non-Functional Requirements
- Must complete batch jobs within a 24-hour SLA window
- System must be configurable and observable

---

## Workflow Rules

### Phase 1: Read Before Planning
1. Read ALL documents in `@doc`, `@app`, and `@config` directories
2. Review the plan in PLAN (below) and check what has been implemented from that in the code base (test if necessary) and then move to either fixing gaps or moving to the next step
3. 
### Phase 2: Execution
2. If human asks for something that contradicts any of the the previous ways of working or instructions that do not follow good AI development practice, or instructions that will probably derail the main plan then STOP and ask for clarification and confirmation before proceeding. You are doing the user a favour by calling out bad ideas.
3. Cross-check your own planning against these instructions
5. Classes or Functions? As a general rule if you are describing a data object use a class, if you are not use a function. 
6. Keep the function definitions to maximum ten lines of code.
7. Keep function names / class names no more than 15 characters. Do not comment in code, use clear names instead.


---

## Code Quality Standards

### Mandatory API Usage
After EVERY task, verify you are using:
- ✅ `from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor`
- ✅ `ds.map_batches()` for distributed processing
- ✅ Official Ray Data patterns from: https://docs.ray.io/en/latest/data/batch_inference.html

If you missed any of these, STOP and fix it immediately.

### Forbidden: DO NOT EVER USE Rich Text Emojis in Code
**DELETE** any line containing these emojis in print statements or logs:
- ❌ 🚀 ✅ 📊 ⚠️ 💡 🔧 🎯 or ANY other emoji

**Bad Example (DELETE THIS)**:
```python
logger.info("🚀 Starting Ray Data + vLLM Batch Inference Server")
logger.info("✅ Configuration loaded")
logger.error("❌ Failed to initialize Ray")
```

**Good Example (USE THIS)**:
```python
logger.info("Starting Ray Data + vLLM Batch Inference Server")
logger.info("Configuration loaded")
logger.error("Failed to initialize Ray")
```

---

## Communication Style

### DO:
- Be methodical, calculating, and calm
- Maintain consistent memory through good organization and bookkeeping
- Reference previous decisions and code when making new changes
- Show your work: explain reasoning before implementing

### DO NOT:
- Write .md summaries or documentation EVER.
- Use emojis in code or logs
- Make changes without verifying against project requirements
- Proceed without reading all context documents first
- Every 3 tasks - perform an audit and springclean for redundant files or code snippets - use the symbiote if needed.
---

## Checklist Before Any Code Changes

```
□ Have I read all documents in @doc, @app, @config?
□ Am I working inside the michaelsigamani/proj-grounded-telescopes container?
□ Am I using ray.data.llm with vLLMEngineProcessorConfig and build_llm_processor?
□ Am I using ds.map_batches for distributed processing?
□ Have I removed all emoji characters from code and logs?
□ Does this change align with the 24-hour SLA requirement?
□ Have I cross-checked against @todo.txt?
```
