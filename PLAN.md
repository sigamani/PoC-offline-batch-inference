Here is the content formatted exactly as a plain text file. Copy/paste directly into a `.txt` file:

---

Test Scenarios

1. Baseline Performance
   Fixed input/output lengths. Sweep batch size and QPS.

2. Memory Stress
   Constrain KV cache. Measure throughput degradation and preemption behavior.

3. Real-World Simulation
   Use ShareGPT data with variable input/output lengths.

4. Concurrency Scaling
   Multi-node Ray cluster tests, including throughput and tail latency.

5. SLA Validation
   Ensure large batch workloads complete within a 24-hour window.

---

System Architecture

Flow:
Client
→ FastAPI/gRPC Gateway
→ Redis Queue (Streams preferred)
→ Ray Data (batch building)
→ vLLM Engine (PagedAttention + FCFS preemption)
→ Shared Storage (S3-compatible, Parquet)

---

Memory Management

* Use FCFS preemption for variable output lengths and limited GPU memory.
* Target KV cache allocation: 70–90% of GPU memory.
* Define explicit preemption policy and backpressure threshold.

---

Key Configuration Insights

* vLLM performs better than TGI under memory pressure.
* Batch size tuning is critical; optimal range ~128–256 for tests.
* KV cache fraction strongly affects throughput.
* Larger models reduce max sustainable QPS.
* Use Ray for horizontal scaling.

---

Critical Performance Factors

1. GPU memory split (weights vs KV cache)
2. Batch size vs preemption rate
3. Queueing for high QPS
4. Model size impact on throughput curve

---

Monitoring Requirements

* Throughput
* KV cache usage
* Preemption events
* SLA countdown/violations
* Queue depth

---

Repository Structure (Refactor Plan)

tests/
matrix.yaml      # models, batch sizes, qps, token lengths, kv fractions
scenarios.md     # Baseline, Memory Stress, Real-World, Concurrency, SLA
kpis.md          # throughput, latency, preemptions

