#!/usr/bin/env bash
#
# Start vLLM serving Qwen3-30B-A3B for the text-to-SQL agent.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Workload profile this config targets:
#   - prompts 1.5-3K tokens (schema + system prompt), outputs short (SQL ~100-300 tok)
#   - 2-3 dependent LLM calls per agent request (generate -> verify -> maybe revise)
#   - only 11 distinct DB schemas -> highly repetitive prompt prefixes
#   - SLO: P95 end-to-end agent latency < 5s at 10+ RPS
# Rationale for each flag is in REPORT.md section 1.

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --quantization fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching
