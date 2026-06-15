# Text-to-SQL on Qwen3-30B-A3B — Serving, Observability, Eval & SLO Report

> Status: Sections 1, 4 drafted off-GPU against Nebius Token Factory
> (Qwen3-30B-A3B-Instruct-2507, same model). Numbers tagged **[TBD on H100]**
> are filled from the real 1×H100 run.

---

## 1. Serving configuration (Phase 1)

Model is fixed: `Qwen/Qwen3-30B-A3B-Instruct-2507`. Hardware is fixed: 1×H100 80GB.

**Model shape that drives the config:** Qwen3-30B-A3B is a Mixture-of-Experts
model — 30.5B total parameters but only ~3B active per token (8 of 128 experts
fire). Consequence: compute per token is cheap, but **all 30B weights must sit in
VRAM**. In BF16 that is ~61GB of an 80GB card, leaving only ~15GB for the KV
cache. On this workload the bottleneck is KV-cache memory (concurrency), not FLOPs.

**Workload shape:** prompts are 1.5–3K tokens (rendered schema + system prompt),
outputs are short structured SQL (~100–300 tokens), each agent request makes 2–3
dependent calls, and there are only 11 distinct DB schemas, so prompt prefixes
repeat heavily.

| Flag | Value | One-line justification |
|---|---|---|
| `--quantization fp8` | on | Hopper has native FP8 tensor cores. Quantizing weights ~61GB → ~31GB frees ~30GB for KV cache — the single biggest concurrency lever here, with negligible quality loss for SQL. |
| `--max-model-len` | 8192 | Prompts ≤3K + short outputs never need the native 262K context; the long default would reserve huge per-sequence KV. A tight cap lets far more sequences run concurrently. |
| `--enable-prefix-caching` | on | Only 11 schemas and one shared system prompt → the 1.5–3K-token prefix is identical across most requests. Prefill is computed once and reused, cutting the most expensive part of each call. |
| `--gpu-memory-utilization` | 0.92 | Reclaims a bit more KV headroom past the 0.90 default without risking OOM on an 80GB card. |
| `--dtype` | bfloat16 | H100-native compute dtype; base precision for the FP8-quantized weights. |
| `--tensor-parallel-size` | 1 (implicit) | Single GPU — nothing to shard; TP would only add overhead. |

**Why this is non-thinking-friendly:** the `-Instruct-2507` variant does not emit
`<think>` blocks, so outputs stay short — exactly what we want for latency. (Had we
been handed the Thinking variant, disabling reasoning would have been the first move.)

**Levers deliberately held in reserve for Phase 6** (tuned *under load* against the
dashboard, not guessed up front): `--kv-cache-dtype fp8` (≈2× KV capacity if we go
KV-bound), `--max-num-seqs` (cap batch concurrency vs. preemption), tightening
`--max-model-len` to 4096, and `--max-num-batched-tokens` (prefill chunk budget vs.
decode interleave). Keeping these in reserve is intentional — Phase 1 is the
informed static config; Phase 6 is the load-driven iteration.

**Manual sanity check:** `screenshots/vllm_manual_query.png` — vLLM serving on
:8000, one eval question returning sensible SQL. **[TBD on H100]**

---

## 2. Baseline eval (Phase 5)

Execution accuracy on 30 BIRD questions (`evals/eval_set.jsonl`): run the agent's
final SQL and the gold SQL against the target DB, compare canonicalized row sets
(sorted, stringified, NULL→''). Per-iteration pass rate uses carry-forward — a
question that terminated early keeps its last result at later iterations.

- Overall pass rate: **[TBD on H100]** (`results/eval_baseline.json`)
- Pass rate by iteration (0 / 1 / 2): **[TBD on H100]**
- Avg iterations / # questions that triggered a revise: **[TBD on H100]**
- Grafana during the run: `screenshots/grafana_eval_run.png` **[TBD on H100]**

Commentary: **[TBD — does iter-2 pass rate beat iter-0? If yes the loop earns its
keep; if flat, the architecture is doing nothing. Cite the numbers.]**

> Harness validated off-GPU against Nebius: per-iteration carry-forward works, and
> the `formula_1` duplicate-rows question went pass@iter `[0.0, 1.0]` — the verify
> step caught identical JOIN-multiplied rows and revise added DISTINCT to fix it.

---

## 3. Hitting the SLO (Phase 6)

**SLO:** P95 end-to-end agent latency < 5s at 10+ RPS over a 5-min window.
Load: `uv run python load_test/driver.py --rps <n> --duration 300`.

- Baseline vs. SLO (P50 / P95 / P99 @ target RPS): **[TBD on H100]**

**Iteration log** — *saw X → hypothesized Y → changed Z → result W*:

1. **[TBD]** saw … → hypothesized … → changed … → result …
2. **[TBD]** …
3. **[TBD]** …

- Before/after the change that moved the needle:
  `screenshots/grafana_before.png`, `screenshots/grafana_after.png` **[TBD]**
- Final config + final P50/P95/P99: **[TBD on H100]**
- Quality after tuning: `results/eval_after_tuning.json` — did pass rate survive?
  **[TBD]**
- **Honest verdict:** SLO hit, or missed with the gap quantified. **[TBD]**

---

## 4. Did the agent loop add value?

The verify→revise loop is a self-consistency mechanism: instead of trusting a
single generation, the agent executes the SQL, inspects the result, and retries
with targeted feedback when the result looks implausible (SQL errored, zero rows
where rows are implied, or duplicate rows where a single fact / distinct list was
asked for).

Evidence the loop does real work comes from the **per-iteration pass rate**: if
iter-2 ≈ iter-0, the loop is pure latency cost; if iter-2 > iter-0, it is
rescuing questions. **[TBD on H100 — cite the gap.]**

Preliminary off-GPU signal (Nebius, not gradeable but directional): on the
`formula_1` circuit-coordinates question, generation alone produced JOIN-duplicated
rows (wrong), the verifier flagged "duplicate rows for a single-fact question", and
revise added `DISTINCT` — moving that question from fail to pass (pass@iter
`[0.0, 1.0]`). That is exactly the failure class the loop is designed to catch.

The cost side is real and matters for §3: a revise turns a ~1-call request into 3
calls, roughly tripling its latency. The loop's value is only net-positive if the
pass-rate lift justifies that tail-latency cost under the SLO.

---

## 5. What I'd do with more time

- **Pre-quantized FP8 checkpoint** instead of online `--quantization fp8`, to remove
  startup quantization time and pin exact weight scales for reproducible latency.
- **Schema pruning before the prompt:** retrieve only the tables/columns relevant to
  the question (embedding or keyword match over `render_schema`) to cut the 1.5–3K
  prompt to a few hundred tokens — directly attacks the prefill-bound bottleneck.
- **Cheap deterministic verifier first:** many revises trigger on mechanical issues
  (0 rows, SQL error, duplicate rows) detectable in Python without an LLM call —
  gate the LLM verifier behind those checks to save a full call on the common path.
- **Constrained decoding / grammar** (vLLM guided decoding) to force valid
  single-statement SQLite and drop the markdown-fence extraction step entirely.
- **Speculative decoding** with a small draft model to cut decode latency on the
  short SQL outputs, if Phase 6 shows decode (not prefill) is the tail.
- **Separate `max-model-len` per call type:** verify/revise need less context than a
  fresh generate; routing them to a tighter budget would lift concurrency further.
