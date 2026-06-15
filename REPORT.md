# Text-to-SQL on Qwen3-30B-A3B — Serving, Observability, Eval & SLO Report

> Status: Run on 1×H100 80GB. §1 serving config and §3 SLO iteration log are
> filled with real numbers. §2 baseline eval and the per-iteration pass rates
> in §4 are pending a final eval pass (harness validated; numbers tagged
> **[TBD — eval run]**).

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
:8000, a chat-completion returning a valid SQLite query (`SELECT * FROM t;`,
`finish_reason: stop`). Model loaded in ~29 GiB (FP8 weights), confirming the
quantization math above.

**Serving-readiness fix found on the H100:** vLLM 0.10.2 is incompatible with
`transformers` 5.x (the lockfile, built off-GPU without vllm in the resolver,
had pulled 5.9.0) — pinned `transformers<5` in `pyproject.toml`. Also installed
`python3-dev` so vLLM's Triton JIT could compile its CUDA kernels. Both are the
kind of environment drift that only surfaces on the real serving box.

---

## 2. Baseline eval (Phase 5)

Execution accuracy on 30 BIRD questions (`evals/eval_set.jsonl`): run the agent's
final SQL and the gold SQL against the target DB, compare canonicalized row sets
(sorted, stringified, NULL→''). Per-iteration pass rate uses carry-forward — a
question that terminated early keeps its last result at later iterations.

Two eval passes are run so the loop's value can be isolated from the SLO tuning:
`eval_baseline.json` at the Phase-1 agent (`MAX_ITERATIONS=3`) and
`eval_after_tuning.json` at the Phase-6 agent (`MAX_ITERATIONS=2`).

- Overall pass rate (baseline / after-tuning): **[TBD — eval run]**
- Pass rate by iteration (0 / 1 / 2): **[TBD — eval run]**
- Avg iterations / # questions that triggered a revise: **[TBD — eval run]**

Commentary: **[TBD — does iter-1/iter-2 pass rate beat iter-0 (loop earns its
keep), and does cutting iter-2 in Phase 6 cost any accuracy? Cite the numbers.]**

> A prerequisite bug was fixed before any clean run: `render_schema` (provided)
> crashed with `AttributeError` on BIRD DBs whose foreign keys implicitly
> reference the parent PK (SQLite returns a NULL parent column). Because schema
> rendering is the first graph node, this 500'd **every** request to those DBs —
> ~12% of the load pool — independent of RPS. Fixed by falling back to the child
> column name when the parent is NULL. This had to be fixed for both the eval and
> the SLO numbers to mean anything.
>
> Harness validated off-GPU against Nebius: per-iteration carry-forward works, and
> the `formula_1` duplicate-rows question went pass@iter `[0.0, 1.0]` — the verify
> step caught identical JOIN-multiplied rows and revise added DISTINCT to fix it.

---

## 3. Hitting the SLO (Phase 6)

**SLO:** P95 end-to-end agent latency < 5s at 10+ RPS over a 5-min window.
Load: `uv run python load_test/driver.py --rps 10 --duration 300`.

**Baseline @ 10 RPS × 300s** (Phase-1 serving, agent `MAX_ITERATIONS=3`):

| P50 | P95 | P99 | max | achieved RPS | ok |
|---|---|---|---|---|---|
| 1.41s | **6.17s** | 10.26s | 85s | 8.82 | 99.7% |

→ **MISS** — P95 6.17s > 5s. (`results/load_test_baseline.json`)

**Diagnosis — read off Grafana *under load* (`grafana_before*.png`):** vLLM was
idle, not the bottleneck. KV cache ~3%, `requests waiting` flat at 0,
preemptions 0, TTFT p95 ~40ms, TPOT p95 ~20ms, `running` peaked ~30. The latency
lives entirely in the **agent's sequential LLM call chain** (generate → verify →
revise → verify), not in serving. Critically, this means the *reserved serving
levers* (`--max-num-seqs`, `--kv-cache-dtype fp8`, `--max-num-batched-tokens`,
`--max-model-len 4096`) would tune a resource that is already 97% idle — they
cannot move this P95. That is the central call of this phase: **the fix is
agent-side, and turning vLLM knobs here would be cargo-culting.**

**Iteration log** — *saw X → hypothesized Y → changed Z → result W*:

1. **Cap output length → REGRESSION (reverted).** *Saw* an 85s max latency and a
   ~2-min spike on vLLM's E2E-latency panel while KV/queue stayed idle.
   *Hypothesized* a runaway generation decoding to the model's default token
   ceiling. *Changed* `llm()` to `max_tokens=256`. *Result:* P95 **6.17s →
   77.6s** — a 12× regression. The cap truncated valid SQL mid-statement →
   execution errored → verifier rejected → the revise loop fired on nearly every
   request, multiplying calls ~2 → ~6. **Lesson:** the tail was the *revise
   loop*, not token count; truncation feeds the loop. Reverted.

2. **Shorten the revise loop → SLO HIT.** *Saw* (from iter-1) that loop depth is
   the true tail driver. *Hypothesized* the 2nd revise iteration is mostly
   latency cost and rarely fixes anything a 1st revise didn't. *Changed*
   `MAX_ITERATIONS 3 → 2`. *Result:*

   | metric | baseline | after | Δ |
   |---|---|---|---|
   | P50 | 1.41s | 1.28s | −9% |
   | **P95** | **6.17s** | **4.01s** | **−35% ✅** |
   | P99 | 10.26s | 6.58s | −36% |
   | max | 85s | 19.4s | −77% |
   | achieved RPS | 8.82 | 9.44 | +7% |
   | ok | 99.7% | 99.7% | flat |

   Cutting the worst-case chain from ~6 calls to ~4 crushed the tail; because the
   slow requests no longer hog agent threads, queueing eased and throughput rose
   too. (`results/load_test_iter2.json`)

- Before/after: `screenshots/grafana_before*.png` (serving idle under load),
  `screenshots/grafana_after.png` **[TBD — re-grab; Prometheus scrape of vLLM
  needs reconnecting after a vLLM restart]**.
- **Final config:** vLLM **unchanged from Phase 1** (it was never the
  bottleneck); agent `MAX_ITERATIONS=2`. Final P50/P95/P99 @ 10 RPS × 5 min:
  **1.28 / 4.01 / 6.58 s.**
- Quality after tuning: `results/eval_after_tuning.json` vs `eval_baseline.json`
  — did pass rate survive cutting iter-2? **[TBD — eval run]**
- **Honest verdict:** **SLO HIT.** P95 4.01s < 5s at 10 RPS offered (9.44
  achieved) over a 5-min window. Caveats: achieved RPS is 9.44 not a clean 10
  (open-loop driver + drain tail), and a *cold* vLLM restart produces run-to-run
  variance (a re-run cancelled ~6% of tail requests during warm-up); warm runs
  are clean. The win came from one agent-side change, with serving left as-is —
  exactly what the diagnosis predicted.

---

## 4. Did the agent loop add value?

The verify→revise loop is a self-consistency mechanism: instead of trusting a
single generation, the agent executes the SQL, inspects the result, and retries
with targeted feedback when the result looks implausible (SQL errored, zero rows
where rows are implied, or duplicate rows where a single fact / distinct list was
asked for).

Evidence the loop does real work comes from the **per-iteration pass rate**: if
iter-1 ≈ iter-0, the loop is pure latency cost; if iter-1 > iter-0, it is
rescuing questions. **[TBD — eval run; compare `eval_baseline.json` (MAX=3) vs
`eval_after_tuning.json` (MAX=2) — did the 2nd revise ever lift accuracy, and did
removing it in Phase 6 cost any pass rate?]**

The **cost** side is now quantified directly from Phase 6: the revise loop is the
single biggest driver of tail latency. Cutting just the *second* revise iteration
(`MAX_ITERATIONS 3→2`) dropped P95 from 6.17s to 4.01s and the max from 85s to
19s — i.e. that one extra iteration was adding ~2s at P95 and ~65s at the
extreme. So the loop's first revise is plausibly worth its cost (it rescued the
`formula_1` case off-GPU), but each additional iteration is expensive tail
latency that must be justified by a matching pass-rate lift. The Phase 6 result
is a bet that iter-2 rarely pays off; the pending eval confirms or refutes it.

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
