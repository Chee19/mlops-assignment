# CLAUDE.md — Project context, plan & todos

## What this is
Nebius Module 3 MLOps assignment: text-to-SQL agent (LangGraph) on top of vLLM
serving Qwen3-30B-A3B-Instruct-2507, observed with Prometheus/Grafana (serving)
and Langfuse (agent), evaluated by execution accuracy on BIRD, load-tested
against an SLO. Full spec in `README.md`; grading is reasoning-weighted
(Phase 6 diagnosis = 25%, configs/dashboards/evals 15% each).

**SLO:** P95 end-to-end agent latency < 5 s at 10+ RPS over a 5-min window.
**STATUS: SLO HIT** — P95 4.01 s at 10 RPS × 5 min (baseline was 6.17 s).

## ⏭ RESUME TOMORROW (H100) — what's left (~15-20 min GPU, then I write)
VM: `89.169.108.250`, user `sean-hw-1`, key `~/.ssh/nebius_ed25519`.
Connect (with tunnels): `ssh -i ~/.ssh/nebius_ed25519 -L 3000:localhost:3000 -L 9090:localhost:9090 -L 3001:localhost:3001 -L 8000:localhost:8000 -L 8001:localhost:8001 sean-hw-1@89.169.108.250`
Each new shell: `export PATH="$HOME/.local/bin:$PATH"; cd ~/mlops-assignment`. Use **tmux** — processes die on terminal close.

1. **Bring the stack back up** (all 3 likely died on disconnect):
   - `docker compose up -d` (if containers down)
   - `git pull origin main` (has schema fix + MAX_ITERATIONS=2)
   - vLLM: `bash scripts/start_vllm.sh` (own shell; ~2-3 min) — confirm `curl :8000/v1/models`
   - agent: `uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001 > /tmp/agent.log 2>&1 &` — confirm a `/answer` curl (request field is **`db`**, not `db_id`)
2. **Eval — after tuning (MAX=2, current code):**
   `uv run python evals/run_eval.py --out results/eval_after_tuning.json` → paste summary to Claude (CHECKPOINT 2).
3. **Eval — baseline (MAX=3):** set `agent/graph.py` `MAX_ITERATIONS` 2→3, restart agent server, `uv run python evals/run_eval.py --out results/eval_baseline.json`, then set back to 2 + restart. Paste summary.
4. **`grafana_after.png`:** Prometheus stopped scraping vLLM after the restart — check `:9090 → Status → Targets` (vLLM UP?) and set Grafana refresh=5s. Fire a short burst (`--rps 10 --duration 60`), screenshot the Latency panel showing P95 ~4 s.
5. Claude fills REPORT §2 (eval numbers) + §4 (per-iteration pass rate). Then shut the VM down.

## Hard rules (agreed with user)
- Fill designated stubs ONLY. Never modify provided/complete files
  (`agent/server.py`, `agent/execution.py`, `agent/schema.py`,
  `load_test/driver.py`, `scripts/load_data.py`, `docker-compose.yml`,
  `infra/prometheus.yml`) without asking the user first.
- Always `uv`, never bare pip/python. On this Mac: `uv run --no-sync ...`
  (plain `uv run` tries to build vllm from source and fails; vllm only
  installs on the H100 VM).
- `.env` is user-managed: never read/print/edit the secret values; ask the
  user to make changes to it.
- Teach while working: explain choices and tradeoffs as tasks complete.

## Working setup (current)
- Machine: user's Mac (no GPU). H100 VM comes later for real numbers.
- Dev backend: Nebius Token Factory, OpenAI-compatible.
  - Base URL: `https://api.tokenfactory.nebius.com/v1/` (verified live)
  - Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` (verified — exact assignment model)
  - Configured via `VLLM_BASE_URL` / `VLLM_MODEL` / `OPENAI_API_KEY` in `.env`.
- BIRD data loaded: 11 DBs under `data/bird/`, `evals/eval_set.jsonl` (30 q),
  `load_test/perf_pool.jsonl` (1500 q).
- Deps installed via `uv sync --no-install-package vllm`.

## Plan & todo list
Off-GPU first (Phases 3→4→5 harness), then one efficient H100 session
(Phases 1→2→5-real→6), then report.

- [x] Phase 0 (local): uv sync, BIRD data, .env → Nebius verified
- [x] Phase 3 — Agent: prompts.py + verify/revise/router implemented; 5 eval
      questions fired via `POST :8001/answer`; revise triggered naturally on
      the `financial` avg-crimes-1995 question (2 revises, capped at 3 iters).
      Remaining: final prompt tuning against the real H100 endpoint.
- [x] Phase 4 — Tracing: Langfuse up, keys in `.env` (uses LANGFUSE_BASE_URL —
      the v4 name; LANGFUSE_HOST is deprecated). auth_check True. 10 tagged
      questions fired through the server; all 10 traces landed with metadata
      {phase,db_id,run,q_index}. Q4 financial = richest (20 obs, 2 revises) for
      the waterfall screenshot. DONE: `screenshots/langfuse_trace.png` (financial
      20-obs trace, graph shows execute 3/3 / verify 3/3 / revise 2/2) +
      `screenshots/langfuse_tags.png` (list filtered by metadata db_id=financial).
      Langfuse project: sean-nebius / mlops-hw3. Phase 4 CLOSED.
- [x] Phase 5 (harness, local): `eval_one()` + `summarize()` implemented and
      validated vs Nebius (smoke runs in /tmp, NOT results/). Carry-forward
      per-iteration pass rate works; formula_1 duplicate-rows case showed
      pass@iter [0.0, 1.0] — verify→revise rescued it (good REPORT.md material).
- [ ] H100 session — Phase 0 (VM): ports 3000/9090/3001/8000/8001 forwarded,
      repo + data + docker compose up
- [~] Phase 1 — vLLM config: DRAFTED off-GPU. `scripts/start_vllm.sh` has the
      baseline flags (fp8, max-model-len 8192, prefix-caching, gpu-util 0.92,
      bf16); rationale table in REPORT.md §1. Phase 6 levers held in reserve
      (kv-cache-dtype fp8, max-num-seqs, max-model-len→4096, max-num-batched-tokens).
      REMAINING on H100: run it, confirm load, screenshot `vllm_manual_query.png`.
- [~] Phase 2 — Grafana: DRAFTED off-GPU. `serving.json` extended to 13 panels
      (kept 2 starters + 3 rows): Latency [E2E p50/p95/p99, TTFT-vs-TPOT split,
      queue p95, running-vs-waiting], Throughput [prompt-vs-gen tokens/s,
      completion rate], KV cache [gpu_cache_usage_perc w/ thresholds,
      preemptions/s]. REMAINING on H100: confirm vLLM metric NAMES match the
      live /metrics (version drift possible), fire load, screenshot
      `grafana_serving.png` with panels reacting.
- [ ] Phase 5 (real): baseline eval on H100 → `results/eval_baseline.json`;
      screenshot `grafana_eval_run.png`; per-iteration pass-rate read
- [ ] Phase 6 — SLO: load test 10 RPS × 300 s; iterate
      "saw X → hypothesized Y → changed Z → result W" (3–4 iterations);
      screenshots `grafana_before.png`/`grafana_after.png`;
      `results/eval_after_tuning.json`; honest verdict
- [ ] Phase 7 — REPORT.md ≤3 pages: serving config, baseline eval,
      SLO iteration log, agent-value paragraph (cite per-iteration pass rate),
      specific "with more time"

## Deliverables tracker (where each artifact stands)
Legend: ✅ done · 🟡 drafted, needs H100 numbers/screenshot · ⛔ blocked on H100

| Deliverable | Status | Where / note |
|---|---|---|
| `agent/graph.py` (verify/revise/router) | ✅ | implemented + tested vs Nebius |
| `agent/prompts.py` (6 prompts) | ✅ | generate/verify/revise; revise loop fires |
| `evals/run_eval.py` (eval_one/summarize) | ✅ | validated on /tmp smoke runs |
| `scripts/start_vllm.sh` (flags) | 🟡 | flags set; must actually run on H100 |
| `infra/grafana/.../serving.json` | 🟡 | 13 panels; verify metric names on live /metrics |
| `REPORT.md` §1 serving config | ✅ | full flag table + MoE rationale |
| `REPORT.md` §4 agent value | 🟡 | drafted; cite real per-iter pass rate |
| `REPORT.md` §5 with-more-time | ✅ | 6 specific items |
| `REPORT.md` §2 baseline eval | ⛔ | `[TBD on H100]` placeholders in place |
| `REPORT.md` §3 SLO iteration log | ⛔ | `[TBD on H100]` placeholders in place |
| `results/eval_baseline.json` | ⛔ | produced by Phase 5 real run |
| `results/eval_after_tuning.json` | ⛔ | produced by Phase 6 |
| `screenshots/langfuse_trace.png` | ✅ | saved (financial 20-obs, 3/3 + 2/2) |
| `screenshots/langfuse_tags.png` | ✅ | saved (metadata filter db_id=financial) |
| `screenshots/vllm_manual_query.png` | ⛔ | Phase 1 on H100 |
| `screenshots/grafana_serving.png` | ⛔ | Phase 2 on H100 (panels under load) |
| `screenshots/grafana_eval_run.png` | ⛔ | Phase 5 on H100 (dashboard during eval) |
| `screenshots/grafana_before.png` / `_after.png` | ⛔ | Phase 6 on H100 (the change that moved P95) |

## H100 runbook (one efficient session — run top to bottom)
All commands run ON THE VM unless marked [LAPTOP]. Use plain `uv run` here
(NOT --no-sync): vllm must actually install on the GPU box.

There are 3 STOP points where you paste output to Claude and wait before
continuing — Claude can't proceed blind there:
  - CHECKPOINT 1 (Step 2): live /metrics names → Claude fixes dashboard exprs
  - CHECKPOINT 2 (Step 3): eval summary JSON → Claude reads pass rates, writes §2
  - CHECKPOINT 3 (Step 4): load-test result + what moved → Claude diagnoses the lever

### Step 0 — connect + bring up stack
```bash
# [LAPTOP] forward the 5 UIs over SSH (keep this shell open)
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 3001:localhost:3001 \
    -L 8000:localhost:8000 -L 8001:localhost:8001 <user>@<vm-host>

# [VM] repo + deps + data + o11y stack
git clone <repo-url> && cd <repo-folder>
uv sync                         # installs vllm too (works on GPU)
cp .env.example .env            # then set HF_TOKEN; Langfuse keys optional on VM
uv run python scripts/load_data.py
docker compose up -d
# sanity [LAPTOP browser]: :9090 Prometheus, :3000 Grafana (admin/admin), :3001 Langfuse
```

### Step 1 — Phase 1: serve vLLM + manual query
```bash
# [VM] start serving (FP8 online-quantizes at load; first boot may take minutes)
bash scripts/start_vllm.sh                       # leave running in its own shell
# in another [VM] shell, confirm it loaded + sensible SQL:
curl -s localhost:8000/v1/models | python3 -m json.tool
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen/Qwen3-30B-A3B-Instruct-2507",
  "messages":[{"role":"user","content":"Reply with one SQLite query selecting all rows from table t."}],
  "max_tokens":64,"temperature":0}' | python3 -m json.tool
```
SCREENSHOT → `screenshots/vllm_manual_query.png` (terminal showing serve log + a query returning SQL).

### Step 2 — Phase 2: reconcile metric names, then dashboard
```bash
# [VM] CRITICAL: confirm the panel metric names exist on this vLLM version
curl -s localhost:8000/metrics | grep -E "vllm:(e2e_request_latency|time_to_first_token|time_per_output_token|request_queue_time|gpu_cache_usage|num_preemptions|prompt_tokens|generation_tokens|request_success|num_requests)" | grep -v '#' | cut -d'{' -f1 | sort -u
```
🛑 CHECKPOINT 1 — paste that output to Claude and WAIT. Any panel showing "No data"
gets its `expr` fixed in serving.json (e.g. gpu_cache_usage_perc vs kv_cache_usage_perc).
Reload the Grafana dashboard after Claude's edits.
```bash
# [VM] point agent at LOCAL vLLM (edit .env: VLLM_BASE_URL=http://localhost:8000/v1,
#       remove the Nebius lines), then start the agent server:
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001 &
# fire a burst so every panel reacts, watch Grafana:
uv run python load_test/driver.py --rps 8 --duration 60
```
SCREENSHOT → `screenshots/grafana_serving.png` (full dashboard, panels moving).

### Step 3 — Phase 5: baseline eval (real pass rates)
```bash
# [VM] 30 questions × ~2 calls; watch Grafana during it
uv run python evals/run_eval.py --out results/eval_baseline.json
```
SCREENSHOT → `screenshots/grafana_eval_run.png` (dashboard during the eval).
🛑 CHECKPOINT 2 — paste the printed summary JSON to Claude. Claude reads
summary.pass_rate_at_iteration (does iter-2 beat iter-0? = loop earns its keep)
and fills REPORT §2.

### Step 4 — Phase 6: SLO load test + iterate (the 25% phase)
```bash
# [VM] the SLO run: 10 RPS for 5 min
uv run python load_test/driver.py --rps 10 --duration 300 --out results/load_test_baseline.json
```
🛑 CHECKPOINT 3 (the 25% phase — do NOT tune blind). Screenshot the dashboard and
paste Claude the load-test summary + which panels spiked. Claude diagnoses:
  queue p95 ↑ = concurrency-capped · KV ~100% + preemptions>0 = KV-bound · TPOT ↑ = decode-bound
Then Claude names ONE reserved lever (kv-cache-dtype fp8 / max-num-seqs /
max-model-len 4096). You change it, restart vLLM, re-run, confirm the targeted
metric moved AND whether P95 followed. Repeat 3–4×.
Per iteration → one REPORT §3 line "saw X → hyp Y → changed Z → result W" + a screenshot.
SCREENSHOTS → `screenshots/grafana_before.png`, `screenshots/grafana_after.png`.
```bash
# [VM] final config: re-run eval to prove quality survived tuning
uv run python evals/run_eval.py --out results/eval_after_tuning.json
```

### Step 5 — Phase 7: finish REPORT.md
Fill every `[TBD on H100]`: §2 baseline numbers, §3 baseline-vs-SLO + iteration log +
final numbers + honest verdict, §4 cite the real per-iteration pass-rate gap. Keep ≤3 pages.

## Gotchas learned
- `python-dotenv`: last occurrence of a key in `.env` wins; `server.py` reads
  it only at startup — restart the server after env changes.
- README's `from langfuse.callback import CallbackHandler` is outdated (v2);
  repo uses langfuse v4 and `server.py` already wires
  `langfuse.langchain.CallbackHandler` — Phase 4 only needs keys + metadata tags.
- vLLM ignores API keys but `ChatOpenAI` requires one — `OPENAI_API_KEY`
  doubles as the Nebius key now, harmless dummy later on the H100.
- `prompts.py` (like `.env`) is read at server start — ALWAYS restart the
  agent server after editing prompts, or you're testing stale prompts.
- Agent server runs detached: `uv run --no-sync uvicorn agent.server:app
  --host 127.0.0.1 --port 8001` (log: /tmp/agent_server.log).
- Prompt lesson: temp-0 generation ignored a one-line DISTINCT rule; adding a
  duplicate-rows check to VERIFY_SYSTEM let the revise loop fix it instead.
  Single-shot prompt nudges < architectural checks.
- `_parse_verdict` hardened: `bool("false")` is truthy in Python, so a
  stringified `{"ok":"false"}` used to skip revise. `_coerce_ok()` now maps
  false/no/0/"" → False; genuinely-absent/unparseable still fails open (True).
  Covered by an inline 8-case check (run ad-hoc; no test file in repo).

## Gotchas learned ON THE H100 (2026-06-15/16 session)
- **vLLM 0.10.2 ✗ transformers 5.x**: the off-GPU lockfile (built without vllm
  in the resolver) pulled transformers 5.9.0 → `Qwen2Tokenizer has no attribute
  all_special_tokens_extended`. Fix: `uv add 'transformers<5'` (pins it in
  pyproject so every `uv run` respects it — a bare `uv pip install` gets reverted
  because **`uv run` auto-syncs to the lockfile before running**).
- **Triton needs `Python.h`**: vLLM's torch.compile JIT shells out to gcc against
  the Python C headers → `fatal error: Python.h` on a bare VM. Fix:
  `sudo apt-get install -y python3-dev build-essential`.
- **`render_schema` (provided) crashed on NULL FK parent column** (SQLite returns
  NULL `to` when a FK implicitly references the parent PK). 500'd ~12% of the load
  pool (every request to those DBs). Fixed with user approval: fall back to child
  column name. This was the cause of the load-test 500s, NOT concurrency.
- **Processes die when the terminal closes** (SIGHUP). vLLM + agent server both
  died on disconnect. Use **tmux** for anything long-running.
- **Driver request field is `db`** (maps from `db_id`); a manual curl with
  `db_id` returns 422. The 500s in logs only show the access line — the exception
  detail is in the HTTPException *response body* (server.py:65), not the log.
- **driver.py is open-loop with a 60s drain cap** → `wall_clock = duration + 60`
  when there's a backlog (the suspiciously-exact 90.0s / 360.0s). `achieved_rps`
  and latency percentiles only count `ok` requests.
- **SLO diagnosis = agent-bound, not serving-bound.** Under 10 RPS load: KV ~3%,
  waiting 0, preemptions 0, TTFT/TPOT 40/20ms — vLLM idle. So reserved vLLM
  levers can't help; the fix was agent-side.
- **Phase 6 iteration outcomes:** (1) `max_tokens=256` → REGRESSION (truncated SQL
  → revise-loop explosion, P95 6.17→77.6s), reverted. (2) `MAX_ITERATIONS 3→2` →
  SLO HIT (P95 6.17→4.01s). Final config: vLLM unchanged, agent MAX_ITERATIONS=2.
- **Prometheus stopped scraping vLLM after a vLLM restart** — `grafana_after.png`
  still outstanding; check `:9090 → Status → Targets` next session.
- **Push auth**: this Mac commits as `Chee22` but repo is `Chee19`'s — push needs
  the right GitHub account active; commits land, push 403s until switched.
