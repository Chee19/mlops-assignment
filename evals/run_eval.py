"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_ok, gold_rows, gold_err = run_sql(db_id, question["gold_sql"])

    t0 = time.monotonic()
    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": db_id},
            timeout=120.0,
        )
        resp.raise_for_status()
        answer = resp.json()
        agent_err = None
    except Exception as e:  # noqa: BLE001
        answer = {}
        agent_err = f"{type(e).__name__}: {e}"
    latency = time.monotonic() - t0

    # Re-execute every SQL attempt: "had we stopped here, would it be right?"
    attempts = [
        h["sql"] for h in answer.get("history", [])
        if h.get("node") in ("generate_sql", "revise")
    ]
    iteration_correct: list[bool] = []
    for sql in attempts:
        ok, rows, _ = run_sql(db_id, sql)
        iteration_correct.append(gold_ok and ok and matches(gold_rows, rows))

    return {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": question["gold_sql"],
        "gold_error": gold_err,
        "final_sql": answer.get("sql", ""),
        "iterations": answer.get("iterations", 0),
        "agent_ok": answer.get("ok", False),
        "agent_error": agent_err or answer.get("error"),
        "latency_seconds": latency,
        "iteration_correct": iteration_correct,
        "correct": iteration_correct[-1] if iteration_correct else False,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    max_iters = max((len(r["iteration_correct"]) for r in results), default=0)

    pass_at_iteration: list[float] = []
    for k in range(max_iters):
        correct_at_k = 0
        for r in results:
            ic = r["iteration_correct"]
            if not ic:
                continue
            # carry-forward: terminated runs keep their last result
            correct_at_k += ic[min(k, len(ic) - 1)]
        pass_at_iteration.append(correct_at_k / n if n else 0.0)

    return {
        "total_questions": n,
        "correct": sum(r["correct"] for r in results),
        "pass_rate": (sum(r["correct"] for r in results) / n) if n else 0.0,
        "pass_rate_at_iteration": pass_at_iteration,
        "avg_iterations": (sum(r["iterations"] for r in results) / n) if n else 0.0,
        "revise_triggered": sum(1 for r in results if len(r["iteration_correct"]) > 1),
        "agent_errors": sum(1 for r in results if r["agent_error"]),
        "avg_latency_seconds": (sum(r["latency_seconds"] for r in results) / n) if n else 0.0,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
