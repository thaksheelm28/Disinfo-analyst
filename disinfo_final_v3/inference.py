"""
inference.py — Baseline LLM agent for Disinformation Analyst (OpenEnv)

MANDATORY env vars:
    HF_TOKEN       Your Hugging Face / API key  (NO default — must be set)
    API_BASE_URL   The API endpoint for the LLM (default: HF router)
    MODEL_NAME     The model identifier          (default: Qwen/Qwen2.5-72B-Instruct)

OPTIONAL env vars:
    LOCAL_IMAGE_NAME  Docker image name when using from_docker_image()
    OPENENV_BASE_URL  Override the environment server URL

STDOUT FORMAT  (exact — validator checks this):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    python inference.py                         # all 3 tasks (requires HF_TOKEN)
    python inference.py --task task1_classify   # single task
    python inference.py --url http://host:7860  # remote server
    python inference.py --dry-run               # no API key needed
    python inference.py --seed 42               # reproducible run
"""

from __future__ import annotations
import argparse, json, os, sys, textwrap
from typing import List, Optional
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Env vars — EXACTLY as required by the submission checklist
#
#   Checklist item 1: inference.py is in the root directory  ✓ (this file)
#   Checklist item 2: Defaults set ONLY for API_BASE_URL and MODEL_NAME
#                     HF_TOKEN has NO default
#   Checklist item 3: LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
#   Checklist item 4: All LLM calls use OpenAI client via these variables
#   Checklist item 5: Stdout follows [START]/[STEP]/[END] format exactly
# ---------------------------------------------------------------------------

API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")           # NO default — mandatory at runtime
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # Optional — for from_docker_image()

BENCHMARK       = "disinfo-analyst"
DEFAULT_ENV_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
MAX_STEPS       = 60

# ---------------------------------------------------------------------------
# Structured log helpers — output format is MANDATORY (validator checks it)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# System prompts — one per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "task1_classify": textwrap.dedent("""
    You are an expert disinformation analyst. Classify every node in the graph.
    Valid labels: real | disinformation | satire | misleading | unverified
    Signals: source domain, share count, text content, timestamp.
    Reply with ONLY a single JSON action per turn:
      {"type": "tag", "node_id": "<id>", "label": "<label>"}
    When all nodes are tagged, reply: {"type": "done"}
    """).strip(),

    "task2_trace": textwrap.dedent("""
    You are an expert disinformation analyst. Trace propagation chains.
    Use edges, timestamps, and credibility to find upstream sources.
    Reply with ONLY a single JSON action per turn:
      {"type": "trace", "src": "<origin_id>", "dst": "<downstream_id>"}
    When done, reply: {"type": "done"}
    """).strip(),

    "task3_severity": textwrap.dedent("""
    You are an expert disinformation analyst. Rate claim severity [0.0-1.0].
    0.0=harmless, 0.5=moderately harmful, 1.0=extremely harmful.
    Reply with ONLY a single JSON action per turn:
      {"type": "score", "claim_id": "<id>", "severity": <0.0-1.0>}
    When done, reply: {"type": "done"}
    """).strip(),
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def reset_env(url: str, task: str, seed: int = 42) -> dict:
    r = requests.post(f"{url}/reset", json={"task": task, "seed": seed, "max_steps": MAX_STEPS}, timeout=30)
    r.raise_for_status()
    return r.json()

def step_env(url: str, action: dict) -> dict:
    r = requests.post(f"{url}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()

def evaluate_env(url: str) -> dict:
    r = requests.get(f"{url}/evaluate", timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(obs: dict, task: str) -> str:
    nodes  = obs.get("nodes", [])
    edges  = obs.get("edges", [])
    claims = obs.get("claims", [])
    tagged = obs.get("agent_tags", {})
    scored = obs.get("agent_scores", {})

    lines = [f"Task: {task}", f"Steps remaining: {obs.get('steps_remaining', '?')}"]

    if task == "task1_classify":
        untagged = [n for n in nodes if n["node_id"] not in tagged]
        lines.append(f"Nodes to tag ({len(untagged)} remaining):")
        for n in untagged[:5]:
            lines.append(f"  {n['node_id']} | {n['kind']} | {n['source']} | shares={n['share_count']} | {n['text'][:80]}")
        if not untagged:
            lines.append("All nodes tagged. Reply: {\"type\": \"done\"}")

    elif task == "task2_trace":
        lines.append(f"Nodes ({len(nodes)}):")
        for n in nodes[:8]:
            lines.append(f"  {n['node_id']} | {n['kind']} | {n['timestamp']}")
        lines.append(f"Edges ({len(edges)}):")
        for e in edges[:8]:
            lines.append(f"  {e['src']} --{e['relation']}--> {e['dst']}")

    elif task == "task3_severity":
        unscored = [c for c in claims if c["claim_id"] not in scored]
        lines.append(f"Claims to score ({len(unscored)} remaining):")
        for c in unscored[:5]:
            lines.append(f"  {c['claim_id']}: {c['text'][:100]}")
        if not unscored:
            lines.append("All claims scored. Reply: {\"type\": \"done\"}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Parse LLM response → action dict
# ---------------------------------------------------------------------------

def parse_action(text: str, task: str) -> dict:
    text = text.strip()
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {"type": "done"}

# ---------------------------------------------------------------------------
# Run one task — live LLM via OpenAI client
# Checklist item 4: ALL LLM calls go through OpenAI client
# ---------------------------------------------------------------------------

def run_task(url: str, task: str, client: OpenAI, seed: int = 42, verbose: bool = True) -> dict:
    obs      = reset_env(url, task, seed)
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[task]}]
    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.get("steps_remaining", 1) <= 0:
                break

            messages.append({"role": "user", "content": build_prompt(obs, task)})

            try:
                # All LLM calls use the OpenAI client configured via API_BASE_URL / MODEL_NAME / HF_TOKEN
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=200,
                    stream=False,
                )
                reply = (completion.choices[0].message.content or "").strip()
            except Exception as exc:
                print(f"[DEBUG] LLM error: {exc}", flush=True)
                reply = '{"type": "done"}'

            action = parse_action(reply, task)
            result = step_env(url, action)

            reward      = result.get("reward", {}).get("step_reward", 0.0) or 0.0
            done        = result.get("done", False)
            error       = result.get("info", {}).get("error", None)
            obs         = result.get("observation", obs)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=json.dumps(action), reward=reward, done=done, error=error)
            messages.append({"role": "assistant", "content": reply})

            if done or action.get("type") == "done":
                break

        final   = evaluate_env(url)
        score   = final.get("final_score", 0.0)
        success = score > 0.1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return final

# ---------------------------------------------------------------------------
# Dry-run — deterministic baseline, no API key required
# ---------------------------------------------------------------------------

def run_dry_run(url: str, seed: int = 42, verbose: bool = True) -> dict:
    results = {}
    for task in ["task1_classify", "task2_trace", "task3_severity"]:
        obs = reset_env(url, task, seed)
        rewards: List[float] = []
        steps_taken = 0
        score   = 0.0
        success = False

        log_start(task=task, env=BENCHMARK, model="dry-run")

        try:
            if task == "task1_classify":
                for n in obs.get("nodes", []):
                    result = step_env(url, {"type": "tag", "node_id": n["node_id"], "label": "disinformation"})
                    r = result.get("reward", {}).get("step_reward", 0.0) or 0.0
                    rewards.append(r)
                    steps_taken += 1
                    log_step(steps_taken, f"tag:{n['node_id']}", r, result.get("done", False), None)

            elif task == "task2_trace":
                nodes = obs.get("nodes", [])
                if len(nodes) >= 2:
                    result = step_env(url, {"type": "trace", "src": nodes[0]["node_id"], "dst": nodes[1]["node_id"]})
                    r = result.get("reward", {}).get("step_reward", 0.0) or 0.0
                    rewards.append(r)
                    steps_taken = 1
                    log_step(1, f"trace:{nodes[0]['node_id']}->{nodes[1]['node_id']}", r, result.get("done", False), None)

            elif task == "task3_severity":
                for c in obs.get("claims", []):
                    result = step_env(url, {"type": "score", "claim_id": c["claim_id"], "severity": 0.5})
                    r = result.get("reward", {}).get("step_reward", 0.0) or 0.0
                    rewards.append(r)
                    steps_taken += 1
                    log_step(steps_taken, f"score:{c['claim_id']}", r, result.get("done", False), None)

            step_env(url, {"type": "done"})
            final   = evaluate_env(url)
            score   = final.get("final_score", 0.0)
            success = score > 0.1
            results[task] = score

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        if verbose:
            print(f"[DEBUG] {task}: score={score:.4f}", flush=True)

    return results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DisinfoEnv baseline agent")
    parser.add_argument("--task",    default=None, choices=["task1_classify", "task2_trace", "task3_severity"])
    parser.add_argument("--url",     default=DEFAULT_ENV_URL)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    tasks   = [args.task] if args.task else ["task1_classify", "task2_trace", "task3_severity"]

    if args.dry_run:
        run_dry_run(url=args.url, seed=args.seed, verbose=verbose)
        return

    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN environment variable (no default is provided).", file=sys.stderr)
        sys.exit(1)

    # All LLM calls use the OpenAI client configured via API_BASE_URL / MODEL_NAME / HF_TOKEN
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    all_scores = {}

    for task_id in tasks:
        result = run_task(args.url, task_id, client, seed=args.seed, verbose=verbose)
        all_scores[task_id] = result.get("final_score", 0.0)

    if len(all_scores) > 1 and verbose:
        print(f"\n{'='*50}", flush=True)
        print(f"BASELINE SCORES (model={MODEL_NAME}, seed={args.seed}):", flush=True)
        for k, v in all_scores.items():
            print(f"  {k}: {v:.4f}", flush=True)
        print(f"  mean: {sum(all_scores.values())/len(all_scores):.4f}", flush=True)

if __name__ == "__main__":
    main()
