"""
inference.py — Baseline LLM agent for Disinformation Analyst (OpenEnv)

Runs a model against all three tasks and reports reproducible baseline scores.

Usage:
    python inference.py                         # all 3 tasks, default scenario
    python inference.py --task task1_classify   # single task
    python inference.py --url http://host:7860  # remote server
    python inference.py --dry-run               # local, no API key needed
    python inference.py --seed 42               # reproducible scores

Credentials:
    export HF_TOKEN=sk-...
    export MODEL_NAME=gpt-4o-mini      # optional, default gpt-4o-mini
    export API_BASE_URL=https://...    # optional, default http://localhost:7860
"""

from __future__ import annotations
import argparse, json, os, sys, textwrap
import requests
from openai import OpenAI

DEFAULT_MODEL    = os.getenv("MODEL_NAME", "gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MAX_AGENT_STEPS  = 60

# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "task1_classify": textwrap.dedent("""
    You are an expert disinformation analyst. Your ONLY task right now is to
    classify every node in the graph with a label.

    Valid labels: real | disinformation | satire | misleading | unverified

    Signals to use:
    - Source domain (globalfreemind.net, truthwatcher.io → likely fake)
    - Share count (viral spread → check carefully)
    - Text content (sensational claims, conspiracy language)
    - Timestamp (early seeds often originate disinformation)

    Reply with ONLY a single JSON action per turn:
      {"type": "tag", "node_id": "<id>", "label": "<label>"}
    When all nodes are tagged, reply: {"type": "done"}
    """).strip(),

    "task2_trace": textwrap.dedent("""
    You are an expert disinformation analyst. Your ONLY task is to trace
    propagation chains: identify which nodes are upstream sources of others.

    Use the edges (shares/paraphrases/quotes) and timestamps to determine
    who originated content and how it spread. Earlier timestamps + low
    credibility sources are likely origins.

    Reply with ONLY a single JSON action per turn:
      {"type": "trace", "src": "<origin_id>", "dst": "<downstream_id>"}
    When done tracing, reply: {"type": "done"}
    """).strip(),

    "task3_severity": textwrap.dedent("""
    You are an expert disinformation analyst. Your ONLY task is to assign a
    severity score [0.0–1.0] to each factual claim based on potential
    real-world harm.

    0.0 = completely harmless (local news, weather)
    0.5 = moderately harmful (misleading but not dangerous)
    1.0 = extremely harmful (health misinformation, incitement)

    Reply with ONLY a single JSON action per turn:
      {"type": "score", "claim_id": "<id>", "severity": <0.0–1.0>}
    When all claims are scored, reply: {"type": "done"}
    """).strip(),
}

# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------

def _post(url, path, body):
    r = requests.post(f"{url}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()

def _get(url, path):
    r = requests.get(f"{url}{path}", timeout=30)
    r.raise_for_status()
    return r.json()

def reset_env(url, task, scenario=None, seed=None):
    body = {"task": task}
    if scenario: body["scenario"] = scenario
    if seed is not None: body["seed"] = seed
    return _post(url, "/reset", body)

def step_env(url, action):
    return _post(url, "/step", {"action": action})

def get_state(url):
    return _get(url, "/state")

def evaluate_env(url):
    return _get(url, "/evaluate")

# ---------------------------------------------------------------------------
# Observation → readable prompt
# ---------------------------------------------------------------------------

def build_prompt(obs: dict, task: str) -> str:
    lines = [f"=== TASK: {task} ===\n"]

    lines.append("NODES:")
    for n in obs.get("nodes", []):
        lines.append(
            f"  [{n['node_id']}] ({n['kind']}) {n['source']} "
            f"shares={n['share_count']} ts={n['timestamp']}\n"
            f"    {n['text'][:120]}"
        )

    lines.append("\nEDGES:")
    for e in obs.get("edges", []):
        lines.append(f"  {e['src']} --{e['relation']}--> {e['dst']}")

    lines.append("\nCLAIMS:")
    for c in obs.get("claims", []):
        lines.append(f"  [{c['claim_id']}] {c['text']}")

    if obs.get("agent_tags"):
        lines.append(f"\nTagged so far: {obs['agent_tags']}")
    if obs.get("agent_traces"):
        lines.append(f"Traced so far: {obs['agent_traces']}")
    if obs.get("agent_scores"):
        lines.append(f"Scored so far: {obs['agent_scores']}")

    lines.append(f"\nSteps remaining: {obs.get('steps_remaining', '?')}")
    lines.append("\nReply with ONE JSON action object only.")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Parse LLM output → action dict
# ---------------------------------------------------------------------------

def parse_action(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1][4:] if parts[1].startswith("json") else parts[1]
    try:
        return json.loads(raw)
    except Exception:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s != -1 and e > s:
            try: return json.loads(raw[s:e])
            except: pass
    return None

# ---------------------------------------------------------------------------
# Single-task agent loop
# ---------------------------------------------------------------------------

def run_task(url, task, model, client, scenario=None, seed=None, verbose=True):
    obs = reset_env(url, task, scenario=scenario, seed=seed)
    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"Nodes: {len(obs.get('nodes',[]))}  Claims: {len(obs.get('claims',[]))}")
        print(f"{'='*60}")

    system = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["task1_classify"])
    messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": build_prompt(obs, task)},
    ]
    done, step = False, 0

    while not done and step < MAX_AGENT_STEPS:
        step += 1
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.1, max_tokens=200,
        )
        raw = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": raw})

        action = parse_action(raw)
        if action is None:
            messages.append({"role": "user", "content":
                "Could not parse your JSON. Reply with ONLY a JSON object."})
            continue

        if verbose:
            print(f"  [step {step:2d}] {raw.strip()[:80]}")

        result = step_env(url, action)
        obs    = result.get("observation", result)  # handle both dict shapes
        done   = result.get("done", False)

        # Show cumulative reward at every step
        rwd = result.get("reward", {})
        if verbose and isinstance(rwd, dict):
            print(f"           → cumulative={rwd.get('cumulative',0):.3f} "
                  f"tag={rwd.get('tag_progress',0):.2f} "
                  f"trace={rwd.get('trace_progress',0):.2f} "
                  f"severity={rwd.get('score_progress',0):.2f}")

        if not done:
            messages.append({"role": "user", "content": build_prompt(obs, task)})

    final = evaluate_env(url)
    if verbose:
        print(f"\n  Final score: {final.get('final_score', '?'):.4f}")
        print(f"  tag_f1={final.get('tag_f1','?')}  "
              f"trace={final.get('trace_score','?')}  "
              f"severity={final.get('severity_score','?')}")
    return final

# ---------------------------------------------------------------------------
# Dry-run (local, no server, no API key)
# ---------------------------------------------------------------------------

def run_dry_run(scenario=None, seed=42, verbose=True):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from disinfo_env import DisinfoEnv

    env = DisinfoEnv(seed=seed)
    results = {}

    for task_id in ["task1_classify", "task2_trace", "task3_severity"]:
        obs = env.reset(task=task_id, scenario=scenario)
        if verbose:
            print(f"\n[Dry-run] {task_id}  nodes={len(obs.nodes)}  claims={len(obs.claims)}")

        if task_id == "task1_classify":
            for n in obs.nodes:
                env.step({"type":"tag","node_id":n.node_id,"label":"disinformation"})
        elif task_id == "task2_trace":
            nodes = obs.nodes
            if len(nodes) >= 2:
                env.step({"type":"trace","src":nodes[0].node_id,"dst":nodes[1].node_id})
        elif task_id == "task3_severity":
            for c in obs.claims:
                env.step({"type":"score","claim_id":c.claim_id,"severity":0.5})

        env.step({"type":"done"})
        r = env.evaluate()
        results[task_id] = r.final_score
        if verbose:
            print(f"  Score: {r.final_score:.4f}  "
                  f"(tag_f1={r.tag_f1}  trace={r.trace_score}  severity={r.severity_score})")

    if verbose:
        print(f"\n{'='*40}")
        print(f"Baseline scores (seed={seed}):")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print(f"  mean: {sum(results.values())/len(results):.4f}")
    return results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DisinfoEnv baseline agent")
    parser.add_argument("--task", default=None,
                        choices=["task1_classify","task2_trace","task3_severity"])
    parser.add_argument("--scenario", default=None,
                        choices=["easy","default","hard","adversarial"])
    parser.add_argument("--url",   default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.dry_run:
        run_dry_run(scenario=args.scenario, seed=args.seed, verbose=verbose)
        return

    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        print("ERROR: Set HF_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    base_url = os.getenv("API_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    tasks  = [args.task] if args.task else ["task1_classify","task2_trace","task3_severity"]
    all_scores = {}

    for task_id in tasks:
        result = run_task(
            args.url, task_id, args.model, client,
            scenario=args.scenario, seed=args.seed, verbose=verbose,
        )
        all_scores[task_id] = result.get("final_score", 0.0)

    if len(all_scores) > 1:
        print(f"\n{'='*50}")
        print(f"BASELINE SCORES (model={args.model}, seed={args.seed}):")
        for k, v in all_scores.items():
            print(f"  {k}: {v:.4f}")
        print(f"  mean: {sum(all_scores.values())/len(all_scores):.4f}")

if __name__ == "__main__":
    main()
