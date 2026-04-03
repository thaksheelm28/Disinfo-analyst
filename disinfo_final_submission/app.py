"""
app.py — FastAPI server for Disinformation Analyst OpenEnv

Endpoints:
  POST /reset         → reset for a task, return Observation
  POST /step          → execute one action, return StepResult
  GET  /state         → current Observation (read-only)
  GET  /evaluate      → EvaluationResult
  GET  /tasks         → list all available tasks + action schema
  GET  /health        → liveness probe
  GET  /baseline      → run dry-run baseline agent on all 3 tasks, return scores
  GET  /grader        → return grader score for the current episode
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from disinfo_env import DisinfoEnv, TASKS
from models import Observation, StepResult, EvaluationResult

# ---------------------------------------------------------------------------
app = FastAPI(
    title="Disinformation Analyst — OpenEnv",
    description=(
        "Three-task LLM evaluation environment: "
        "node classification (easy), propagation tracing (medium), "
        "claim severity rating (hard)."
    ),
    version="0.2.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env: DisinfoEnv | None = None

# ---------------------------------------------------------------------------
# Action schema — field definitions for every action type
# ---------------------------------------------------------------------------

ACTION_SCHEMA = {
    "tag": {
        "description": "Label a graph node with a disinformation category.",
        "fields": {
            "type":    {"type": "str", "const": "tag"},
            "node_id": {"type": "str", "description": "ID of the node to label"},
            "label":   {
                "type": "enum",
                "values": ["real", "disinformation", "satire", "misleading", "unverified"],
                "description": "Disinformation label to assign",
            },
        },
        "example": {"type": "tag", "node_id": "article_0", "label": "disinformation"},
    },
    "trace": {
        "description": "Assert that node `src` is an upstream propagation source of node `dst`.",
        "fields": {
            "type": {"type": "str", "const": "trace"},
            "src":  {"type": "str", "description": "Upstream (origin) node ID"},
            "dst":  {"type": "str", "description": "Downstream (spread) node ID"},
        },
        "example": {"type": "trace", "src": "seed_0", "dst": "article_2"},
    },
    "score": {
        "description": "Assign a real-world harm severity [0.0–1.0] to a factual claim.",
        "fields": {
            "type":      {"type": "str",   "const": "score"},
            "claim_id":  {"type": "str",   "description": "ID of the claim to score"},
            "severity":  {"type": "float", "min": 0.0, "max": 1.0,
                          "description": "Harm severity (0=harmless, 1=extremely harmful)"},
        },
        "example": {"type": "score", "claim_id": "claim_0", "severity": 0.8},
    },
    "done": {
        "description": "Signal that the agent has finished all actions for this episode.",
        "fields": {
            "type": {"type": "str", "const": "done"},
        },
        "example": {"type": "done"},
    },
}

# ---------------------------------------------------------------------------

def _require_env() -> DisinfoEnv:
    if _env is None:
        raise HTTPException(400, "Environment not initialised. Call POST /reset first.")
    return _env

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task:      str       = "task1_classify"
    scenario:  str | None = None
    seed:      int | None = None
    max_steps: int        = 50

class StepRequest(BaseModel):
    action: dict[str, Any]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions, difficulty, and action schema."""
    return {
        "tasks": TASKS,
        "action_schema": ACTION_SCHEMA,
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = ResetRequest()):
    global _env
    _env = DisinfoEnv(seed=req.seed, max_steps=req.max_steps)
    return _env.reset(task=req.task, scenario=req.scenario, seed=req.seed)


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    env = _require_env()
    if env.done:
        raise HTTPException(400, "Episode finished. Call POST /reset.")
    try:
        return env.step(req.action)
    except Exception as e:
        raise HTTPException(422, str(e))


@app.get("/state", response_model=Observation)
def state():
    """Return current observation without advancing the environment."""
    return _require_env().state()


@app.get("/evaluate", response_model=EvaluationResult)
def evaluate():
    """Return the full graded result for the current episode."""
    return _require_env().evaluate()


@app.get("/grader")
def grader():
    """
    Return grader score after an episode is completed (or in-progress).
    Alias for /evaluate — returns per-task sub-scores and final composite score.
    """
    result = _require_env().evaluate()
    # Return as plain dict so the validator can consume it without the full model
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result.__dict__


@app.get("/baseline")
def baseline():
    """
    Trigger the dry-run baseline agent against all 3 tasks (no API key needed).
    Returns reproducible baseline scores using seed=42.
    """
    SEED = 42
    env  = DisinfoEnv(seed=SEED)
    results: dict[str, Any] = {}

    task_configs = {
        "task1_classify": lambda obs, e: [
            e.step({"type": "tag", "node_id": n.node_id, "label": "disinformation"})
            for n in obs.nodes
        ],
        "task2_trace": lambda obs, e: (
            [e.step({"type": "trace",
                     "src": obs.nodes[0].node_id,
                     "dst": obs.nodes[1].node_id})]
            if len(obs.nodes) >= 2 else []
        ) if obs.nodes else [],
        "task3_severity": lambda obs, e: [
            e.step({"type": "score", "claim_id": c.claim_id, "severity": 0.5})
            for c in obs.claims
        ],
    }

    for task_id, run_steps in task_configs.items():
        obs = env.reset(task=task_id, seed=SEED)
        run_steps(obs, env)
        env.step({"type": "done"})
        r = env.evaluate()
        results[task_id] = {
            "final_score":     r.final_score,
            "tag_f1":          r.tag_f1,
            "trace_score":     r.trace_score,
            "severity_score":  r.severity_score,
        }

    scores = [v["final_score"] for v in results.values()]
    return {
        "seed":    SEED,
        "tasks":   results,
        "mean_score": round(sum(scores) / len(scores), 4),
    }
