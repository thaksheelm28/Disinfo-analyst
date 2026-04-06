"""
app.py — FastAPI server for Disinformation Analyst (OpenEnv)

Endpoints (OpenEnv spec + required extras):
  POST /reset       → Observation           (start / restart episode)
  POST /step        → StepResult            (execute one action)
  GET  /state       → Observation           (read-only current state)
  GET  /evaluate    → EvaluationResult      (grade the current episode)
  GET  /tasks       → task list + action schema
  GET  /health      → {"status": "ok"}
  POST /baseline    → trigger dry-run baseline, return scores for all 3 tasks
  GET  /grader      → grader score for the current episode

Hugging Face Spaces: port 7860
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from disinfo_env import DisinfoEnv, TASKS

# ---------------------------------------------------------------------------
# Pydantic request/response models for the HTTP layer
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "task1_classify"
    scenario: Optional[str] = None
    seed: Optional[int] = 42
    max_steps: int = 50


class StepRequest(BaseModel):
    action: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.2.0"
    environment: str = "disinfo-analyst"


# ---------------------------------------------------------------------------
# Per-request environment — simple in-memory singleton (one session)
# We keep a global env so the validator's sequential reset→step→evaluate works.
# ---------------------------------------------------------------------------

_env_lock = threading.Lock()
_env: DisinfoEnv = DisinfoEnv(seed=42)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: pre-load a default episode so /health also checks env health
    global _env
    with _env_lock:
        _env = DisinfoEnv(seed=42)
        _env.reset(task="task1_classify", seed=42)
    yield


app = FastAPI(
    title="Disinformation Analyst — OpenEnv",
    description=(
        "Three-task OpenEnv environment: node classification (easy), "
        "propagation tracing (medium), and claim severity rating (hard)."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: serialize Pydantic / dataclass → plain dict
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: _to_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness check — must return 200 for HF Space validation."""
    return HealthResponse()


# ---------------------------------------------------------------------------
# /reset   POST
# ---------------------------------------------------------------------------

@app.post("/reset", tags=["openenv"])
def reset(req: ResetRequest = ResetRequest()) -> JSONResponse:
    """
    Reset the environment and return the initial Observation.

    Body (all optional):
      task      : "task1_classify" | "task2_trace" | "task3_severity"
      scenario  : "easy" | "default" | "hard" | "adversarial"
      seed      : int (default 42 — for reproducibility)
      max_steps : int (default 50)
    """
    global _env
    if req.task not in TASKS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{req.task}'. Valid: {list(TASKS)}",
        )
    with _env_lock:
        _env = DisinfoEnv(seed=req.seed, max_steps=req.max_steps)
        obs = _env.reset(task=req.task, scenario=req.scenario, seed=req.seed)
    return JSONResponse(content=_to_dict(obs))


# ---------------------------------------------------------------------------
# /step    POST
# ---------------------------------------------------------------------------

@app.post("/step", tags=["openenv"])
def step(req: StepRequest) -> JSONResponse:
    """
    Execute one action. Returns StepResult {observation, reward, done, info}.

    Action types:
      {"type": "tag",   "node_id": "...", "label": "disinformation"}
      {"type": "trace", "src": "...", "dst": "..."}
      {"type": "score", "claim_id": "...", "severity": 0.85}
      {"type": "done"}
    """
    with _env_lock:
        if _env.done:
            raise HTTPException(
                status_code=409,
                detail="Episode is finished. Call /reset to start a new episode.",
            )
        try:
            result = _env.step(req.action)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    return JSONResponse(content=_to_dict(result))


# ---------------------------------------------------------------------------
# /state   GET
# ---------------------------------------------------------------------------

@app.get("/state", tags=["openenv"])
def state() -> JSONResponse:
    """Return current Observation without advancing the environment."""
    with _env_lock:
        obs = _env.state()
    return JSONResponse(content=_to_dict(obs))


# ---------------------------------------------------------------------------
# /evaluate   GET
# ---------------------------------------------------------------------------

@app.get("/evaluate", tags=["openenv"])
def evaluate() -> JSONResponse:
    """
    Return the full graded EvaluationResult for the current episode.
    Scores are deterministic given the same actions + seed.
    """
    with _env_lock:
        result = _env.evaluate()
    return JSONResponse(content=_to_dict(result))


# ---------------------------------------------------------------------------
# /tasks   GET  (REQUIRED by spec)
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["openenv"])
def tasks() -> JSONResponse:
    """
    Return all task definitions and the action schema.

    Satisfies the mandatory /tasks endpoint from the submission spec.
    """
    action_schema = {
        "TagAction": {
            "type": "tag",
            "description": "Assign a disinformation label to a graph node.",
            "fields": {
                "node_id": {"type": "str", "description": "Node ID from Observation.nodes"},
                "label": {
                    "type": "enum",
                    "values": ["real", "disinformation", "satire", "misleading", "unverified"],
                },
            },
            "example": {"type": "tag", "node_id": "art_3", "label": "disinformation"},
        },
        "TraceAction": {
            "type": "trace",
            "description": "Assert that node `src` is an upstream source of node `dst`.",
            "fields": {
                "src": {"type": "str", "description": "Upstream / origin node ID"},
                "dst": {"type": "str", "description": "Downstream / recipient node ID"},
            },
            "example": {"type": "trace", "src": "seed_0", "dst": "art_2"},
        },
        "ScoreAction": {
            "type": "score",
            "description": "Assign a float severity score [0.0–1.0] to a claim.",
            "fields": {
                "claim_id": {"type": "str", "description": "Claim ID from Observation.claims"},
                "severity": {"type": "float", "min": 0.0, "max": 1.0},
            },
            "example": {"type": "score", "claim_id": "claim_0", "severity": 0.87},
        },
        "DoneAction": {
            "type": "done",
            "description": "Signal that the agent has finished the episode.",
            "fields": {},
            "example": {"type": "done"},
        },
    }

    return JSONResponse(content={
        "tasks": {
            tid: {
                **tcfg,
                "action_schema": action_schema,
            }
            for tid, tcfg in TASKS.items()
        },
        "action_schema": action_schema,
        "note": (
            "Pass any action dict to POST /step. "
            "Start a task with POST /reset?task=<task_id>."
        ),
    })


# ---------------------------------------------------------------------------
# /grader   GET  (REQUIRED by spec)
# ---------------------------------------------------------------------------

@app.get("/grader", tags=["openenv"])
def grader() -> JSONResponse:
    """
    Return the grader score for the current episode.

    Returns the full EvaluationResult including per-metric breakdown.
    Deterministic — same actions + seed always produce the same scores.
    """
    with _env_lock:
        result = _env.evaluate()
        task = _env._task
    data = _to_dict(result)
    data["task"] = task
    data["grader_info"] = {
        "tag_weight": 0.40,
        "trace_weight": 0.35,
        "severity_weight": 0.25,
        "formula": "final_score = 0.40*tag_f1 + 0.35*trace_score + 0.25*severity_score",
        "partial_credit": {
            "backwards_trace": 0.25,
            "invalid_action_penalty": -0.05,
            "unscored_claim_default": 0.5,
        },
    }
    return JSONResponse(content=data)


# ---------------------------------------------------------------------------
# /baseline   POST  (REQUIRED by spec)
# ---------------------------------------------------------------------------

@app.post("/baseline", tags=["openenv"])
def baseline(seed: int = 42) -> JSONResponse:
    """
    Run the built-in dry-run baseline agent against all 3 tasks and return scores.

    This is a deterministic oracle-adjacent agent that:
      - Task 1: tags ALL nodes as 'disinformation' (tests recall)
      - Task 2: traces all visible edges (tests path correctness)
      - Task 3: scores all claims at 0.5 (maximum uncertainty baseline)

    Returns per-task scores and a mean score.
    Satisfies the mandatory /baseline endpoint.
    """
    scores: Dict[str, float] = {}
    details: Dict[str, Any] = {}

    task_list = ["task1_classify", "task2_trace", "task3_severity"]

    for task_id in task_list:
        env = DisinfoEnv(seed=seed, max_steps=200)
        obs = env.reset(task=task_id, seed=seed)

        if task_id == "task1_classify":
            # Baseline: tag every node as disinformation
            for node in obs.nodes:
                env.step({"type": "tag", "node_id": node.node_id, "label": "disinformation"})

        elif task_id == "task2_trace":
            # Baseline: assert all observed edges
            for edge in obs.edges:
                env.step({"type": "trace", "src": edge.src, "dst": edge.dst})

        elif task_id == "task3_severity":
            # Baseline: score every claim at 0.5 (maximum uncertainty)
            for claim in obs.claims:
                env.step({"type": "score", "claim_id": claim.claim_id, "severity": 0.5})

        env.step({"type": "done"})
        result = env.evaluate()
        scores[task_id] = round(result.final_score, 4)
        details[task_id] = {
            "tag_f1": result.tag_f1,
            "trace_score": result.trace_score,
            "severity_score": result.severity_score,
            "final_score": result.final_score,
            "strategy": {
                "task1_classify": "Tag all nodes as 'disinformation'",
                "task2_trace":    "Assert all observed edges",
                "task3_severity": "Score all claims at 0.5 (max uncertainty)",
            }[task_id],
        }

    mean_score = round(sum(scores.values()) / len(scores), 4)

    return JSONResponse(content={
        "scores": scores,
        "mean_score": mean_score,
        "seed": seed,
        "details": details,
        "model": "dry-run-baseline",
        "note": (
            "Baseline uses a deterministic oracle-adjacent agent. "
            "Run inference.py with a real LLM for a meaningful benchmark."
        ),
    })


# ---------------------------------------------------------------------------
# Root  GET
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root() -> JSONResponse:
    return JSONResponse(content={
        "environment": "disinfo-analyst",
        "version": "0.2.0",
        "description": (
            "Three-task OpenEnv environment for disinformation analysis. "
            "Tasks: node classification (easy), propagation tracing (medium), "
            "claim severity rating (hard)."
        ),
        "endpoints": {
            "POST /reset":    "Start a new episode",
            "POST /step":     "Execute one action",
            "GET  /state":    "Read current observation (no side effects)",
            "GET  /evaluate": "Get graded EvaluationResult",
            "GET  /tasks":    "List tasks + action schema",
            "GET  /grader":   "Get grader score for current episode",
            "POST /baseline": "Run dry-run baseline on all 3 tasks",
            "GET  /health":   "Liveness check",
        },
        "tasks": list(TASKS.keys()),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
