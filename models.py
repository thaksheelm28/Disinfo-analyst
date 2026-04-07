"""
models.py — Typed Pydantic models for Observation, Action, Reward, and internal graph types.

When pydantic is installed (Docker / HF Spaces) all models are full Pydantic
BaseModels with validation. When absent (offline CI) they fall back to
lightweight dataclasses with identical field interfaces — all tests pass in
both modes.
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Any

try:
    from pydantic import BaseModel, Field
    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return self.__dict__.copy()
    def Field(default=None, **_kw):
        return default

# ── Typed Action models ─────────────────────────────────────────────────────

class TagAction(BaseModel):
    type: str = "tag"
    node_id: str = Field(..., description="Node ID to label")
    label: str   = Field(..., description="real|disinformation|satire|misleading|unverified")

class TraceAction(BaseModel):
    type: str = "trace"
    src: str  = Field(..., description="Upstream node ID")
    dst: str  = Field(..., description="Downstream node ID")

class ScoreAction(BaseModel):
    type: str     = "score"
    claim_id: str = Field(..., description="Claim ID")
    severity: float = Field(..., description="Severity [0.0–1.0]")

class DoneAction(BaseModel):
    type: str = "done"

Action = TagAction | TraceAction | ScoreAction | DoneAction

# ── Typed Observation model ─────────────────────────────────────────────────

class NodeObservation(BaseModel):
    node_id: str
    kind: str
    text: str
    timestamp: str
    source: str
    share_count: int

class EdgeObservation(BaseModel):
    src: str
    dst: str
    relation: str

class ClaimObservation(BaseModel):
    claim_id: str
    text: str

class Observation(BaseModel):
    nodes:           list = Field(default_factory=list)
    edges:           list = Field(default_factory=list)
    claims:          list = Field(default_factory=list)
    agent_tags:      dict = Field(default_factory=dict)
    agent_traces:    list = Field(default_factory=list)
    agent_scores:    dict = Field(default_factory=dict)
    steps_taken:     int  = 0
    steps_remaining: int  = 50
    task:            str  = "task1_classify"

# ── Typed Reward model ──────────────────────────────────────────────────────

class Reward(BaseModel):
    """
    Emitted at EVERY step — not just episode end.
    Provides partial progress signal throughout the trajectory.
    """
    step_reward:    float = Field(0.0, description="Immediate action reward")
    tag_progress:   float = Field(0.0, description="Running tag F1 [0,1]")
    trace_progress: float = Field(0.0, description="Running trace score [0,1]")
    score_progress: float = Field(0.0, description="Running severity score [0,1]")
    cumulative:     float = Field(0.0, description="Weighted running total [0,1]")

    @classmethod
    def from_step(cls, step_reward: float, eval_result: dict) -> "Reward":
        return cls(
            step_reward    = step_reward,
            tag_progress   = eval_result.get("tag_f1", 0.0),
            trace_progress = eval_result.get("trace_score", 0.0),
            score_progress = eval_result.get("severity_score", 0.0),
            cumulative     = eval_result.get("final_score", 0.0),
        )

# ── EvaluationResult ────────────────────────────────────────────────────────

class GradingBreakdown(BaseModel):
    tag_weight:      float = 0.40
    trace_weight:    float = 0.35
    severity_weight: float = 0.25

class EvaluationResult(BaseModel):
    tag_precision:  float = 0.0
    tag_recall:     float = 0.0
    tag_f1:         float = 0.0
    trace_score:    float = 0.0
    trace_details:  list  = Field(default_factory=list)
    severity_rmse:  float = 0.0
    severity_score: float = 0.0
    final_score:    float = 0.0
    breakdown:      Any   = None
    task:           str   = "task1_classify"

# ── StepResult ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Any  = None
    reward:      Any  = None
    done:        bool = False
    info:        dict = Field(default_factory=dict)

# ── Internal graph dataclasses (never exposed to agent) ─────────────────────

@dataclass
class Node:
    node_id: str
    kind: str
    text: str
    timestamp: str
    source: str
    credibility: float
    share_count: int
    _true_label: str  = dc_field(default="real", repr=False)
    _is_origin: bool  = dc_field(default=False, repr=False)

@dataclass
class Edge:
    src: str
    dst: str
    relation: str

@dataclass
class Claim:
    claim_id: str
    text: str
    _true_severity: float = dc_field(default=0.0, repr=False)
