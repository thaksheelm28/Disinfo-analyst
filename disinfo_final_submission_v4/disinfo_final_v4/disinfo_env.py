"""
disinfo_env.py — Disinformation Analyst Environment (OpenEnv compliant)

Three distinct graded tasks an agent must solve, ranging easy → medium → hard:

  Task 1 (easy)   — NODE CLASSIFICATION
    Tag every node with the correct disinformation label.
    Graded by F1 of 'disinformation' predictions.

  Task 2 (medium) — PROPAGATION TRACING
    Trace the upstream source chain: identify which nodes are origins/spreaders.
    Graded by BFS path-existence score over claimed (src→dst) pairs.

  Task 3 (hard)   — CLAIM SEVERITY RATING
    Assign float severity [0.0–1.0] to each factual claim.
    Graded by 1 – normalised RMSE against ground-truth severities.

Each task has its own reset()/step()/state()/evaluate() cycle and its own
grader that emits a typed Reward with partial-progress sub-scores at every
step — not just at episode end.

OpenEnv API:
  env.reset(task, scenario, seed)  → Observation
  env.step(action)                 → StepResult(observation, reward, done, info)
  env.state()                      → Observation  (read-only)
  env.evaluate()                   → EvaluationResult
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from models import (
    Node, Edge, Claim,
    Observation, NodeObservation, EdgeObservation, ClaimObservation,
    Action, TagAction, TraceAction, ScoreAction, DoneAction,
    Reward, StepResult, EvaluationResult, GradingBreakdown,
)
from graph_factory import build_scenario_graph
from grader import grade


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "task1_classify": {
        "name":        "Node Classification (Easy)",
        "description": (
            "Tag every graph node with its correct disinformation label. "
            "Focuses purely on content understanding — no graph reasoning needed."
        ),
        "difficulty":  "easy",
        "primary_metric": "tag_f1",
    },
    "task2_trace": {
        "name":        "Propagation Tracing (Medium)",
        "description": (
            "Identify upstream source relationships: assert which nodes "
            "are origins or spreaders of other nodes. Requires multi-hop "
            "graph reasoning over timestamps and propagation edges."
        ),
        "difficulty":  "medium",
        "primary_metric": "trace_score",
    },
    "task3_severity": {
        "name":        "Claim Severity Rating (Hard)",
        "description": (
            "Assign a float severity score [0.0–1.0] to each factual claim. "
            "Requires nuanced judgment about real-world harm potential — "
            "no binary answer exists."
        ),
        "difficulty":  "hard",
        "primary_metric": "severity_score",
    },
}

VALID_LABELS = {"real", "disinformation", "satire", "misleading", "unverified"}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DisinfoEnv:
    """
    OpenEnv-compatible environment with three distinct graded tasks.

    Quick-start (single task):
        env = DisinfoEnv()
        obs = env.reset(task="task1_classify", scenario="easy")
        while not env.done:
            result = env.step(action)   # result.reward has partial progress
        final = env.evaluate()

    Run all three tasks:
        for task_id in ["task1_classify", "task2_trace", "task3_severity"]:
            obs = env.reset(task=task_id)
            ...
            print(env.evaluate().final_score)
    """

    def __init__(self, seed: int | None = None, max_steps: int = 50):
        self._seed     = seed
        self.max_steps = max_steps
        self._rng      = random.Random(seed)
        # Internal state — populated by reset()
        self._nodes:        dict[str, Node]  = {}
        self._edges:        list[Edge]       = []
        self._claims:       dict[str, Claim] = {}
        self._agent_tags:   dict[str, str]   = {}
        self._agent_traces: list[tuple[str, str]] = []
        self._agent_scores: dict[str, float] = {}
        self._done       = True
        self._step_count = 0
        self._task       = "task1_classify"
        self._scenario   = "easy"

    # ------------------------------------------------------------------
    # OpenEnv Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        task:     str = "task1_classify",
        scenario: str | None = None,
        seed:     int | None = None,
    ) -> Observation:
        """
        Reset the environment for a given task and return the initial observation.

        task     : "task1_classify" | "task2_trace" | "task3_severity"
        scenario : "easy" | "default" | "hard" | "adversarial"
                   If None, auto-selects based on task difficulty.
        seed     : RNG seed for reproducibility.
        """
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASKS)}")

        # Auto-pick scenario from task difficulty if not specified
        if scenario is None:
            difficulty = TASKS[task]["difficulty"]
            scenario = {"easy": "easy", "medium": "default", "hard": "hard"}[difficulty]

        self._task     = task
        self._scenario = scenario

        if seed is not None:
            self._seed = seed
            self._rng  = random.Random(seed)

        data_path = Path(__file__).parent / "scenarios.json"
        with open(data_path) as f:
            scenarios = json.load(f)
        cfg = scenarios.get(scenario) or scenarios["default"]

        graph = build_scenario_graph(cfg, rng=self._rng)
        self._nodes  = {n.node_id: n for n in graph["nodes"]}
        self._edges  = graph["edges"]
        self._claims = {c.claim_id: c for c in graph["claims"]}

        self._agent_tags   = {}
        self._agent_traces = []
        self._agent_scores = {}
        self._done         = False
        self._step_count   = 0

        return self._observation()

    def step(self, action: dict | Action) -> StepResult:
        """
        Execute one action. Returns StepResult(observation, reward, done, info).

        reward.step_reward    — immediate action reward
        reward.tag_progress   — running tag F1 (partial signal at every step)
        reward.trace_progress — running trace score (partial signal)
        reward.score_progress — running severity score (partial signal)
        reward.cumulative     — weighted running total
        """
        if self._done:
            raise RuntimeError("Episode finished — call reset() to start a new one.")

        # Parse dict → typed action
        if isinstance(action, dict):
            action = _parse_action(action)

        self._step_count += 1
        step_reward, info = self._dispatch(action)

        done = self._done or self._step_count >= self.max_steps
        if done and not self._done:
            self._done = True

        # Compute partial-progress reward at every step
        eval_result = grade(
            nodes=self._nodes, edges=self._edges, claims=self._claims,
            agent_tags=self._agent_tags, agent_traces=self._agent_traces,
            agent_scores=self._agent_scores,
        )

        reward = Reward.from_step(step_reward, eval_result)

        return StepResult(
            observation=self._observation(),
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> Observation:
        """
        Return the current observation WITHOUT advancing the environment.
        Read-only — no side effects, no step counter increment.
        """
        return self._observation()

    def evaluate(self) -> EvaluationResult:
        """Return the full typed graded result for the current episode."""
        result = grade(
            nodes=self._nodes, edges=self._edges, claims=self._claims,
            agent_tags=self._agent_tags, agent_traces=self._agent_traces,
            agent_scores=self._agent_scores,
        )
        return EvaluationResult(
            tag_precision=result["tag_precision"],
            tag_recall=result["tag_recall"],
            tag_f1=result["tag_f1"],
            trace_score=result["trace_score"],
            trace_details=result["trace_details"],
            severity_rmse=result["severity_rmse"],
            severity_score=result["severity_score"],
            final_score=result["final_score"],
            breakdown=GradingBreakdown(**result["breakdown"]),
            task=self._task,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observation(self) -> Observation:
        return Observation(
            nodes=[
                NodeObservation(
                    node_id=n.node_id, kind=n.kind, text=n.text,
                    timestamp=n.timestamp, source=n.source, share_count=n.share_count,
                )
                for n in self._nodes.values()
            ],
            edges=[
                EdgeObservation(src=e.src, dst=e.dst, relation=e.relation)
                for e in self._edges
            ],
            claims=[
                ClaimObservation(claim_id=c.claim_id, text=c.text)
                for c in self._claims.values()
            ],
            agent_tags=copy.copy(self._agent_tags),
            agent_traces=[list(t) for t in self._agent_traces],
            agent_scores=copy.copy(self._agent_scores),
            steps_taken=self._step_count,
            steps_remaining=max(0, self.max_steps - self._step_count),
            task=self._task,
        )

    def _dispatch(self, action: Action) -> tuple[float, dict]:
        if isinstance(action, TagAction):
            return self._act_tag(action)
        elif isinstance(action, TraceAction):
            return self._act_trace(action)
        elif isinstance(action, ScoreAction):
            return self._act_score(action)
        elif isinstance(action, DoneAction):
            self._done = True
            return 0.0, {"message": "Episode ended by agent."}
        else:
            return -0.1, {"error": f"Unknown action type."}

    def _act_tag(self, action: TagAction) -> tuple[float, dict]:
        if action.node_id not in self._nodes:
            return -0.05, {"error": f"Unknown node_id '{action.node_id}'."}
        if action.label not in VALID_LABELS:
            return -0.05, {"error": f"Invalid label '{action.label}'. Choose from {VALID_LABELS}."}
        self._agent_tags[action.node_id] = action.label
        return 0.0, {"tagged": action.node_id, "label": action.label}

    def _act_trace(self, action: TraceAction) -> tuple[float, dict]:
        if action.src not in self._nodes:
            return -0.05, {"error": f"Unknown src node '{action.src}'."}
        if action.dst not in self._nodes:
            return -0.05, {"error": f"Unknown dst node '{action.dst}'."}
        self._agent_traces.append((action.src, action.dst))
        return 0.0, {"traced": f"{action.src} → {action.dst}"}

    def _act_score(self, action: ScoreAction) -> tuple[float, dict]:
        if action.claim_id not in self._claims:
            return -0.05, {"error": f"Unknown claim_id '{action.claim_id}'."}
        if not isinstance(action.severity, (int, float)) or not (0.0 <= float(action.severity) <= 1.0):
            return -0.05, {"error": "severity must be a float in [0.0, 1.0]."}
        self._agent_scores[action.claim_id] = float(action.severity)
        return 0.0, {"scored": action.claim_id, "severity": action.severity}

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._done

    @property
    def task_info(self) -> dict:
        return TASKS.get(self._task, {})

    def __repr__(self) -> str:
        return (
            f"DisinfoEnv(task={self._task!r}, scenario={self._scenario!r}, "
            f"nodes={len(self._nodes)}, steps={self._step_count}/{self.max_steps})"
        )


# ---------------------------------------------------------------------------
# Action parser: dict → typed Pydantic action
# ---------------------------------------------------------------------------

def _parse_action(d: dict) -> Action:
    t = d.get("type", "")
    if t == "tag":
        return TagAction(**d)
    elif t == "trace":
        return TraceAction(**d)
    elif t == "score":
        return ScoreAction(**d)
    elif t == "done":
        return DoneAction()
    else:
        raise ValueError(f"Unknown action type: '{t}'")
