"""
grader.py — Deterministic oracle grader for Disinformation Analyst

Three sub-scores, each in [0.0, 1.0]:
  1. tag_score    — F1 of (precision × recall) on disinformation labels
  2. trace_score  — how well the agent's claimed propagation chains match
                    the ground-truth graph (based on path edit distance)
  3. severity_score — 1 - normalised RMSE of claim severity ratings

Final score = weighted average:
  0.40 × tag_score + 0.35 × trace_score + 0.25 × severity_score
"""

from __future__ import annotations

import math

from models import Node, Edge, Claim


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def grade(
    nodes: dict,
    edges: list,
    claims: dict,
    agent_tags: dict[str, str],
    agent_traces: list[tuple[str, str]],
    agent_scores: dict[str, float],
) -> dict:
    """
    Compute the full graded result.

    Returns a dict with:
      - tag_precision, tag_recall, tag_f1
      - trace_score
      - severity_rmse, severity_score
      - final_score
      - breakdown (human-readable)
    """
    tag_result     = _grade_tags(nodes, agent_tags)
    trace_result   = _grade_traces(nodes, edges, agent_traces)
    severity_result = _grade_severity(claims, agent_scores)

    final = (
        0.40 * tag_result["f1"]
        + 0.35 * trace_result["trace_score"]
        + 0.25 * severity_result["severity_score"]
    )

    return {
        # Tag sub-scores
        "tag_precision":   round(tag_result["precision"], 4),
        "tag_recall":      round(tag_result["recall"], 4),
        "tag_f1":          round(tag_result["f1"], 4),
        # Trace sub-score
        "trace_score":     round(trace_result["trace_score"], 4),
        "trace_details":   trace_result["details"],
        # Severity sub-score
        "severity_rmse":   round(severity_result["rmse"], 4),
        "severity_score":  round(severity_result["severity_score"], 4),
        # Final
        "final_score":     round(final, 4),
        "breakdown": {
            "tag_weight":      0.40,
            "trace_weight":    0.35,
            "severity_weight": 0.25,
        },
    }


# ---------------------------------------------------------------------------
# Sub-graders
# ---------------------------------------------------------------------------

def _grade_tags(nodes: dict, agent_tags: dict[str, str]) -> dict:
    """
    Compute precision and recall of the agent's 'disinformation' tag
    against ground-truth node labels.

    Any node the agent labels 'disinformation' that is truly
    'disinformation' is a true positive.
    """
    # Ground truth: all node_ids that are truly disinformation
    true_disinfo = {
        nid for nid, n in nodes.items()
        if n._true_label == "disinformation"
    }

    # Agent predictions: all node_ids tagged as disinformation
    pred_disinfo = {
        nid for nid, label in agent_tags.items()
        if label == "disinformation"
    }

    tp = len(true_disinfo & pred_disinfo)
    fp = len(pred_disinfo - true_disinfo)
    fn = len(true_disinfo - pred_disinfo)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def _grade_traces(nodes: dict, edges: list, agent_traces: list[tuple[str, str]]) -> dict:
    """
    Compare the agent's claimed propagation chains to the ground-truth graph.

    For each agent trace (src → dst), we check:
      - Is there a directed path from src to dst in the ground-truth graph?
      - If yes, score = 1.0
      - If no, score = max(0, 1 - shortest_path_error / graph_diameter)

    Overall trace_score = mean of per-trace scores.
    """
    # Build adjacency map from ground-truth edges
    adjacency: dict[str, set[str]] = {nid: set() for nid in nodes}
    for edge in edges:
        adjacency[edge.src].add(edge.dst)

    if not agent_traces:
        return {"trace_score": 0.0, "details": []}

    details = []
    scores = []

    for (src, dst) in agent_traces:
        if src not in adjacency or dst not in adjacency:
            scores.append(0.0)
            details.append({"src": src, "dst": dst, "result": "invalid_node", "score": 0.0})
            continue

        dist = _bfs_distance(adjacency, src, dst)
        if dist == 0 and src == dst:
            s = 0.0  # self-trace — useless
        elif dist is not None:
            s = 1.0  # valid path exists
        else:
            # No path: penalise by reverse direction or completely wrong
            reverse_dist = _bfs_distance(adjacency, dst, src)
            if reverse_dist is not None:
                s = 0.25  # backwards trace, partial credit
            else:
                s = 0.0

        scores.append(s)
        details.append({
            "src":    src,
            "dst":    dst,
            "result": "hit" if s == 1.0 else ("backwards" if s == 0.25 else "miss"),
            "score":  s,
        })

    return {
        "trace_score": sum(scores) / len(scores),
        "details":     details,
    }


def _grade_severity(claims: dict, agent_scores: dict[str, float]) -> dict:
    """
    Compute RMSE between agent severity scores and ground-truth severities.
    Convert to a [0, 1] score via 1 - rmse (since max rmse is 1.0).
    Unscored claims count as 0.5 (worst case: the agent is maximally uncertain).
    """
    true_vals = []
    pred_vals = []

    for cid, claim in claims.items():
        true_vals.append(claim._true_severity)
        pred_vals.append(agent_scores.get(cid, 0.5))

    if not true_vals:
        return {"rmse": 0.0, "severity_score": 1.0}

    mse  = sum((t - p) ** 2 for t, p in zip(true_vals, pred_vals)) / len(true_vals)
    rmse = math.sqrt(mse)

    return {
        "rmse":           rmse,
        "severity_score": max(0.0, 1.0 - rmse),
    }


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def _bfs_distance(adjacency: dict[str, set[str]], start: str, end: str) -> int | None:
    """BFS shortest path length from start → end. Returns None if unreachable."""
    if start == end:
        return 0
    visited = {start}
    queue   = [(start, 0)]
    while queue:
        node, dist = queue.pop(0)
        for neighbour in adjacency.get(node, set()):
            if neighbour == end:
                return dist + 1
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, dist + 1))
    return None
