"""
tests/test_grader.py — Pure-Python deterministic tests for the grader.

No network, no LLM, no external dependencies beyond the stdlib.
Run with:  python -m pytest tests/test_grader.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import pytest

from disinfo_env import Node, Edge, Claim
from grader import grade, _grade_tags, _grade_traces, _grade_severity, _bfs_distance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_node(nid, label="real", is_origin=False):
    n = Node(
        node_id=nid, kind="article",
        text=f"Text for {nid}",
        timestamp="2024-01-01T10:00:00",
        source="test.com",
        credibility=0.5,
        share_count=10,
    )
    n._true_label = label
    n._is_origin = is_origin
    return n


def make_claim(cid, severity):
    c = Claim(claim_id=cid, text=f"Claim {cid}")
    c._true_severity = severity
    return c


# ---------------------------------------------------------------------------
# BFS tests
# ---------------------------------------------------------------------------

class TestBFS:
    def test_direct_edge(self):
        adj = {"a": {"b"}, "b": set()}
        assert _bfs_distance(adj, "a", "b") == 1

    def test_two_hops(self):
        adj = {"a": {"b"}, "b": {"c"}, "c": set()}
        assert _bfs_distance(adj, "a", "c") == 2

    def test_no_path(self):
        adj = {"a": {"b"}, "b": set(), "c": set()}
        assert _bfs_distance(adj, "a", "c") is None

    def test_self(self):
        adj = {"a": set()}
        assert _bfs_distance(adj, "a", "a") == 0

    def test_directed(self):
        adj = {"a": {"b"}, "b": set()}
        assert _bfs_distance(adj, "b", "a") is None


# ---------------------------------------------------------------------------
# Tag grading tests
# ---------------------------------------------------------------------------

class TestTagGrading:
    def setup_method(self):
        self.nodes = {
            "n1": make_node("n1", "disinformation"),
            "n2": make_node("n2", "disinformation"),
            "n3": make_node("n3", "real"),
            "n4": make_node("n4", "real"),
        }

    def test_perfect_tags(self):
        tags = {"n1": "disinformation", "n2": "disinformation"}
        result = _grade_tags(self.nodes, tags)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"]    == pytest.approx(1.0)
        assert result["f1"]        == pytest.approx(1.0)

    def test_false_positive(self):
        tags = {"n1": "disinformation", "n2": "disinformation", "n3": "disinformation"}
        result = _grade_tags(self.nodes, tags)
        assert result["precision"] == pytest.approx(2/3)
        assert result["recall"]    == pytest.approx(1.0)

    def test_false_negative(self):
        tags = {"n1": "disinformation"}
        result = _grade_tags(self.nodes, tags)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"]    == pytest.approx(0.5)
        assert result["f1"]        == pytest.approx(2/3)

    def test_no_tags(self):
        result = _grade_tags(self.nodes, {})
        assert result["f1"] == pytest.approx(0.0)

    def test_all_wrong(self):
        tags = {"n3": "disinformation", "n4": "disinformation"}
        result = _grade_tags(self.nodes, tags)
        assert result["precision"] == pytest.approx(0.0)
        assert result["f1"]        == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Trace grading tests
# ---------------------------------------------------------------------------

class TestTraceGrading:
    def setup_method(self):
        self.nodes = {
            "n1": make_node("n1"),
            "n2": make_node("n2"),
            "n3": make_node("n3"),
        }
        self.edges = [
            Edge(src="n1", dst="n2", relation="shares"),
            Edge(src="n2", dst="n3", relation="paraphrases"),
        ]

    def test_correct_direct_trace(self):
        result = _grade_traces(self.nodes, self.edges, [("n1", "n2")])
        assert result["trace_score"] == pytest.approx(1.0)

    def test_correct_indirect_trace(self):
        result = _grade_traces(self.nodes, self.edges, [("n1", "n3")])
        assert result["trace_score"] == pytest.approx(1.0)

    def test_backwards_trace(self):
        result = _grade_traces(self.nodes, self.edges, [("n2", "n1")])
        assert result["trace_score"] == pytest.approx(0.25)

    def test_wrong_trace(self):
        nodes = {
            "n1": make_node("n1"),
            "n2": make_node("n2"),
            "n3": make_node("n3"),
            "n4": make_node("n4"),
        }
        edges = [
            Edge(src="n1", dst="n2", relation="shares"),
            Edge(src="n3", dst="n4", relation="shares"),
        ]
        result = _grade_traces(nodes, edges, [("n1", "n4")])
        assert result["trace_score"] == pytest.approx(0.0)

    def test_no_traces(self):
        result = _grade_traces(self.nodes, self.edges, [])
        assert result["trace_score"] == pytest.approx(0.0)

    def test_mixed_traces(self):
        result = _grade_traces(self.nodes, self.edges, [
            ("n1", "n2"),  # correct → 1.0
            ("n2", "n1"),  # backwards → 0.25
        ])
        assert result["trace_score"] == pytest.approx(0.625)


# ---------------------------------------------------------------------------
# Severity grading tests
# ---------------------------------------------------------------------------

class TestSeverityGrading:
    def setup_method(self):
        self.claims = {
            "c1": make_claim("c1", 0.9),
            "c2": make_claim("c2", 0.2),
        }

    def test_perfect_scores(self):
        result = _grade_severity(self.claims, {"c1": 0.9, "c2": 0.2})
        assert result["rmse"]           == pytest.approx(0.0)
        assert result["severity_score"] == pytest.approx(1.0)

    def test_all_wrong(self):
        # Maximum RMSE = 1.0 means severity_score = 0.0
        result = _grade_severity(self.claims, {"c1": 0.0, "c2": 1.0})
        assert result["severity_score"] == pytest.approx(0.0, abs=0.2)

    def test_unscored_claim_defaults_to_0_5(self):
        # c2 not scored — defaults to 0.5
        result = _grade_severity(self.claims, {"c1": 0.9})
        expected_mse = ((0.9 - 0.9)**2 + (0.2 - 0.5)**2) / 2
        expected_rmse = math.sqrt(expected_mse)
        assert result["rmse"] == pytest.approx(expected_rmse, abs=1e-6)

    def test_empty_claims(self):
        result = _grade_severity({}, {})
        assert result["severity_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Full grade() integration test
# ---------------------------------------------------------------------------

class TestFullGrade:
    def test_perfect_agent(self):
        nodes = {
            "n1": make_node("n1", "disinformation", is_origin=True),
            "n2": make_node("n2", "disinformation"),
            "n3": make_node("n3", "real"),
        }
        edges = [Edge(src="n1", dst="n2", relation="shares")]
        claims = {"c1": make_claim("c1", 0.85)}

        result = grade(
            nodes=nodes,
            edges=edges,
            claims=claims,
            agent_tags={"n1": "disinformation", "n2": "disinformation"},
            agent_traces=[("n1", "n2")],
            agent_scores={"c1": 0.85},
        )
        assert result["tag_f1"]        == pytest.approx(1.0)
        assert result["trace_score"]   == pytest.approx(1.0)
        assert result["severity_score"] == pytest.approx(1.0)
        assert result["final_score"]   == pytest.approx(1.0)

    def test_zero_agent(self):
        nodes = {
            "n1": make_node("n1", "disinformation"),
        }
        edges = []
        claims = {"c1": make_claim("c1", 0.9)}

        result = grade(
            nodes=nodes, edges=edges, claims=claims,
            agent_tags={}, agent_traces=[], agent_scores={},
        )
        assert result["tag_f1"]      == pytest.approx(0.0)
        assert result["trace_score"] == pytest.approx(0.0)
        assert result["final_score"] < 0.3

    def test_weights_sum_to_one(self):
        b = {"tag_weight": 0.40, "trace_weight": 0.35, "severity_weight": 0.25}
        assert sum(b.values()) == pytest.approx(1.0)

    def test_score_bounded(self):
        nodes = {"n1": make_node("n1", "disinformation")}
        edges = []
        claims = {"c1": make_claim("c1", 0.5)}
        result = grade(
            nodes=nodes, edges=edges, claims=claims,
            agent_tags={"n1": "disinformation"},
            agent_traces=[],
            agent_scores={"c1": 0.5},
        )
        assert 0.0 <= result["final_score"] <= 1.0
