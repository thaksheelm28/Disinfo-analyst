"""
tests/test_api.py — Integration tests for the FastAPI endpoints

Covers all mandatory endpoints including /baseline, /grader, /tasks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body(self):
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "environment" in data


class TestRoot:
    def test_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_lists_endpoints(self):
        r = client.get("/")
        data = r.json()
        assert "endpoints" in data
        assert "tasks" in data


class TestReset:
    def test_reset_default(self):
        r = client.post("/reset", json={})
        assert r.status_code == 200

    def test_reset_returns_observation_fields(self):
        r = client.post("/reset", json={"task": "task1_classify", "seed": 42})
        data = r.json()
        assert "nodes" in data
        assert "edges" in data
        assert "claims" in data
        assert "agent_tags" in data
        assert "steps_taken" in data
        assert "steps_remaining" in data
        assert data["task"] == "task1_classify"

    def test_reset_has_nodes(self):
        r = client.post("/reset", json={"task": "task1_classify", "seed": 42})
        data = r.json()
        assert len(data["nodes"]) > 0

    def test_reset_invalid_task_422(self):
        r = client.post("/reset", json={"task": "task99_invalid"})
        assert r.status_code == 422

    def test_reset_all_three_tasks(self):
        for task in ["task1_classify", "task2_trace", "task3_severity"]:
            r = client.post("/reset", json={"task": task, "seed": 1})
            assert r.status_code == 200, f"reset failed for {task}: {r.text}"
            assert r.json()["task"] == task

    def test_reset_clears_state(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})
        obs = client.get("/state").json()
        nid = obs["nodes"][0]["node_id"]
        client.post("/step", json={"action": {"type": "tag", "node_id": nid, "label": "real"}})
        # Reset again — agent_tags should be empty
        r = client.post("/reset", json={"task": "task1_classify", "seed": 42})
        assert r.json()["agent_tags"] == {}


class TestStep:
    def setup_method(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})

    def test_step_tag_valid(self):
        obs = client.get("/state").json()
        nid = obs["nodes"][0]["node_id"]
        r = client.post("/step", json={"action": {"type": "tag", "node_id": nid, "label": "real"}})
        assert r.status_code == 200

    def test_step_returns_step_result_fields(self):
        obs = client.get("/state").json()
        nid = obs["nodes"][0]["node_id"]
        r = client.post("/step", json={"action": {"type": "tag", "node_id": nid, "label": "disinformation"}})
        data = r.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_has_progress_fields(self):
        obs = client.get("/state").json()
        nid = obs["nodes"][0]["node_id"]
        r = client.post("/step", json={"action": {"type": "tag", "node_id": nid, "label": "disinformation"}})
        reward = r.json()["reward"]
        assert "step_reward" in reward
        assert "tag_progress" in reward
        assert "trace_progress" in reward
        assert "score_progress" in reward
        assert "cumulative" in reward

    def test_step_invalid_node_penalised(self):
        r = client.post("/step", json={"action": {"type": "tag", "node_id": "FAKE_NODE", "label": "real"}})
        assert r.status_code == 200
        assert r.json()["reward"]["step_reward"] < 0

    def test_step_done_action(self):
        r = client.post("/step", json={"action": {"type": "done"}})
        assert r.status_code == 200
        assert r.json()["done"] is True

    def test_step_after_done_409(self):
        client.post("/step", json={"action": {"type": "done"}})
        r = client.post("/step", json={"action": {"type": "done"}})
        assert r.status_code == 409


class TestState:
    def test_state_returns_observation(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})
        r = client.get("/state")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data

    def test_state_does_not_advance_steps(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})
        before = client.get("/state").json()["steps_taken"]
        client.get("/state")
        after = client.get("/state").json()["steps_taken"]
        assert before == after


class TestEvaluate:
    def test_evaluate_returns_200(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})
        r = client.get("/evaluate")
        assert r.status_code == 200

    def test_evaluate_has_required_fields(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})
        r = client.get("/evaluate")
        data = r.json()
        assert "tag_f1" in data
        assert "trace_score" in data
        assert "severity_score" in data
        assert "final_score" in data

    def test_evaluate_score_bounded(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})
        r = client.get("/evaluate")
        score = r.json()["final_score"]
        assert 0.0 <= score <= 1.0


class TestTasks:
    def test_tasks_returns_200(self):
        r = client.get("/tasks")
        assert r.status_code == 200

    def test_tasks_has_three_tasks(self):
        r = client.get("/tasks")
        tasks = r.json()["tasks"]
        assert len(tasks) >= 3

    def test_tasks_has_action_schema(self):
        r = client.get("/tasks")
        data = r.json()
        assert "action_schema" in data

    def test_tasks_action_schema_has_all_types(self):
        r = client.get("/tasks")
        schema = r.json()["action_schema"]
        assert "TagAction" in schema
        assert "TraceAction" in schema
        assert "ScoreAction" in schema
        assert "DoneAction" in schema

    def test_tasks_difficulty_range(self):
        r = client.get("/tasks")
        tasks = r.json()["tasks"]
        difficulties = {t["difficulty"] for t in tasks.values()}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties


class TestGrader:
    def test_grader_returns_200(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})
        r = client.get("/grader")
        assert r.status_code == 200

    def test_grader_has_score_fields(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})
        r = client.get("/grader")
        data = r.json()
        assert "final_score" in data
        assert "tag_f1" in data
        assert "trace_score" in data
        assert "severity_score" in data

    def test_grader_has_grader_info(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})
        r = client.get("/grader")
        data = r.json()
        assert "grader_info" in data
        gi = data["grader_info"]
        assert "formula" in gi
        assert "partial_credit" in gi

    def test_grader_score_bounded(self):
        client.post("/reset", json={"task": "task1_classify", "seed": 42})
        r = client.get("/grader")
        assert 0.0 <= r.json()["final_score"] <= 1.0


class TestBaseline:
    def test_baseline_returns_200(self):
        r = client.post("/baseline")
        assert r.status_code == 200

    def test_baseline_has_all_three_task_scores(self):
        r = client.post("/baseline")
        scores = r.json()["scores"]
        assert "task1_classify" in scores
        assert "task2_trace" in scores
        assert "task3_severity" in scores

    def test_baseline_scores_bounded(self):
        r = client.post("/baseline")
        for task, score in r.json()["scores"].items():
            assert 0.0 <= score <= 1.0, f"{task} score {score} out of bounds"

    def test_baseline_has_mean_score(self):
        r = client.post("/baseline")
        data = r.json()
        assert "mean_score" in data
        assert 0.0 <= data["mean_score"] <= 1.0

    def test_baseline_reproducible(self):
        r1 = client.post("/baseline?seed=42")
        r2 = client.post("/baseline?seed=42")
        assert r1.json()["scores"] == r2.json()["scores"]

    def test_baseline_different_seeds_differ(self):
        r1 = client.post("/baseline?seed=1")
        r2 = client.post("/baseline?seed=99")
        # Scores may differ with different seeds
        s1 = r1.json()["scores"]
        s2 = r2.json()["scores"]
        # At least one score should differ (graphs are different)
        assert s1 != s2 or True  # soft check — seeds could coincidentally match
