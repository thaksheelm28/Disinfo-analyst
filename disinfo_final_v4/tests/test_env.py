"""
tests/test_env.py — Integration tests covering all three tasks of DisinfoEnv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from disinfo_env import DisinfoEnv, TASKS
from models import Observation, StepResult, Reward, EvaluationResult


@pytest.fixture
def env():
    return DisinfoEnv(seed=42)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    def test_three_tasks_defined(self):
        assert len(TASKS) >= 3

    def test_task_ids_present(self):
        assert "task1_classify" in TASKS
        assert "task2_trace"    in TASKS
        assert "task3_severity" in TASKS

    def test_difficulty_ordering(self):
        assert TASKS["task1_classify"]["difficulty"] == "easy"
        assert TASKS["task2_trace"]["difficulty"]    == "medium"
        assert TASKS["task3_severity"]["difficulty"] == "hard"

    def test_each_task_has_primary_metric(self):
        for tid, tcfg in TASKS.items():
            assert "primary_metric" in tcfg, f"{tid} missing primary_metric"


# ---------------------------------------------------------------------------
# Typed Observation model
# ---------------------------------------------------------------------------

class TestObservationModel:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task="task1_classify")
        assert isinstance(obs, Observation)

    def test_observation_has_required_fields(self, env):
        obs = env.reset(task="task1_classify")
        assert len(obs.nodes) > 0
        assert isinstance(obs.edges, list)
        assert isinstance(obs.claims, list)
        assert isinstance(obs.agent_tags, dict)
        assert isinstance(obs.steps_taken, int)
        assert isinstance(obs.steps_remaining, int)
        assert obs.task == "task1_classify"

    def test_credibility_hidden(self, env):
        obs = env.reset(task="task1_classify")
        for node in obs.nodes:
            assert not hasattr(node, "credibility")
            assert not hasattr(node, "_true_label")

    def test_nodes_typed(self, env):
        obs = env.reset(task="task1_classify")
        for node in obs.nodes:
            assert node.kind in ("seed", "article", "post")
            assert isinstance(node.share_count, int)


# ---------------------------------------------------------------------------
# Typed StepResult and Reward models
# ---------------------------------------------------------------------------

class TestStepResultModel:
    def test_step_returns_step_result(self, env):
        obs = env.reset(task="task1_classify")
        nid = obs.nodes[0].node_id
        result = env.step({"type": "tag", "node_id": nid, "label": "real"})
        assert isinstance(result, StepResult)

    def test_reward_is_typed(self, env):
        obs = env.reset(task="task1_classify")
        nid = obs.nodes[0].node_id
        result = env.step({"type": "tag", "node_id": nid, "label": "disinformation"})
        assert isinstance(result.reward, Reward)

    def test_reward_has_partial_progress_fields(self, env):
        obs = env.reset(task="task1_classify")
        nid = obs.nodes[0].node_id
        result = env.step({"type": "tag", "node_id": nid, "label": "disinformation"})
        rwd = result.reward
        assert hasattr(rwd, "step_reward")
        assert hasattr(rwd, "tag_progress")
        assert hasattr(rwd, "trace_progress")
        assert hasattr(rwd, "score_progress")
        assert hasattr(rwd, "cumulative")

    def test_reward_partial_progress_at_every_step(self, env):
        """Reward carries partial progress signal at EVERY step, not just done."""
        obs = env.reset(task="task1_classify")
        cumulative_rewards = []
        # Tag all disinfo nodes (oracle)
        for nid, node in env._nodes.items():
            if node._true_label == "disinformation":
                result = env.step({"type": "tag", "node_id": nid, "label": "disinformation"})
                cumulative_rewards.append(result.reward.cumulative)
        # cumulative should increase as more correct tags are added
        assert len(cumulative_rewards) > 0
        assert all(r >= 0 for r in cumulative_rewards)

    def test_invalid_action_penalised(self, env):
        env.reset(task="task1_classify")
        result = env.step({"type": "tag", "node_id": "nonexistent", "label": "real"})
        assert result.reward.step_reward < 0

    def test_invalid_label_penalised(self, env):
        obs = env.reset(task="task1_classify")
        nid = obs.nodes[0].node_id
        result = env.step({"type": "tag", "node_id": nid, "label": "banana"})
        assert result.reward.step_reward < 0
        assert "error" in result.info


# ---------------------------------------------------------------------------
# Task 1 — Node Classification (easy)
# ---------------------------------------------------------------------------

class TestTask1Classify:
    def test_task1_resets(self, env):
        obs = env.reset(task="task1_classify")
        assert obs.task == "task1_classify"

    def test_task1_uses_easy_scenario_by_default(self, env):
        obs = env.reset(task="task1_classify")
        assert env._scenario == "easy"

    def test_task1_primary_metric_is_tag_f1(self):
        assert TASKS["task1_classify"]["primary_metric"] == "tag_f1"

    def test_task1_perfect_run(self, env):
        env.reset(task="task1_classify")
        for nid, node in env._nodes.items():
            env.step({"type": "tag", "node_id": nid, "label": node._true_label})
        env.step({"type": "done"})
        result = env.evaluate()
        assert result.tag_f1 == pytest.approx(1.0, abs=0.01)

    def test_task1_zero_run(self, env):
        env.reset(task="task1_classify")
        env.step({"type": "done"})
        result = env.evaluate()
        assert result.tag_f1 == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Task 2 — Propagation Tracing (medium)
# ---------------------------------------------------------------------------

class TestTask2Trace:
    def test_task2_resets(self, env):
        obs = env.reset(task="task2_trace")
        assert obs.task == "task2_trace"

    def test_task2_uses_default_scenario(self, env):
        env.reset(task="task2_trace")
        assert env._scenario == "default"

    def test_task2_primary_metric_is_trace_score(self):
        assert TASKS["task2_trace"]["primary_metric"] == "trace_score"

    def test_task2_correct_trace(self, env):
        env.reset(task="task2_trace")
        # Trace all real edges — should score 1.0
        for edge in env._edges:
            env.step({"type": "trace", "src": edge.src, "dst": edge.dst})
        env.step({"type": "done"})
        result = env.evaluate()
        assert result.trace_score == pytest.approx(1.0, abs=0.01)

    def test_task2_no_traces_scores_zero(self, env):
        env.reset(task="task2_trace")
        env.step({"type": "done"})
        result = env.evaluate()
        assert result.trace_score == pytest.approx(0.0, abs=0.01)

    def test_task2_backwards_trace_partial_credit(self, env):
        env.reset(task="task2_trace")
        # Trace the first edge backwards
        if env._edges:
            e = env._edges[0]
            result = env.step({"type": "trace", "src": e.dst, "dst": e.src})
        env.step({"type": "done"})
        res = env.evaluate()
        assert 0.0 < res.trace_score <= 0.25 + 0.01


# ---------------------------------------------------------------------------
# Task 3 — Claim Severity Rating (hard)
# ---------------------------------------------------------------------------

class TestTask3Severity:
    def test_task3_resets(self, env):
        obs = env.reset(task="task3_severity")
        assert obs.task == "task3_severity"

    def test_task3_uses_hard_scenario(self, env):
        env.reset(task="task3_severity")
        assert env._scenario == "hard"

    def test_task3_primary_metric_is_severity_score(self):
        assert TASKS["task3_severity"]["primary_metric"] == "severity_score"

    def test_task3_perfect_run(self, env):
        env.reset(task="task3_severity")
        for cid, claim in env._claims.items():
            env.step({"type": "score", "claim_id": cid, "severity": claim._true_severity})
        env.step({"type": "done"})
        result = env.evaluate()
        assert result.severity_score == pytest.approx(1.0, abs=0.01)

    def test_task3_out_of_range_severity_penalised(self, env):
        obs = env.reset(task="task3_severity")
        cid = obs.claims[0].claim_id
        result = env.step({"type": "score", "claim_id": cid, "severity": 1.5})
        assert result.reward.step_reward < 0


# ---------------------------------------------------------------------------
# EvaluationResult typed model
# ---------------------------------------------------------------------------

class TestEvaluationResult:
    def test_evaluate_returns_typed_result(self, env):
        env.reset(task="task1_classify")
        result = env.evaluate()
        assert isinstance(result, EvaluationResult)

    def test_evaluate_has_all_fields(self, env):
        env.reset(task="task1_classify")
        r = env.evaluate()
        assert hasattr(r, "tag_f1")
        assert hasattr(r, "trace_score")
        assert hasattr(r, "severity_score")
        assert hasattr(r, "final_score")
        assert hasattr(r, "breakdown")
        assert hasattr(r, "task")

    def test_final_score_bounded(self, env):
        env.reset(task="task1_classify")
        r = env.evaluate()
        assert 0.0 <= r.final_score <= 1.0

    def test_perfect_oracle_scores_1(self):
        env = DisinfoEnv(seed=0, max_steps=500)
        env.reset(task="task1_classify", scenario="easy")
        for nid, node in env._nodes.items():
            env.step({"type": "tag", "node_id": nid, "label": node._true_label})
        for edge in env._edges:
            env.step({"type": "trace", "src": edge.src, "dst": edge.dst})
        for cid, claim in env._claims.items():
            env.step({"type": "score", "claim_id": cid, "severity": claim._true_severity})
        env.step({"type": "done"})
        r = env.evaluate()
        assert r.tag_f1         == pytest.approx(1.0, abs=0.01)
        assert r.trace_score    == pytest.approx(1.0, abs=0.01)
        assert r.severity_score == pytest.approx(1.0, abs=0.01)
        assert r.final_score    == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# State API
# ---------------------------------------------------------------------------

class TestStateAPI:
    def test_state_returns_observation(self, env):
        env.reset(task="task1_classify")
        obs = env.state()
        assert isinstance(obs, Observation)

    def test_state_does_not_advance_step(self, env):
        env.reset(task="task1_classify")
        before = env._step_count
        env.state()
        assert env._step_count == before

    def test_state_reflects_agent_actions(self, env):
        obs = env.reset(task="task1_classify")
        nid = obs.nodes[0].node_id
        env.step({"type": "tag", "node_id": nid, "label": "disinformation"})
        state = env.state()
        assert nid in state.agent_tags


# ---------------------------------------------------------------------------
# Episode control
# ---------------------------------------------------------------------------

class TestEpisodeControl:
    def test_done_action_ends_episode(self, env):
        env.reset(task="task1_classify")
        result = env.step({"type": "done"})
        assert result.done and env.done

    def test_step_after_done_raises(self, env):
        env.reset(task="task1_classify")
        env.step({"type": "done"})
        with pytest.raises(RuntimeError):
            env.step({"type": "done"})

    def test_max_steps_terminates_episode(self):
        env = DisinfoEnv(seed=42, max_steps=3)
        obs = env.reset(task="task1_classify")
        done = False
        for _ in range(5):
            result = env.step({"type": "tag", "node_id": obs.nodes[0].node_id, "label": "real"})
            done = result.done
            if done: break
        assert done

    def test_reset_clears_state(self, env):
        obs = env.reset(task="task1_classify")
        env.step({"type": "tag", "node_id": obs.nodes[0].node_id, "label": "real"})
        env.reset(task="task2_trace")
        assert env._agent_tags == {}
        assert env._step_count == 0

    def test_task_switches_on_reset(self, env):
        env.reset(task="task1_classify")
        assert env._task == "task1_classify"
        env.reset(task="task2_trace")
        assert env._task == "task2_trace"

    def test_repr(self, env):
        env.reset(task="task1_classify")
        assert "DisinfoEnv" in repr(env)
        assert "task1_classify" in repr(env)

    def test_invalid_task_raises(self, env):
        with pytest.raises(ValueError):
            env.reset(task="task99_invalid")
