# Disinformation Analyst — OpenEnv Environment

A three-task OpenEnv environment where an LLM agent analyses synthetic news and social-media propagation graphs to detect, trace, and score disinformation. This environment simulates real-world trust-and-safety work that human analysts do every day at platforms like Twitter/X, Meta, and news verification organisations.

---

## Why This Environment?

Most content-moderation benchmarks ask an agent to classify a single article in isolation. This environment is different:

- **Graph-based reasoning** — content is connected. A viral post traces back through articles to a planted seed. The agent must reason about the *network*, not just individual texts.
- **Three distinct sub-tasks** — each requires a different reasoning strategy (classification, graph traversal, numerical judgment).
- **Partial progress signals** — the typed `Reward` model emits `tag_progress`, `trace_progress`, and `score_progress` at *every* step, not just episode end.
- **Fully deterministic grader** — zero LLM calls in evaluation. Scores are reproducible to the bit given the same RNG seed.
- **Hidden ground truth** — credibility scores and true labels are stripped from the observation, forcing the agent to infer them from timestamps, source domains, share counts, and propagation structure.

---

## Tasks

### Task 1 — Node Classification `(easy)`

**Objective:** Tag every graph node with the correct disinformation label.

**Primary metric:** `tag_f1` — F1 score of `disinformation` predictions vs ground-truth labels.

**Reasoning required:** Content understanding. Read node text and metadata (source domain, share count, timestamp) to classify each node as `real`, `disinformation`, `satire`, `misleading`, or `unverified`.

**Expected baseline score:** ~0.45–0.60 (GPT-4o-mini, seed=42)

---

### Task 2 — Propagation Tracing `(medium)`

**Objective:** Identify upstream source relationships — assert which nodes are origins or spreaders of other nodes.

**Primary metric:** `trace_score` — mean BFS path-existence score over claimed `(src→dst)` pairs. Correct path = 1.0, backwards = 0.25 partial credit, no path = 0.0.

**Reasoning required:** Multi-hop graph reasoning. Combine timestamps (earlier = more likely origin), source credibility signals, and propagation edge relations (`shares`, `paraphrases`, `refutes`, `quotes`) to trace how content spread.

**Expected baseline score:** ~0.30–0.50 (GPT-4o-mini, seed=42)

---

### Task 3 — Claim Severity Rating `(hard)`

**Objective:** Assign a float severity score `[0.0–1.0]` to each factual claim based on real-world harm potential.

**Primary metric:** `severity_score` = `1 − normalised RMSE` vs ground-truth severities.

**Scoring guide:**
- `0.0` — completely harmless (local news, weather)
- `0.3–0.5` — moderately misleading but not dangerous
- `0.7–0.9` — health misinformation, financial fraud
- `1.0` — incitement, dangerous medical advice

**Reasoning required:** Nuanced judgment about real-world harm. No binary answer. Requires weighing claim content, likely spread, and affected population.

**Expected baseline score:** ~0.40–0.55 (GPT-4o-mini, seed=42)

---

## File Structure

```
disinfo_analyst/
├── models.py           # Pydantic: Observation, Action, Reward, StepResult, EvaluationResult
│                       # Dataclasses: Node, Edge, Claim (internal, not exposed to agent)
├── disinfo_env.py      # DisinfoEnv — reset/step/state/evaluate, 3 tasks
├── grader.py           # Deterministic oracle grader (tag F1, trace BFS, severity RMSE)
├── graph_factory.py    # Synthetic rumour propagation graph builder
├── scenarios.json      # Graph configs (easy / default / hard / adversarial)
├── app.py              # FastAPI server (all OpenEnv endpoints)
├── inference.py        # Baseline LLM agent (OpenAI client, all 3 tasks)
├── openenv.yaml        # OpenEnv spec file
├── Dockerfile          # Hugging Face Spaces compatible
├── requirements.txt
├── pyproject.toml
├── tests/
│   ├── test_env.py     # Integration tests — all 3 tasks, typed models, reward signals
│   └── test_grader.py  # Unit tests — grader sub-functions (BFS, F1, RMSE)
└── README.md
```

---

## Observation Space

Returned by `reset()`, `step()`, and `state()` as a typed `Observation` Pydantic model.

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `list[NodeObservation]` | All graph nodes. Fields: `node_id`, `kind` (seed/article/post), `text`, `timestamp`, `source`, `share_count`. **Credibility is hidden.** |
| `edges` | `list[EdgeObservation]` | Directed propagation edges. Fields: `src`, `dst`, `relation` (shares/paraphrases/refutes/quotes). |
| `claims` | `list[ClaimObservation]` | Factual claims to score. Fields: `claim_id`, `text`. |
| `agent_tags` | `dict[str, str]` | Tags the agent has assigned so far. |
| `agent_traces` | `list[list[str]]` | Propagation chains the agent has asserted. |
| `agent_scores` | `dict[str, float]` | Severity scores the agent has assigned. |
| `steps_taken` | `int` | Steps used in this episode. |
| `steps_remaining` | `int` | Steps left before forced termination. |
| `task` | `str` | Active task ID. |

---

## Action Space

All actions are validated Pydantic models. Pass as dicts to `step()`.

### `tag` — label a node
```python
{"type": "tag", "node_id": "art_3", "label": "disinformation"}
```
Valid labels: `real`, `disinformation`, `satire`, `misleading`, `unverified`.

### `trace` — assert a propagation link
```python
{"type": "trace", "src": "seed_0", "dst": "art_3"}
```
Asserts `seed_0` is an upstream source of `art_3`.

### `score` — assign claim severity
```python
{"type": "score", "claim_id": "claim_0", "severity": 0.87}
```
`severity` must be in `[0.0, 1.0]`.

### `done` — end the episode
```python
{"type": "done"}
```

**Invalid actions** return `step_reward = -0.05` and an `"error"` key in `info`.

---

## Reward Model

`step()` returns a typed `Reward` Pydantic model at **every step** — not just episode end:

| Field | Type | Description |
|-------|------|-------------|
| `step_reward` | `float` | Immediate action reward (`0.0` valid, `-0.05` invalid) |
| `tag_progress` | `float [0,1]` | Running tag F1 score |
| `trace_progress` | `float [0,1]` | Running trace BFS score |
| `score_progress` | `float [0,1]` | Running severity accuracy |
| `cumulative` | `float [0,1]` | Weighted running total |

---

## Grading

`evaluate()` returns a typed `EvaluationResult` with:

```
final_score = 0.40 × tag_f1 + 0.35 × trace_score + 0.25 × severity_score
```

| Sub-score | Weight | Method |
|-----------|--------|--------|
| `tag_f1` | 40% | F1 of `disinformation` node predictions |
| `trace_score` | 35% | Mean BFS path score; backwards = 0.25 partial credit |
| `severity_score` | 25% | `1 − RMSE`; unscored claims default to 0.5 |

All scores are in `[0.0, 1.0]`. A perfect oracle agent achieves `1.0`.

---

## Quick Start

### Local Python
```python
from disinfo_env import DisinfoEnv

env = DisinfoEnv(seed=42)

# Task 1 — Node Classification
obs = env.reset(task="task1_classify", scenario="easy")
while not env.done:
    result = env.step({"type": "tag", "node_id": obs.nodes[0].node_id, "label": "disinformation"})
    obs = result.observation
    print(f"cumulative reward: {result.reward.cumulative:.3f}")

final = env.evaluate()
print(f"Task 1 score: {final.final_score:.4f}")

# Task 2 — Propagation Tracing
obs = env.reset(task="task2_trace")
...

# Task 3 — Claim Severity Rating
obs = env.reset(task="task3_severity")
...
```

### Run the Server
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### API Usage
```bash
# Reset for Task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "task1_classify", "scenario": "easy", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"type": "tag", "node_id": "art_0", "label": "disinformation"}}'

# Read state (no side effects)
curl http://localhost:7860/state

# Get score
curl http://localhost:7860/evaluate

# List all tasks
curl http://localhost:7860/tasks
```

### Run Baseline Agent
```bash
export OPENAI_API_KEY=sk-...

# All 3 tasks (produces reproducible baseline scores)
python inference.py --seed 42

# Single task
python inference.py --task task1_classify --scenario easy --seed 42

# Dry-run (no server, no API key)
python inference.py --dry-run --seed 42
```

### Docker
```bash
docker build -t disinfo-analyst .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... disinfo-analyst
```

### Run Tests
```bash
python -m pytest tests/ -v
```

---

## Scenarios (Graph Configs)

| Name | Seeds | Articles | Posts | Claims | Disinfo ratio |
|------|-------|----------|-------|--------|---------------|
| `easy` | 1 | 6 | 8 | 2 | 50% |
| `default` | 2 | 10 | 15 | 4 | 40% |
| `hard` | 3 | 16 | 24 | 5 | 35% |
| `adversarial` | 3 | 20 | 30 | 6 | 25% |

Tasks auto-select a scenario based on difficulty (easy→easy, medium→default, hard→hard). Override with `--scenario`.

---

## Deploying to Hugging Face Spaces

1. Create a new Space: `New Space → Docker SDK`
2. Push this directory as the Space repo
3. Add `OPENAI_API_KEY` as a Secret in Settings → Variables and secrets
4. Space will auto-build and serve on port 7860
5. Submit the Space URL to OpenEnv

---

## Design Notes

**Why a propagation graph?** Single-article classification is a solved task for modern LLMs. Requiring reasoning about *how* content propagates forces multi-hop inference that isn't solvable by reading one document.

**Why three separate tasks?** Real trust-and-safety workflows involve multiple stages: identify suspicious content (classification), understand how it spread (tracing), then prioritise response effort (severity). Separating these into distinct tasks lets the grader reward each skill independently and lets researchers study which reasoning capabilities are hardest.

**Why a deterministic grader?** No LLM should be required to grade an LLM agent. The grader uses BFS path queries and standard classification metrics — reproducible to the bit, runnable offline, trivially auditable.

**Reproducibility:** Every graph is generated from a `(scenario, seed)` pair. The same pair always produces identical graphs and correct answers, making cross-model comparison reliable.

---

## License

MIT
