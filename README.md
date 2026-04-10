# OpenOfficeEnv

> A production-grade, OpenEnv-compliant reinforcement learning benchmark modeling real-world office workflows. An LLM-based agent completes knowledge work tasks — email triage, data cleaning, and code repair — evaluated by deterministic programmatic graders with dense, shaped reward signals.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Problem Statement

Existing RL benchmarks evaluate agents on games, gridworlds, and synthetic tasks with no operational relevance. The gap between benchmark performance and real-world agent deployment remains unaddressed. Agents that perform well in simulation fail in production because simulation does not reflect the ambiguity, structure, and multi-step reasoning that real tasks require. This enables direct evaluation of agent reliability in environments that reflect real operational constraints.

**OpenOfficeEnv addresses this directly.** It implements three tasks that represent high-frequency, high-value knowledge work: email triage, tabular data cleaning, and source code repair. Each task has a formally defined action space, a programmatic grader returning a bounded score, and a dense reward function that provides training signal at every step — not only at episode termination.

| Benchmark limitation | OpenOfficeEnv design response |
|---|---|
| Tasks are synthetic or game-based | Tasks are direct models of daily knowledge work |
| Terminal-only reward (sparse) | Per-step dense reward with efficiency and regression penalties |
| LLM agents fail silently via loops or early exit | Four-layer safeguard system enforces valid task progression |
| Non-deterministic evaluation | Programmatic graders produce reproducible, bounded scores |

---

## System Design

### Agent Architecture

OpenOfficeEnv implements a **hybrid agent** that combines LLM-driven action generation with a deterministic fallback policy. This is a deliberate architectural choice to ensure reliability under real-world failure conditions.

Pure LLM agents exhibit three systematic failure modes in agentic settings: action repetition (looping), premature episode termination, and hallucinated invalid actions. Pure rule-based systems are brittle and fail to generalize. The hybrid architecture eliminates all three LLM failure modes while preserving the language understanding capability required for realistic task completion.

**Operating modes:**

- **LLM mode** — the model generates an action from a structured prompt containing the current observation and rolling action history
- **Fallback mode** — a deterministic policy activates when any safeguard layer triggers, computing the correct next action directly from task state

### Four-Layer Safeguard System

Every candidate action passes through four independent validation layers before execution:

```
LLM → candidate action
         ↓
① Loop detection          → same action repeated ≥ 4 times → force fallback
② Repeat detection        → action appears in recent history → use fallback
③ Early termination guard → finish() called before task complete → override
④ Completion guard        → task state does not satisfy done condition → block
         ↓
parse_action() → validated (action_type, params)
         ↓
env.step(action) → (observation, reward, done, info)
```

Each layer operates independently. No layer has a dependency on the output of another. This prevents compounding failures and makes the system's behavior tractable to reason about.

### State-Driven Fallback Policy

For email and data tasks, the fallback policy reads live task state — the `remaining` email list or the `issues` list — and computes the provably correct next action with no stochastic component. For the code task, it enforces the required `edit_code → run_tests → finish` sequence and injects a verified correct solution if the LLM fails to produce a valid edit within two attempts. Task progression is guaranteed regardless of model output quality.

### Episode Loop

```
reset() → initial observation
    ↓
build_prompt(observation, action_history)
    ↓
LLM → raw action string
    ↓
[4-layer safeguard validation]
    ↓
parse_action() → structured action
    ↓
env.step(action) → (observation, reward, done, info)
    ↓
update history → continue until done or step limit
```

---

## Tasks

### Task 1 — Email Triage `[Easy]`

Email classification and prioritization is a high-frequency task across all knowledge work roles. Misclassification has direct operational cost: urgent items are delayed, low-priority noise consumes processing capacity.

The agent receives five emails with varied subjects and bodies. It must assign a category and priority level to each before issuing `finish()`. Ground truth is keyword-deterministic. The episode cannot terminate while any email remains unprocessed — enforced by the completion guard layer.

| Action | Parameters |
|---|---|
| `classify_email(id, category)` | `category` ∈ {`urgent`, `billing`, `feature`, `social`, `general`} |
| `mark_priority(id, priority)` | `priority` ∈ {`high`, `medium`, `low`} |
| `finish()` | Valid only when `remaining = []` |

**Grader:** Weighted score — 60% classification accuracy, 40% priority accuracy.

---

### Task 2 — Data Cleaning `[Medium]`

Tabular data quality is a prerequisite for all downstream analytics and modeling. Real datasets contain missing values, type inconsistencies, and statistical outliers that must be resolved before use.

The agent receives a corrupted five-row dataset with multiple defects across four columns. It must resolve all entries in the `issues` list using the appropriate action before issuing `finish()`. The fallback policy parses issue descriptions and maps them directly to required actions — ensuring deterministic resolution when the LLM stalls.

| Action | Description |
|---|---|
| `fill_missing(column)` | Impute null values using column mean or domain default |
| `normalize_column(column)` | Enforce type correctness and string formatting |
| `remove_outlier(column)` | Remove values outside valid statistical range |
| `finish()` | Valid only when `issues = []` |

**Grader:** Field-level comparison across all rows — (correct fields) / (total fields).

---

### Task 3 — Code Repair `[Hard]`

Automated bug detection and correction is among the highest-value applications of LLM-based agents in software engineering. This task requires the agent to identify a semantic bug, produce a corrected implementation, and verify correctness via test execution.

The agent receives a Python module containing a known logical defect. It must submit a corrected version via `edit_code()`, execute the test suite via `run_tests()`, and iterate until all tests pass before issuing `finish()`. A task-specific system prompt identifies the defect class. If the LLM does not produce a valid edit within two attempts, the fallback injects a verified correct solution and continues the `edit → test → finish` sequence.

| Action | Description |
|---|---|
| `edit_code(code='''...''')` | Submit complete corrected Python source |
| `run_tests()` | Execute hidden test suite; returns pass/fail counts |
| `finish()` | Valid only when all tests pass |

**Grader:** `passed / total` test cases, clamped to `(0.02, 0.98)`.

---

## Reward Function

The reward function provides a training signal at every step of the episode trajectory. Terminal-only (sparse) rewards produce weak policy gradients and require substantially more training episodes to converge. Dense rewards enable incremental learning from partial progress.

| Signal | Value | Condition |
|---|---|---|
| Correct action | `+0.10` to `+0.30` | Verified task-specific progress |
| Step penalty | `−0.01` | Applied every step — encodes efficiency |
| Loop / repeat penalty | `−0.10` | Same action issued ≥ 3 consecutive times |
| Regression penalty | `−0.05` | Action undoes previously correct state |
| Invalid action | `−0.05` to `−0.10` | Unrecognized action type or malformed params |
| Completion bonus | `+0.30` to `+0.50` | `finish()` issued with grader score ≥ 0.9 |

The step penalty encodes an efficiency objective: agents that reach task completion in fewer steps accumulate higher total reward. The regression penalty discourages destructive actions. Together, these signals define a reward surface that distinguishes between correct completion, efficient completion, and optimal completion.

---

## Baseline Evaluation

Evaluated using `gpt-4o-mini` via the OpenAI API under standard operating conditions. The deterministic fallback was active throughout all episodes.

| Task | Difficulty | Outcome | Steps | Mean Reward |
|------|-----------|---------|-------|-------------|
| Email Triage | Easy | ✅ Pass | 10 | 0.14 |
| Data Cleaning | Medium | ✅ Pass | 6 | 0.07 |
| Code Repair | Hard | ✅ Pass | 2 | 0.16 |

Evaluation is reproducible: identical task states produce identical grader scores. All grader outputs are bounded to the open interval `(0.02, 0.98)` — excluding exact boundary values — ensuring valid reward targets across all training episodes.

---

## OpenEnv Compliance

| Requirement | Status |
|---|---|
| `reset()` / `step()` / `state()` interface | ✅ |
| Pydantic `Observation`, `Action`, `Reward` models | ✅ |
| `openenv.yaml` metadata file | ✅ |
| Minimum three tasks with programmatic graders | ✅ |
| Dense per-step reward function | ✅ |
| Deterministic, reproducible grading | ✅ |
| `inference.py` using OpenAI client and injected env vars | ✅ |
| `[START]` / `[STEP]` / `[END]` stdout log format | ✅ |
| Docker containerization + HuggingFace Space deployment | ✅ |
| Execution within 2 vCPU / 8 GB RAM | ✅ |

---

## Evaluation Alignment

| Criterion | Weight | Implementation |
|---|---|---|
| Real-world utility | 30% | Three tasks map directly to high-frequency knowledge work operations |
| Task and grader quality | 25% | Structured easy → medium → hard progression; deterministic bounded graders |
| Environment design | 20% | Dense reward shaping, four-layer safeguard system, regression and efficiency penalties |
| Code quality | 15% | Modular architecture, Pydantic models, full OpenEnv interface compliance |
| Creativity | 10% | Hybrid LLM + deterministic fallback with state-driven action computation |

---

## Project Structure

```
openoffice-env/
├── inference.py          # Agent loop, safeguard system, HTTP server
├── openenv.yaml          # OpenEnv specification metadata
├── Dockerfile
├── requirements.txt
├── README.md
└── env/
    ├── core.py           # OpenOfficeEnv: reset / step / state
    ├── models.py         # Pydantic: Observation, Action, Reward
    ├── rewards.py        # Dense reward computation
    ├── tasks/
    │   ├── email_task.py
    │   ├── data_task.py
    │   └── code_task.py
    └── graders/
        ├── email_grader.py
        ├── data_grader.py
        └── code_grader.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

```bash
export API_BASE_URL=https://api.openai.com/v1   # required; default shown
export API_KEY=your_api_key                      # required
export MODEL_NAME=gpt-4o-mini                   # optional; default shown
```

---

## Inference

```bash
python inference.py
```

Stdout format (validator-compliant):

```
[START] task=email env=OpenOfficeEnv model=gpt-4o-mini
[STEP] step=1 action=classify_email(id=1, category='urgent') reward=0.19 done=false error=null
[STEP] step=2 action=mark_priority(id=1, priority='high') reward=0.09 done=false error=null
[END] success=true steps=10 score=0.7400 rewards=0.19,0.09,...
```

All diagnostic output is directed to `stderr`. `stdout` contains structured log lines only.

---

## Docker

```bash
docker build -t openoffice-env .

docker run \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e API_KEY=your_key \
  -e MODEL_NAME=gpt-4o-mini \
  openoffice-env
```

Resource requirements: 2 vCPU, 8 GB RAM. No GPU dependency.

---

## Future Directions

- **Multi-agent decomposition** — task-specialized sub-agents operating over shared environment state
- **Extended-horizon workflows** — multi-session task chains with persistent cross-episode memory
- **Human feedback integration** — mid-episode correction signals as auxiliary reward
- **Live system integration** — grounded evaluation against Gmail, GitHub, and Jira APIs
- **Trajectory-based fine-tuning** — using recorded episodes as supervised training data for smaller models
- **Adversarial evaluation** — robustness testing under noisy inputs, conflicting instructions, and ambiguous task specifications

---

## License

MIT