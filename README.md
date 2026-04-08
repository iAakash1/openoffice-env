# 🏢 OpenOfficeEnv

> 🚀 OpenEnv RL Hackathon Submission — Real-World Productivity Benchmark

> A hybrid RL + LLM environment where an AI agent acts as an autonomous office worker —
> triaging emails, cleaning data, and fixing code.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)

---

## Overview

**OpenOfficeEnv** is a fully OpenEnv-compliant reinforcement learning environment that simulates
real-world productivity workflows. Unlike game-based or toy benchmarks, this environment requires
an agent to perform tasks that humans execute daily — reasoning about context, making decisions
under ambiguity, and iterating toward a correct solution.

The agent is a **hybrid system**: LLM-driven reasoning with a deterministic fallback layer that
prevents loops, stalls, and premature termination. This makes the system robust enough for
reproducible evaluation while still exercising genuine language model capability.

---

## 🧠 Why This Matters

Most RL environments are games. Real-world AI deployment is not.

| Problem | OpenOfficeEnv Solution |
|---|---|
| RL benchmarks are disconnected from real tasks | Tasks mirror actual productivity workflows |
| LLM agents fail silently with no recovery | Hybrid fallback system catches failures deterministically |
| Sparse rewards make learning hard | Dense per-step rewards with regression penalties |
| Agents loop or quit early | Loop detection + strict completion guards |

> **OpenOfficeEnv bridges the gap between LLM agents and RL training infrastructure.**

---

## ⚙️ Architecture

### Agent Loop

```
┌─────────────────────────────────────────────────────┐
│                   EPISODE LOOP                      │
│                                                     │
│  Observation → LLM Action                          │
│                    ↓                               │
│           Safeguard Layers:                        │
│           1. Loop detection    → fallback          │
│           2. Repeat detection  → fallback          │
│           3. Early finish guard → fallback         │
│           4. Completion guard  → block finish      │
│                    ↓                               │
│           env.step(action)                         │
│                    ↓                               │
│           Reward + Observation                     │
│                    ↓                               │
│           Continue until done                      │
└─────────────────────────────────────────────────────┘
```

### Hybrid Decision Policy

The agent is **not** a pure LLM. It operates in two modes:

- **LLM mode** — model generates action from structured observation + history prompt
- **Fallback mode** — deterministic policy activates when LLM loops, repeats, or exits early

For **email** and **data** tasks, the fallback computes the provably correct next action from
state alone. For the **code** task, the fallback scaffolds the `edit → test → iterate` sequence
and injects a known-good fix if the LLM fails twice.

---

## 🧪 Tasks

### Task 1 — Email Triage `[Easy]`

The agent receives 5 emails and must classify and prioritize each one before finishing.

| Action | Description |
|---|---|
| `classify_email(id, category)` | Assign category: urgent, billing, feature, social, general |
| `mark_priority(id, priority)` | Assign priority: high, medium, low |
| `finish()` | End episode — only valid when all emails are processed |

Classification uses deterministic keyword-based ground truth. The agent cannot finish
until `remaining` is empty.

---

### Task 2 — Data Cleaning `[Medium]`

The agent receives a corrupted dataset and must resolve all detected issues.

| Action | Description |
|---|---|
| `fill_missing(column)` | Impute missing values with column mean or default |
| `normalize_column(column)` | Cast types, title-case strings |
| `remove_outlier(column)` | Remove statistically invalid values |
| `finish()` | Only valid when `issues` list is empty |

Resolution is **state-driven** — the fallback parses issue strings to determine the exact
action needed at each step.

---

### Task 3 — Code Fixing `[Hard]`

The agent receives a Python script with a bug and must fix it to pass all hidden tests.

| Action | Description |
|---|---|
| `edit_code(code='''...''')` | Submit corrected Python code |
| `run_tests()` | Execute test suite against submitted code |
| `finish()` | Only valid when all tests pass |

The LLM is guided by a dedicated `CODE_SYSTEM_PROMPT` that names the exact bug. If the
LLM fails to call `edit_code()` on the first step, the system injects a known-good fix
and continues the `edit → test → finish` loop deterministically.

---

## 🎯 Reward Design

Rewards are **dense** — provided at every step, not just on completion. This is critical
for RL training signal quality.

| Signal | Value | Trigger |
|---|---|---|
| Correct action | `+0.1` to `+0.3` | Task-specific progress |
| Step penalty | `-0.01` | Every step — encourages efficiency |
| Repeat/loop penalty | `-0.10` | Same action taken 3+ times |
| Regression penalty | `-0.05` | Undoing previous progress |
| Wrong action | `-0.05` to `-0.10` | Invalid or incorrect action |
| Completion bonus | `+0.30` to `+0.50` | `finish()` with passing grade |

Dense rewards ensure the agent receives meaningful gradient signal throughout the
trajectory — not just a binary win/loss at the end.

---

## 📊 Baseline Results

Evaluated using `gpt-4o-mini` via the OpenAI API.

| Task | Success | Steps | Avg Reward |
|------|---------|-------|------------|
| Email | ✅ Yes | 10 | 0.14 |
| Data | ✅ Yes | 6 | 0.07 |
| Code | ✅ Yes | 2 | 0.16 |

---

## 🗂 Project Structure

```
openoffice-env/
├── inference.py              # Agent loop + hybrid control system
├── openenv.yaml              # OpenEnv spec metadata
├── Dockerfile                # Container definition
├── requirements.txt
├── README.md
└── env/
    ├── core.py               # OpenOfficeEnv: reset / step / state
    ├── models.py             # Pydantic: Observation, Action, Reward
    ├── rewards.py            # Dense reward shaping
    ├── tasks/
    │   ├── email_task.py
    │   ├── data_task.py
    │   └── code_task.py
    └── graders/
        ├── email_grader.py   # Score 0.0 → 1.0
        ├── data_grader.py
        └── code_grader.py
```

---

## 🔧 Setup

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export HF_TOKEN=your_huggingface_token          # required
export API_BASE_URL=https://api.openai.com/v1   # optional, default shown
export MODEL_NAME=gpt-4o-mini                   # optional, default shown
```

---

## 🚀 Running Inference

```bash
python inference.py
```

Output is written to stdout in strict format:

```
[START] task=email env=OpenOfficeEnv model=gpt-4o-mini
[STEP] step=1 action=classify_email(id=1, category='urgent') reward=0.19 done=false error=null
[STEP] step=2 action=mark_priority(id=1, priority='high') reward=0.09 done=false error=null
...
[END] success=true steps=10 rewards=0.19,0.09,...
```

- One `[START]` per episode
- One `[STEP]` per environment step
- One `[END]` always emitted, even on exception
- Debug output goes to `stderr` only

---

## 🐳 Docker

Build:

```bash
docker build -t openoffice-env .
```

Run:

```bash
docker run -e HF_TOKEN=your_token openoffice-env
```

With custom model:

```bash
docker run \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  openoffice-env
```

Designed to run within **2 vCPU / 8GB RAM**. No GPU required.

---

## ✅ OpenEnv Compliance

| Requirement | Status |
|---|---|
| `reset()` / `step()` / `state()` interface | ✅ |
| Pydantic Observation, Action, Reward models | ✅ |
| `openenv.yaml` with metadata | ✅ |
| Minimum 3 tasks with graders | ✅ |
| Dense reward function | ✅ |
| Deterministic, reproducible grading | ✅ |
| `inference.py` with OpenAI client | ✅ |
| `[START]` / `[STEP]` / `[END]` log format | ✅ |
| Containerized + HuggingFace deployable | ✅ |

---

## 🧩 Key Innovation

The core contribution of OpenOfficeEnv is the **hybrid agent architecture**:

- Pure LLM agents fail due to loops, repetition, and premature termination
- Pure rule-based systems lack generalization

OpenOfficeEnv combines both:

- **LLM** for reasoning, language understanding, and flexible decision making
- **Deterministic fallback** for correctness, stability, and guaranteed progression

This enables:

- Reliable convergence across all three tasks
- Reproducible evaluation with consistent scoring
- Real-world applicability beyond synthetic benchmarks

This hybrid design is the key step toward practical RL + LLM systems that can operate
reliably in production environments.

---

## 🔭 Future Work

- **Multi-agent collaboration** — split tasks across specialized agents with shared state
- **Longer workflows** — chain tasks into multi-session projects with persistent memory
- **Human-in-the-loop** — inject human corrections mid-episode as reward signal
- **Real API integrations** — connect to live Gmail, GitHub, or Jira for grounded evaluation
- **Fine-tuning pipeline** — use episode trajectories to fine-tune smaller models on office tasks
- **Adversarial tasks** — introduce noise, conflicting instructions, and ambiguous inputs

---

## License

MIT