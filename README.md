---
title: CompanyOS
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🏢 CompanyOS — Enterprise Workflow RL Environment

> *"Every day, millions of employees navigate chaotic enterprise systems — outdated tools, conflicting data, approval loops. What if we trained an AI agent to survive this chaos?"*

**CompanyOS** is an OpenEnv-compliant reinforcement learning environment where agents learn to complete multi-step enterprise workflows across three interconnected, partially observable mock applications — TicketDesk, DataHub, and ApprovalFlow.


[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://github.com/meta-pytorch/OpenEnv)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-orange)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-blue)](https://huggingface.co/spaces/satyamshahi/companyos)

---

## 🧩 The Problem

Enterprise AI agents fail in the real world not because they lack intelligence — but because real enterprise systems are **messy, inconsistent, and interdependent**.

Real enterprise chaos looks like this:
- A ticket exists but has **no priority set** and the **assignee is missing**
- Compliance data is **4 days stale** and must be refreshed before it can be trusted
- The CFO who needs to approve is **out of office** — the agent must discover the delegate
- An approval gets **randomly rejected** — the agent must escalate and retry
- Actions have **irreversible consequences** — closing a blocked ticket causes a penalty

Existing RL benchmarks test agents on clean, isolated tasks. CompanyOS tests agents on **the full messy workflow** — just like the real world.

---

## 💡 The Solution

CompanyOS provides a **controlled, reproducible benchmark** of enterprise chaos.

Think of it like ARC-AGI for enterprise reasoning — a faithful abstraction of real workflows that:
- **Resets cleanly** between episodes (impossible with real Jira/SAP/Workday)
- **Scales to thousands of episodes** without rate limits
- **Injects controlled chaos** at tunable difficulty levels
- **Rewards causal reasoning** — not pattern matching or shortcutting

The agent must maintain a **persistent world model** across multiple apps, update its beliefs based on tool call outcomes, and orchestrate the correct sequence of actions to complete the task.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **RL Framework** | [OpenEnv-core](https://github.com/meta-pytorch/OpenEnv) `>=0.2.0` | Environment interface — `reset()`, `step()`, `render()` |
| **API Server** | [FastAPI](https://fastapi.tiangolo.com) `>=0.110` | Exposes env over HTTP — `/reset`, `/step`, `/render` |
| **Server Runtime** | [Uvicorn](https://www.uvicorn.org) | ASGI server for FastAPI |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev) `v2` | Request/response schemas |
| **Containerisation** | [Docker](https://docker.com) | Reproducible deployment |
| **Deployment** | [HuggingFace Spaces](https://huggingface.co/spaces) | Live hosted environment |
| **Model Training** | [Unsloth](https://github.com/unslothai/unsloth) | 4-bit quantised LLM fine-tuning |
| **RL Training** | [HuggingFace TRL](https://github.com/huggingface/trl) — GRPO | Policy optimisation |
| **Base Model** | [Qwen2.5-1.5B-Instruct](https://huggingface.co/unsloth/Qwen2.5-1.5B-Instruct) | Small, fast, trainable on T4/A10G |
| **Experiment Tracking** | [Weights & Biases](https://wandb.ai) | Reward curves, loss logging |
| **Language** | Python 3.11 | Core implementation |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Agent (LLM)                       │
│  Receives observation → Reasons → Picks action      │
└──────────────────────┬──────────────────────────────┘
                       │  action: {app, method, params}
                       ▼
┌─────────────────────────────────────────────────────┐
│            CompanyOSEnv  (OpenEnv compliant)        │
│                                                     │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ TicketDesk  │  │ DataHub  │  │ ApprovalFlow  │  │
│  │ (Jira-like) │  │(Analytics│  │ (HR/Finance)  │  │
│  └─────────────┘  └──────────┘  └───────────────┘  │
│                                                     │
│       Returns: (observation, reward, done, info)    │
└──────────────────────┬──────────────────────────────┘
                       │  served over HTTP
                       ▼
┌─────────────────────────────────────────────────────┐
│         FastAPI Server  (port 7860)                 │
│   POST /reset  │  POST /step  │  GET /render        │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         HuggingFace Spaces  (Docker SDK)            │
│   Training script in Colab calls the Space API      │
└─────────────────────────────────────────────────────┘
```

---

## 🖥️ The Three Apps

### 1. 🗂️ TicketDesk — *The Jira-like Task System*

Manages enterprise tickets. Each ticket has a status, priority, assignee, and verification state.

**Intentional chaos injected:**

| Problem | What the agent must do |
|---|---|
| `priority: null` | Discover and set the missing priority |
| `status: "blocked"` | Unblock before any further action |
| `assignee: "unknown"` | Find and assign the correct person |
| Cannot link approval unless `verified: True` | Must verify ticket first — order matters |
| Cannot close a blocked ticket directly | Must understand state dependencies |

**Available tools:**
```
list_tickets()                           → see all tickets (partial info only)
search_tickets(query)                    → keyword search
get_ticket(ticket_id)                    → full ticket details
update_ticket(ticket_id, field, value)   → modify a field
```

---

### 2. 📊 DataHub — *The Internal Analytics Warehouse*

Stores business metrics and the approver directory. Some data is stale and must be refreshed before use.

**Intentional chaos injected:**

| Problem | What the agent must do |
|---|---|
| `vendor_compliance_score` is 4 days stale | Call `refresh_data()` before trusting the value |
| Score is 72 (below 80 threshold) before refresh | After refresh it becomes 85 — agent must re-query |
| CFO is marked OOO in `approver_directory` | Must discover the CFO delegate via `get_approver()` |
| `employee_count` is 7 days stale | Must detect staleness from the `is_stale` flag |

**Available tools:**
```
list_metrics()                → see available metric names
query_metric(metric_name)     → fetch a value (warns if stale)
refresh_data(metric_name)     → trigger pipeline refresh
get_approver(role)            → look up who approves for a role
```

---

### 3. ✅ ApprovalFlow — *The HR/Finance Approval System*

Handles approval requests for vendor onboarding, expenses, escalations, and more.

**Intentional chaos injected:**

| Problem | What the agent must do |
|---|---|
| CFO (sarah.chen) is OOO | Route to CFO_DELEGATE (david.kim) instead |
| Missing required data fields → rejection | Must gather all required data before submitting |
| Approval takes N polls to resolve | Must poll `check_status()` until decision |
| 20% random rejection rate | Must handle rejection by calling `escalate()` |
| Cannot escalate an already-approved request | Must check status before escalating |

**Available tools:**
```
list_approval_types()                              → see valid types + required fields
submit_approval(approval_type, approver, data)     → submit a request
check_status(approval_id)                          → poll for decision
escalate(approval_id, reason)                      → escalate a rejected request
list_approvals()                                   → see all submitted approvals
```

---

## 🤖 How the Agent Interacts

Each episode the agent receives a **task description** and must complete it by calling tools across the three apps. The agent only sees its **current observation** — it cannot peek at the full environment state.

### Observation (what the agent sees each step):
```json
{
  "task": "Complete vendor onboarding for ACME Corp. Verify the ticket, check compliance data, and submit CFO approval.",
  "step": 3,
  "max_steps": 20,
  "steps_remaining": 17,
  "progress": {
    "ticket_priority_set": true,
    "ticket_verified": false,
    "metric_refreshed": false,
    "approval_submitted": false,
    "approval_approved": false
  },
  "last_result": {
    "success": true,
    "ticket_id": "T-001",
    "updated": {"priority": "high"}
  },
  "tools": { "ticketdesk": [...], "datahub": [...], "approvalflow": [...] }
}
```

### Action (what the agent outputs each step):
```json
{
  "app": "ticketdesk",
  "method": "update_ticket",
  "params": {
    "ticket_id": "T-001",
    "field": "verified",
    "value": true
  }
}
```

### A typical successful episode trace:
```
Step 1:  ticketdesk.get_ticket(T-001)
         → sees priority is null, assignee is unknown

Step 2:  ticketdesk.update_ticket(T-001, priority, "high")
         → reward: +1.0  ✅ progress: ticket_priority_set

Step 3:  ticketdesk.update_ticket(T-001, verified, True)
         → reward: +1.5  ✅ progress: ticket_verified

Step 4:  datahub.query_metric(vendor_compliance_score)
         → returns value=72, is_stale=True ⚠️ WARNING: refresh needed

Step 5:  datahub.refresh_data(vendor_compliance_score)
         → reward: +1.0  ✅ progress: metric_refreshed

Step 6:  datahub.query_metric(vendor_compliance_score)
         → returns value=85, is_stale=False ✅ now trustworthy

Step 7:  datahub.get_approver(CFO)
         → WARNING: CFO is OOO. Use CFO_DELEGATE: david.kim

Step 8:  approvalflow.submit_approval(vendor_onboarding, david.kim, {...})
         → reward: +2.0  ✅ progress: approval_submitted

Step 9:  approvalflow.check_status(APR-001)
         → status: pending. Poll again.

Step 10: approvalflow.check_status(APR-001)
         → status: approved
         → reward: +3.0 + terminal bonus +15.0 🎉 TASK COMPLETE
```

**What makes this hard for an untrained agent:**
A naive random agent scores around **-2.04 mean reward** with **~2% success rate** because it:
- Calls tools in the wrong order (tries to link approval before verifying ticket)
- Ignores stale data warnings and submits with bad compliance scores
- Routes approvals to the OOO CFO instead of the delegate
- Randomly closes blocked tickets causing penalties

A trained agent learns to **model the world state** — tracking what it knows, what it still needs to discover, and what order operations must happen in.

---

## 🎯 The 5 Tasks

| Task | Ticket | Key Challenge | Required Approver |
|---|---|---|---|
| Vendor Onboarding | T-001 | Stale compliance score, OOO CFO | CFO_DELEGATE |
| Expense Report | T-002 | Confirm amount from DataHub | CFO_DELEGATE |
| Bug Escalation | T-003 | Blocked ticket, missing assignee | CTO |
| License Renewal | T-004 | Expiry days metric check | PROCUREMENT |
| Handbook Update | T-005 | Missing priority field | HR_LEAD |

---

## 🏆 Reward System

CompanyOS uses **shaped rewards** to provide a dense learning signal throughout each episode, plus a large terminal bonus on task completion.

### Per-step rewards:

| Action | Reward | Rationale |
|---|---|---|
| Set missing ticket priority | **+1.0** | Addresses a real data quality gap |
| Verify a ticket | **+1.5** | Required prerequisite — order matters |
| Unblock a blocked ticket | **+1.0** | Resolves a critical state dependency |
| Refresh stale metric | **+1.0** | Demonstrates world model awareness |
| Query fresh (non-stale) data | **+0.5** | Rewards using reliable information |
| Submit valid approval | **+2.0** | Core workflow milestone |
| Approval gets approved | **+3.0** | Task-critical outcome |
| Failed API call (noise/wrong usage) | **-0.3** | Penalises sloppy tool use |
| Approval gets rejected | **-0.5** | Mild — agent should escalate |
| Invalid action (hallucinated tool) | **-1.0** | Penalises made-up methods |
| Per step (time pressure) | **-0.1** | Encourages efficiency |

### Terminal rewards:

| Outcome | Reward |
|---|---|
| ✅ All success conditions met | **+15.0** |
| ⏱️ Max steps reached (timeout) | **-5.0** |

### Why this reward design works for RL training:

1. **Dense signal** — agent gets feedback every step, not just at episode end
2. **No shortcutting** — you cannot get the +15 terminal bonus without hitting every milestone
3. **Clear learning curve** — random agent scores ~-2, trained agent converges toward +15–25
4. **Interpretable** — every reward increment maps directly to a real business action

---

## 📈 Training Results

### Random Baseline (untrained agent):
```
Mean reward:   -2.04
Success rate:   2.0%
Best episode: +21.50  (lucky random sequence)
```

### After GRPO Training (Unsloth + HF TRL):
```
Mean reward:  ~+12–18   (converges after ~150 episodes)
Success rate:  ~60–75%
```

The reward curve shows clear learning progression:
1. **Phase 1** — Agent stops making invalid tool calls (reward stabilises above -5)
2. **Phase 2** — Agent learns correct app routing for each subtask
3. **Phase 3** — Agent learns full workflow order including stale data and OOO routing

---

## 🚀 API Reference

Base URL: `https://satyamshahi-companyos.hf.space`

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Take one action |
| GET | `/render` | Inspect current state |
| GET | `/health` | Liveness probe |
| GET | `/manifest` | Tool manifest for agent prompting |
| GET | `/docs` | Interactive Swagger UI |

### Quick example:
```python
import requests

ENV_URL = "https://satyamshahi-companyos.hf.space"

# Start episode
obs = requests.post(f"{ENV_URL}/reset", json={"task_id": "task_vendor_onboarding"}).json()
print("Task:", obs["observation"]["task"])

# Take a step
result = requests.post(f"{ENV_URL}/step", json={
    "app": "ticketdesk",
    "method": "list_tickets",
    "params": {}
}).json()
print(f"Reward: {result['reward']}  Done: {result['done']}")
```

---

## 🏋️ Training

See `training/train.ipynb` for the full GRPO training script (Unsloth + HF TRL).

```python
# Run random baseline to get the BEFORE curve
!python training/random_baseline.py \
  --episodes 100 \
  --env-url https://satyamshahi-companyos.hf.space \
  --output baseline_curve.png
```

---

## 📁 Project Structure

```
companyos/
├── apps/
│   ├── ticketdesk.py        # Jira-like mock (missing fields, blocked tickets)
│   ├── datahub.py           # Analytics warehouse (stale data, OOO approvers)
│   └── approvalflow.py      # Approvals (random rejections, circular deps)
├── env/
│   ├── company_env.py       # OpenEnv-compliant environment core
│   └── task_generator.py    # 5 enterprise workflow tasks
├── server/
│   └── main.py              # FastAPI REST server
├── training/
│   ├── train.ipynb          # GRPO training notebook (Unsloth + TRL)
│   └── random_baseline.py   # Random agent baseline for comparison
├── tests/
│   └── test_env.py          # 38 tests, all passing
├── Dockerfile               # python:3.11-slim, port 7860
├── docker-compose.yaml
└── pyproject.toml
```

---

## 🔬 Why CompanyOS is a Strong RL Benchmark

| Property | CompanyOS | Typical benchmark |
|---|---|---|
| Partial observability | ✅ Agent sees only current state | ❌ Often fully observable |
| Multi-app state dependencies | ✅ 3 interconnected systems | ❌ Usually single system |
| Irreversible actions | ✅ Wrong approval = penalty | ❌ Usually reversible |
| Non-stationary world | ✅ Stale data, OOO approvers | ❌ Static state |
| Reproducible episodes | ✅ Seed-based deterministic reset | ❌ Hard with real APIs |
| Dense reward signal | ✅ Shaped per-step rewards | ❌ Often sparse terminal only |
| Real business semantics | ✅ Faithful enterprise abstractions | ❌ Toy problems |

---

## 👨‍💻 Built By

**Satyam Kumar** 