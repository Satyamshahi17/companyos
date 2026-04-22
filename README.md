---
title: CompanyOS
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# CompanyOS — Enterprise Workflow RL Environment

An OpenEnv-compliant RL environment where agents learn to navigate
a partially observable enterprise — completing multi-step workflows
across three interconnected mock apps.

## Apps
- **TicketDesk** — Jira-like task system (missing fields, blocked tickets)
- **DataHub** — Internal analytics warehouse (stale data, conflicting metrics)
- **ApprovalFlow** — HR/Finance approval system (OOO approvers, expiring requests)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Take one action |
| GET | `/render` | Inspect current state |
| GET | `/health` | Liveness probe |
| GET | `/manifest` | Tool manifest for agent prompting |

## Quick Start

```python
import requests

BASE = "https://your-space.hf.space"

# Start episode
obs = requests.post(f"{BASE}/reset").json()["observation"]
print(obs["task"])

# Take a step
result = requests.post(f"{BASE}/step", json={
    "app": "ticketdesk",
    "method": "list_tickets",
    "params": {}
}).json()

print(result["reward"], result["done"])
```

## Training

See `training/train.ipynb` for the full Unsloth/HF TRL training script.