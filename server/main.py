"""
CompanyOS FastAPI server.
Exposes the RL environment over HTTP so Colab training scripts
can call it remotely — no local install needed on the trainer.

Endpoints:
  POST /reset          start a new episode
  POST /step           take an action
  GET  /render         inspect current state
  GET  /health         liveness probe
  GET  /manifest       tool manifest (for agent prompting)
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from env.company_env import CompanyOSEnv

app = FastAPI(
    title="CompanyOS",
    description="Enterprise Workflow RL Environment — OpenEnv compliant",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single env instance per server process.
# For parallel training, spin up multiple server replicas.
env = CompanyOSEnv(noise_prob=0.1)


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None

class StepRequest(BaseModel):
    app: str
    method: str
    params: dict[str, Any] = {}

class ResetResponse(BaseModel):
    observation: dict[str, Any]

class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()):
    """Start a new episode. Optionally pin task_id and seed."""
    if req.seed is not None:
        env.seed = req.seed
    obs = env.reset(task_id=req.task_id)
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """
    Take one action in the environment.

    Example body:
        {
          "app": "ticketdesk",
          "method": "search_tickets",
          "params": {"query": "vendor"}
        }
    """
    action = {"app": req.app, "method": req.method, "params": req.params}
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/render")
def render() -> dict[str, Any]:
    """Return a human-readable snapshot of the current environment state."""
    return env.render()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": "CompanyOS", "version": "0.1.0"}


@app.get("/manifest")
def manifest() -> dict[str, Any]:
    """Return the full tool manifest — useful for agent system prompts."""
    return {
        "tools": CompanyOSEnv.TOOL_MANIFEST,
        "action_format": {
            "app":    "ticketdesk | datahub | approvalflow",
            "method": "method name from manifest",
            "params": "dict of keyword arguments",
        },
    }

@app.get("/")
def root():
    return {"message": "CompanyOS is live!", "docs": "/docs", "health": "/health"}