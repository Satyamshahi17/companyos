"""
CompanyOS — OpenEnv-compliant data models.

Defines the typed Action, Observation, and State dataclasses that
the OpenEnv framework uses for serialisation and type-safe
client ↔ server communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openenv.core.env_server import Action, Observation, State


@dataclass
class CompanyAction(Action):
    """An action targeting one of the three CompanyOS apps."""
    app: str = ""           # "ticketdesk" | "datahub" | "approvalflow"
    command: str = ""       # e.g. "create", "resolve", "upload", "approve"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompanyObservation(Observation):
    """Observation returned after reset() or step()."""
    task: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    done: bool = False
    reward: float = 0.0
    last_result: dict[str, Any] | None = None
    ticket_desk: dict[str, Any] = field(default_factory=dict)
    data_hub: dict[str, Any] = field(default_factory=dict)
    approval_flow: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompanyState(State):
    """Episode metadata tracked by the environment."""
    episode_id: str = ""
    step_count: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
    task_type: str = ""
    task_description: str = ""
