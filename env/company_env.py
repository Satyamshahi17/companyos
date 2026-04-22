"""
CompanyOS — OpenEnv-compliant RL environment.
An agent must complete multi-step enterprise workflows across
TicketDesk, DataHub, and ApprovalFlow without exploiting shortcuts.

Observation: dict (task description + last action result + step info)
Action:      dict {"app": str, "method": str, "params": dict}
Reward:      shaped per-step + large terminal bonus on success
Done:        True when task complete or max_steps reached
"""

from __future__ import annotations

import random
from typing import Any

# OpenEnv base class — provides the gym-like interface contract
try:
    from openenv import Env  # type: ignore
except ImportError:
    # Fallback stub so the file is importable without openenv installed
    class Env:  # type: ignore
        def reset(self): raise NotImplementedError
        def step(self, action): raise NotImplementedError
        def render(self): raise NotImplementedError

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps import TicketDesk, DataHub, ApprovalFlow
from env.task_generator import TaskGenerator, Task


# ---------------------------------------------------------------------------
# Action routing table
# Maps (app_name, method_name) → callable on the app instance
# ---------------------------------------------------------------------------

def _build_router(td: TicketDesk, dh: DataHub, af: ApprovalFlow) -> dict:
    return {
        # TicketDesk
        ("ticketdesk", "list_tickets"):    lambda p: td.list_tickets(),
        ("ticketdesk", "search_tickets"):  lambda p: td.search_tickets(**p),
        ("ticketdesk", "get_ticket"):      lambda p: td.get_ticket(**p),
        ("ticketdesk", "update_ticket"):   lambda p: td.update_ticket(**p),
        # DataHub
        ("datahub", "list_metrics"):       lambda p: dh.list_metrics(),
        ("datahub", "query_metric"):       lambda p: dh.query_metric(**p),
        ("datahub", "refresh_data"):       lambda p: dh.refresh_data(**p),
        ("datahub", "get_approver"):       lambda p: dh.get_approver(**p),
        # ApprovalFlow
        ("approvalflow", "list_approval_types"): lambda p: af.list_approval_types(),
        ("approvalflow", "submit_approval"):     lambda p: af.submit_approval(**p),
        ("approvalflow", "check_status"):        lambda p: af.check_status(**p),
        ("approvalflow", "escalate"):            lambda p: af.escalate(**p),
        ("approvalflow", "list_approvals"):      lambda p: af.list_approvals(),
    }


class CompanyOSEnv(Env):
    """
    CompanyOS — Enterprise Workflow RL Environment.

    Example usage:
        env = CompanyOSEnv()
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
    """

    # Human-readable tool manifest (shown to agent in observation)
    TOOL_MANIFEST = {
        "ticketdesk": ["list_tickets", "search_tickets(query)", "get_ticket(ticket_id)",
                       "update_ticket(ticket_id, field, value)"],
        "datahub":    ["list_metrics", "query_metric(metric_name)", "refresh_data(metric_name)",
                       "get_approver(role)"],
        "approvalflow": ["list_approval_types", "submit_approval(approval_type, approver, data)",
                         "check_status(approval_id)", "escalate(approval_id, reason)",
                         "list_approvals"],
    }

    def __init__(self, noise_prob: float = 0.1, seed: int | None = None):
        self.noise_prob = noise_prob
        self.seed = seed

        self.ticketdesk   = TicketDesk(noise_prob=noise_prob)
        self.datahub      = DataHub(noise_prob=noise_prob)
        self.approvalflow = ApprovalFlow(noise_prob=noise_prob)
        self.task_gen     = TaskGenerator(seed=seed)

        # Episode state (set by reset)
        self.task: Task | None = None
        self.step_count: int = 0
        self.done: bool = False
        self.progress: dict[str, bool] = {}   # tracks partial completions
        self.last_result: dict = {}
        self.total_reward: float = 0.0

        self._router: dict = {}

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> dict[str, Any]:
        """
        Start a new episode.
        Optionally pin a specific task_id for reproducible evaluation.
        """
        seed = self.seed if self.seed is not None else random.randint(0, 99999)
        random.seed(seed)

        self.ticketdesk.reset(seed=seed)
        self.datahub.reset(seed=seed)
        self.approvalflow.reset(seed=seed)

        self.task = self.task_gen.get(task_id) if task_id else self.task_gen.sample()
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.last_result = {}

        # Initialise progress flags from task success conditions
        self.progress = {k: False for k in self.task.success_conditions}

        self._router = _build_router(self.ticketdesk, self.datahub, self.approvalflow)

        return self._build_obs("Episode started. Read the task and begin.")

    def step(self, action: dict[str, Any]) -> tuple[dict, float, bool, dict]:
        """
        Execute one action.

        action format:
            {
              "app":    "ticketdesk" | "datahub" | "approvalflow",
              "method": "<method_name>",
              "params": { ... }          # keyword args for the method
            }

        Returns: (observation, reward, done, info)
        """
        if self.done:
            return self._build_obs("Episode already finished. Call reset()."), 0.0, True, {}

        self.step_count += 1
        reward = -0.1  # small per-step cost encourages efficiency

        # ---- validate action structure ----
        app     = str(action.get("app", "")).lower()
        method  = str(action.get("method", "")).lower()
        params  = action.get("params", {})

        if not isinstance(params, dict):
            params = {}

        route_key = (app, method)
        handler = self._router.get(route_key)

        if handler is None:
            reward -= 1.0
            result = {
                "error": f"Unknown action ({app}, {method}).",
                "hint": "Check the tool manifest for valid app/method combinations.",
            }
        else:
            try:
                result = handler(params)
            except TypeError as e:
                reward -= 0.5
                result = {"error": f"Bad params for {method}: {e}"}

        self.last_result = result

        # ---- shaped reward logic ----
        reward += self._compute_shaped_reward(app, method, params, result)

        # ---- update progress flags ----
        self._update_progress(app, method, params, result)

        # ---- check terminal conditions ----
        success = all(self.progress.values())
        timeout = self.step_count >= self.task.max_steps

        if success:
            reward += 15.0   # large terminal bonus
            self.done = True
            obs = self._build_obs("TASK COMPLETE. All success conditions met.")
        elif timeout:
            reward -= 5.0
            self.done = True
            obs = self._build_obs("TIMEOUT. Max steps reached without completing task.")
        else:
            obs = self._build_obs()

        self.total_reward += reward

        info = {
            "step": self.step_count,
            "progress": dict(self.progress),
            "total_reward": self.total_reward,
            "success": success,
            "timeout": timeout,
        }

        return obs, round(reward, 3), self.done, info

    def render(self) -> dict[str, Any]:
        """Return a human-readable snapshot of the current env state."""
        return {
            "task": self.task.description if self.task else None,
            "step": self.step_count,
            "progress": self.progress,
            "total_reward": round(self.total_reward, 3),
            "ticketdesk_state":   self.ticketdesk.get_state_snapshot(),
            "datahub_state":      self.datahub.get_state_snapshot(),
            "approvalflow_state": self.approvalflow.get_state_snapshot(),
        }

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self, message: str = "") -> dict[str, Any]:
        assert self.task is not None
        return {
            "task": self.task.description,
            "step": self.step_count,
            "max_steps": self.task.max_steps,
            "steps_remaining": self.task.max_steps - self.step_count,
            "progress": dict(self.progress),
            "last_result": self.last_result,
            "message": message,
            "tools": self.TOOL_MANIFEST,
        }

    # ------------------------------------------------------------------
    # Shaped reward
    # ------------------------------------------------------------------

    def _compute_shaped_reward(
        self, app: str, method: str, params: dict, result: dict
    ) -> float:
        r = 0.0
        is_error = "error" in result

        if is_error:
            r -= 0.3   # penalise failed calls (noise or wrong usage)
            return r

        # TicketDesk rewards
        if app == "ticketdesk":
            if method == "update_ticket":
                field = params.get("field")
                if field == "priority" and params.get("value") is not None:
                    r += 1.0   # set a missing priority
                if field == "verified" and params.get("value") is True:
                    r += 1.5   # verified a ticket
                if field == "status" and params.get("value") == "open":
                    r += 1.0   # unblocked a ticket
                if field == "linked_approval":
                    r += 1.0   # linked an approval

        # DataHub rewards
        if app == "datahub":
            if method == "refresh_data" and result.get("success"):
                r += 1.0   # refreshed stale data
            if method == "query_metric" and not result.get("is_stale"):
                r += 0.5   # queried fresh data

        # ApprovalFlow rewards
        if app == "approvalflow":
            if method == "submit_approval" and result.get("success"):
                r += 2.0   # submitted a valid approval
            if method == "check_status":
                status = result.get("status")
                if status == "approved":
                    r += 3.0
                elif status == "rejected":
                    r -= 0.5   # mild penalty — agent should handle and escalate

        return r

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def _update_progress(
        self, app: str, method: str, params: dict, result: dict
    ) -> None:
        p = self.progress
        is_error = "error" in result

        if is_error:
            return

        ticket_id = self.task.ticket_id if self.task else None

        if app == "ticketdesk" and method == "update_ticket":
            if params.get("ticket_id") == ticket_id:
                field = params.get("field")
                value = params.get("value")
                if field == "priority" and value is not None and "ticket_priority_set" in p:
                    p["ticket_priority_set"] = True
                if field == "verified" and value is True and "ticket_verified" in p:
                    p["ticket_verified"] = True
                if field == "status" and value == "open" and "ticket_unblocked" in p:
                    p["ticket_unblocked"] = True
                if field == "assignee" and value and "ticket_assignee_set" in p:
                    p["ticket_assignee_set"] = True

        if app == "datahub":
            if method == "query_metric" and "metric_queried" in p:
                p["metric_queried"] = True
            if method == "refresh_data" and result.get("success") and "metric_refreshed" in p:
                p["metric_refreshed"] = True

        if app == "approvalflow":
            if method == "submit_approval" and result.get("success") and "approval_submitted" in p:
                p["approval_submitted"] = True
            if method == "check_status" and result.get("status") == "approved" and "approval_approved" in p:
                p["approval_approved"] = True