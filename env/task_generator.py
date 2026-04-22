"""
TaskGenerator — produces one enterprise workflow goal per episode.
Each task maps to a required sequence of app interactions.
"""

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    task_id: str
    description: str          # what the agent sees
    ticket_id: str            # which TicketDesk ticket is involved
    approval_type: str        # which ApprovalFlow type to submit
    required_metric: str      # which DataHub metric must be checked
    required_approver_role: str
    success_conditions: dict[str, Any]   # checked by env to give final reward
    max_steps: int = 20


TASK_TEMPLATES = [
    Task(
        task_id="task_vendor_onboarding",
        description=(
            "Complete vendor onboarding for ACME Corp. "
            "Verify the ticket, check compliance data, and submit CFO approval."
        ),
        ticket_id="T-001",
        approval_type="vendor_onboarding",
        required_metric="vendor_compliance_score",
        required_approver_role="CFO_DELEGATE",   # CFO is OOO — must find delegate
        success_conditions={
            "ticket_verified": True,
            "ticket_priority_set": True,
            "metric_refreshed": True,
            "approval_submitted": True,
            "approval_approved": True,
        },
        max_steps=20,
    ),
    Task(
        task_id="task_expense_report",
        description=(
            "Process expense report #447 for employee john.doe. "
            "Verify the ticket, confirm the amount from DataHub, and get CFO approval."
        ),
        ticket_id="T-002",
        approval_type="expense_report",
        required_metric="expense_report_447_amount",
        required_approver_role="CFO_DELEGATE",
        success_conditions={
            "ticket_verified": True,
            "metric_queried": True,
            "approval_submitted": True,
            "approval_approved": True,
        },
        max_steps=18,
    ),
    Task(
        task_id="task_bug_escalation",
        description=(
            "Escalate critical bug ticket T-003 to engineering. "
            "Unblock the ticket, pull error rate from monitoring, submit CTO escalation."
        ),
        ticket_id="T-003",
        approval_type="bug_escalation",
        required_metric="auth_service_error_rate",
        required_approver_role="CTO",
        success_conditions={
            "ticket_unblocked": True,
            "ticket_assignee_set": True,
            "metric_queried": True,
            "approval_submitted": True,
            "approval_approved": True,
        },
        max_steps=18,
    ),
    Task(
        task_id="task_license_renewal",
        description=(
            "Renew the Q3 software license before it expires. "
            "Check expiry days from DataHub and get Procurement approval."
        ),
        ticket_id="T-004",
        approval_type="license_renewal",
        required_metric="software_license_expiry_days",
        required_approver_role="PROCUREMENT",
        success_conditions={
            "ticket_verified": True,
            "metric_queried": True,
            "approval_submitted": True,
            "approval_approved": True,
        },
        max_steps=16,
    ),
    Task(
        task_id="task_handbook_update",
        description=(
            "Process the HR handbook update for section 4. "
            "Set priority on the ticket, verify it, and get HR Lead approval."
        ),
        ticket_id="T-005",
        approval_type="handbook_update",
        required_metric="employee_count_hr_dept",
        required_approver_role="HR_LEAD",
        success_conditions={
            "ticket_priority_set": True,
            "ticket_verified": True,
            "approval_submitted": True,
            "approval_approved": True,
        },
        max_steps=16,
    ),
]


class TaskGenerator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._pool = list(TASK_TEMPLATES)

    def sample(self) -> Task:
        """Return a random task."""
        return self.rng.choice(self._pool)

    def get(self, task_id: str) -> Task | None:
        return next((t for t in self._pool if t.task_id == task_id), None)

    @property
    def all_task_ids(self) -> list[str]:
        return [t.task_id for t in self._pool]