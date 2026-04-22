"""
ApprovalFlow — HR / Finance approval system mock.
Chaos: circular dependencies, expiring requests, OOO approvers, random rejections.
"""

import random
import copy
from typing import Any

VALID_APPROVAL_TYPES = {
    "vendor_onboarding":   {"required_role": "CFO",         "requires_data_keys": ["vendor_name", "compliance_score"]},
    "expense_report":      {"required_role": "CFO",         "requires_data_keys": ["amount", "employee_id", "report_id"]},
    "bug_escalation":      {"required_role": "CTO",         "requires_data_keys": ["ticket_id", "error_rate"]},
    "license_renewal":     {"required_role": "PROCUREMENT", "requires_data_keys": ["license_name", "expiry_days"]},
    "handbook_update":     {"required_role": "HR_LEAD",     "requires_data_keys": ["section", "change_summary"]},
}

# Tracks submitted approvals within an episode
class ApprovalFlow:
    """Mock approval system. Enforces data requirements and approver routing."""

    def __init__(self, noise_prob: float = 0.1):
        self.noise_prob = noise_prob
        self.approvals: dict[str, dict] = {}
        self._id_counter = 0
        self.action_log: list[dict] = []

    # ------------------------------------------------------------------ lifecycle

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)
        self.approvals = {}
        self._id_counter = 0
        self.action_log = []

    # ------------------------------------------------------------------ API

    def list_approval_types(self) -> dict[str, Any]:
        """List valid approval types and their required data keys."""
        self._log("list_approval_types", {}, len(VALID_APPROVAL_TYPES))
        return {
            "approval_types": {
                k: {
                    "required_role": v["required_role"],
                    "requires_data_keys": v["requires_data_keys"],
                }
                for k, v in VALID_APPROVAL_TYPES.items()
            }
        }

    def submit_approval(
        self, approval_type: str, approver: str, data: dict
    ) -> dict[str, Any]:
        """
        Submit an approval request.
        Will be rejected if:
          - approval_type is invalid
          - required data keys are missing
          - wrong approver is specified (e.g. using CFO when OOO)
          - random noise triggers a transient rejection
        """
        if self._random_failure():
            return {"error": "ApprovalFlow service unavailable. Retry."}

        spec = VALID_APPROVAL_TYPES.get(approval_type)
        if not spec:
            return {
                "error": f"Unknown approval type '{approval_type}'.",
                "hint": f"Valid types: {list(VALID_APPROVAL_TYPES.keys())}",
            }

        # Check required data keys
        missing_keys = [k for k in spec["requires_data_keys"] if k not in data]
        if missing_keys:
            return {
                "error": f"Missing required data fields: {missing_keys}",
                "hint": f"Required for '{approval_type}': {spec['requires_data_keys']}",
            }

        # Enforce OOO: CFO approvals must use delegate
        if spec["required_role"] == "CFO" and approver == "sarah.chen":
            return {
                "error": "Approver sarah.chen (CFO) is OOO.",
                "hint": "Route to CFO_DELEGATE: david.kim instead.",
            }

        self._id_counter += 1
        approval_id = f"APR-{self._id_counter:03d}"

        self.approvals[approval_id] = {
            "id": approval_id,
            "type": approval_type,
            "approver": approver,
            "data": copy.deepcopy(data),
            "status": "pending",
            "steps_until_decision": random.randint(1, 3),  # agent must poll
            "rejection_reason": None,
        }

        self._log("submit_approval", {"type": approval_type, "approver": approver}, approval_id)
        return {
            "success": True,
            "approval_id": approval_id,
            "message": "Approval submitted. Poll check_status() to track progress.",
        }

    def check_status(self, approval_id: str) -> dict[str, Any]:
        """
        Poll the status of a submitted approval.
        Advances the internal step counter; decision is made after N polls.
        """
        if self._random_failure():
            return {"error": "ApprovalFlow read timeout. Retry."}

        approval = self.approvals.get(approval_id)
        if not approval:
            return {"error": f"Approval {approval_id} not found."}

        if approval["status"] == "pending":
            approval["steps_until_decision"] -= 1

            if approval["steps_until_decision"] <= 0:
                # 80% approve, 20% reject (forces agent to handle rejection)
                if random.random() < 0.80:
                    approval["status"] = "approved"
                else:
                    approval["status"] = "rejected"
                    approval["rejection_reason"] = random.choice([
                        "Insufficient supporting data.",
                        "Budget threshold exceeded — needs secondary approval.",
                        "Policy violation detected — review required.",
                    ])

        self._log("check_status", {"approval_id": approval_id}, approval["status"])
        result = {
            "approval_id": approval_id,
            "status": approval["status"],
            "type": approval["type"],
        }
        if approval["status"] == "pending":
            result["message"] = "Still pending. Poll again."
        if approval["status"] == "rejected":
            result["rejection_reason"] = approval["rejection_reason"]
            result["hint"] = "Fix the issue and resubmit via submit_approval()."
        return result

    def escalate(self, approval_id: str, reason: str) -> dict[str, Any]:
        """
        Escalate a pending or rejected approval to a higher authority.
        Resets the approval to pending with a faster resolution.
        """
        if self._random_failure():
            return {"error": "Escalation service unavailable. Retry."}

        approval = self.approvals.get(approval_id)
        if not approval:
            return {"error": f"Approval {approval_id} not found."}

        if approval["status"] == "approved":
            return {"error": "Cannot escalate an already-approved request."}

        approval["status"] = "pending"
        approval["steps_until_decision"] = 1   # faster after escalation
        approval["rejection_reason"] = None
        approval["escalation_reason"] = reason

        self._log("escalate", {"approval_id": approval_id, "reason": reason}, "escalated")
        return {
            "success": True,
            "approval_id": approval_id,
            "message": "Escalated. Poll check_status() — decision expected sooner.",
        }

    def list_approvals(self) -> dict[str, Any]:
        """List all submitted approvals and their current statuses."""
        self._log("list_approvals", {}, len(self.approvals))
        return {
            "approvals": [
                {"id": a["id"], "type": a["type"], "status": a["status"]}
                for a in self.approvals.values()
            ]
        }

    # ------------------------------------------------------------------ helpers

    def _random_failure(self) -> bool:
        return random.random() < self.noise_prob

    def _log(self, method: str, args: dict, result_summary: Any) -> None:
        self.action_log.append(
            {"app": "ApprovalFlow", "method": method, "args": args, "result": result_summary}
        )

    def get_state_snapshot(self) -> dict:
        statuses = [a["status"] for a in self.approvals.values()]
        return {
            "total": len(statuses),
            "pending": statuses.count("pending"),
            "approved": statuses.count("approved"),
            "rejected": statuses.count("rejected"),
        }