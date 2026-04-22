"""
TicketDesk — Jira-like mock app.
Chaos: missing fields, wrong priorities, permission errors, blocked states.
"""

import random
import copy
from typing import Any

TICKET_TEMPLATES = [
    {
        "id": "T-001",
        "title": "Onboard vendor ACME Corp",
        "status": "open",
        "priority": None,           # intentionally missing
        "assignee": "unknown",
        "tags": ["vendor", "compliance"],
        "linked_approval": None,
        "verified": False,
    },
    {
        "id": "T-002",
        "title": "Process expense report #447",
        "status": "open",
        "priority": "high",
        "assignee": "john.doe",
        "tags": ["finance", "expense"],
        "linked_approval": None,
        "verified": False,
    },
    {
        "id": "T-003",
        "title": "Escalate critical bug in auth service",
        "status": "blocked",        # intentionally blocked
        "priority": "critical",
        "assignee": None,           # intentionally missing
        "tags": ["engineering", "bug"],
        "linked_approval": None,
        "verified": False,
    },
    {
        "id": "T-004",
        "title": "Renew software license for Q3",
        "status": "open",
        "priority": "medium",
        "assignee": "jane.smith",
        "tags": ["procurement", "license"],
        "linked_approval": None,
        "verified": False,
    },
    {
        "id": "T-005",
        "title": "Update employee handbook section 4",
        "status": "open",
        "priority": None,           # intentionally missing
        "assignee": "hr.team",
        "tags": ["hr", "documentation"],
        "linked_approval": None,
        "verified": False,
    },
]


class TicketDesk:
    """Mock Jira-like ticketing system. State is reset fresh each episode."""

    def __init__(self, noise_prob: float = 0.1):
        self.noise_prob = noise_prob
        self.tickets: dict[str, dict] = {}
        self.action_log: list[dict] = []

    # ------------------------------------------------------------------ lifecycle

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)
        self.tickets = {t["id"]: copy.deepcopy(t) for t in TICKET_TEMPLATES}
        self.action_log = []

    # ------------------------------------------------------------------ API

    def list_tickets(self) -> dict[str, Any]:
        """List all ticket IDs and titles."""
        self._log("list_tickets", {}, len(self.tickets))
        return {
            "tickets": [
                {"id": t["id"], "title": t["title"], "status": t["status"]}
                for t in self.tickets.values()
            ]
        }

    def search_tickets(self, query: str) -> dict[str, Any]:
        """Search tickets by keyword. Returns partial info only."""
        if self._random_failure():
            return {"error": "TicketDesk API timeout. Retry."}

        q = query.lower()
        results = [
            {"id": t["id"], "title": t["title"], "status": t["status"], "tags": t["tags"]}
            for t in self.tickets.values()
            if q in t["title"].lower() or any(q in tag for tag in t["tags"])
        ]
        self._log("search_tickets", {"query": query}, len(results))
        return {"results": results} if results else {
            "results": [], "hint": "No tickets found. Try broader keywords."
        }

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Get full details of a single ticket."""
        if self._random_failure():
            return {"error": "TicketDesk read error. Retry."}
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            return {"error": f"Ticket {ticket_id} not found."}
        self._log("get_ticket", {"ticket_id": ticket_id}, "ok")
        return {"ticket": copy.deepcopy(ticket)}

    def update_ticket(self, ticket_id: str, field: str, value: Any) -> dict[str, Any]:
        """Update a field on a ticket."""
        if self._random_failure():
            return {"error": "Permission denied. Try again."}
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            return {"error": f"Ticket {ticket_id} not found."}

        # Business rule: must verify before linking approval
        if field == "linked_approval" and not ticket["verified"]:
            return {
                "error": "Cannot link approval: ticket must be verified first.",
                "hint": "Set verified=True before linking an approval.",
            }
        # Business rule: cannot close a blocked ticket directly
        if field == "status" and value == "closed" and ticket["status"] == "blocked":
            return {
                "error": "Cannot close a blocked ticket.",
                "hint": "Update status to 'open' first to unblock.",
            }

        allowed = {"priority", "assignee", "status", "verified", "linked_approval"}
        if field not in allowed:
            return {"error": f"Field '{field}' not editable. Allowed: {sorted(allowed)}"}

        ticket[field] = value
        self._log("update_ticket", {"ticket_id": ticket_id, "field": field, "value": value}, "ok")
        return {"success": True, "ticket_id": ticket_id, "updated": {field: value}}

    # ------------------------------------------------------------------ helpers

    def _random_failure(self) -> bool:
        return random.random() < self.noise_prob

    def _log(self, method: str, args: dict, result_summary: Any) -> None:
        self.action_log.append(
            {"app": "TicketDesk", "method": method, "args": args, "result": result_summary}
        )

    def get_state_snapshot(self) -> dict:
        return {
            "open": sum(1 for t in self.tickets.values() if t["status"] == "open"),
            "blocked": sum(1 for t in self.tickets.values() if t["status"] == "blocked"),
            "verified": sum(1 for t in self.tickets.values() if t["verified"]),
        }