"""
DataHub — Internal analytics + data warehouse mock.
Chaos: stale metrics, conflicting values, refresh pipeline delays.
"""

import random
import copy
from typing import Any

STALE_DAYS = 2  # anything older is considered unreliable

METRIC_TEMPLATES = {
    "vendor_compliance_score": {
        "value": 72,
        "unit": "percent",
        "stale_days": 4,            # older than threshold → stale
        "source": "compliance_db",
        "threshold_pass": 80,
        "note": "Score below threshold — manual review required.",
    },
    "expense_report_447_amount": {
        "value": 4350.00,
        "unit": "USD",
        "stale_days": 0,
        "source": "finance_system",
        "threshold_pass": None,
        "note": None,
    },
    "auth_service_error_rate": {
        "value": 18.5,
        "unit": "percent",
        "stale_days": 0,
        "source": "monitoring",
        "threshold_pass": 5,
        "note": "Above SLA. Escalation required.",
    },
    "software_license_expiry_days": {
        "value": 12,
        "unit": "days",
        "stale_days": 1,
        "source": "procurement_db",
        "threshold_pass": None,
        "note": "Expires soon. Renewal approval needed.",
    },
    "employee_count_hr_dept": {
        "value": 24,
        "unit": "headcount",
        "stale_days": 7,            # very stale
        "source": "hr_system",
        "threshold_pass": None,
        "note": "Data may be outdated — refresh before use.",
    },
    "approver_directory": {
        "value": {
            "CFO": "sarah.chen",
            "CTO": "mike.ross",
            "HR_LEAD": "priya.nair",
            "PROCUREMENT": "tom.baker",
            "CFO_DELEGATE": "david.kim",   # CFO is OOO
        },
        "unit": "directory",
        "stale_days": 0,
        "source": "hr_system",
        "threshold_pass": None,
        "note": "CFO sarah.chen is OOO. Route CFO approvals to CFO_DELEGATE.",
    },
}

# Values that get updated after a refresh (simulates pipeline re-run)
REFRESH_OVERRIDES = {
    "vendor_compliance_score": {"value": 85, "note": "Refreshed. Score now above threshold."},
    "employee_count_hr_dept":  {"value": 31, "note": "Refreshed. Headcount updated to 31."},
}


class DataHub:
    """Mock internal data warehouse. Agents must detect and resolve stale data."""

    def __init__(self, noise_prob: float = 0.1):
        self.noise_prob = noise_prob
        self.metrics: dict[str, dict] = {}
        self.action_log: list[dict] = []

    # ------------------------------------------------------------------ lifecycle

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)
        self.metrics = copy.deepcopy(METRIC_TEMPLATES)
        self.action_log = []

    # ------------------------------------------------------------------ API

    def list_metrics(self) -> dict[str, Any]:
        """List all available metric keys."""
        self._log("list_metrics", {}, len(self.metrics))
        return {
            "available_metrics": list(self.metrics.keys()),
            "hint": "Use query_metric(name) to fetch a specific value.",
        }

    def query_metric(self, metric_name: str) -> dict[str, Any]:
        """
        Fetch a metric. Stale metrics include a warning.
        Agent should call refresh_data() before trusting stale values.
        """
        if self._random_failure():
            return {"error": "DataHub query timeout. Retry."}

        metric = self.metrics.get(metric_name)
        if not metric:
            return {
                "error": f"Metric '{metric_name}' not found.",
                "hint": "Call list_metrics() to see available metrics.",
            }

        result = copy.deepcopy(metric)
        result["metric_name"] = metric_name
        result["is_stale"] = metric["stale_days"] > STALE_DAYS

        if result["is_stale"]:
            result["warning"] = (
                f"Data is {metric['stale_days']} days old "
                f"(threshold: {STALE_DAYS} days). "
                f"Call refresh_data('{metric_name}') before using this value."
            )

        self._log("query_metric", {"metric_name": metric_name}, result["value"])
        return result

    def refresh_data(self, metric_name: str) -> dict[str, Any]:
        """
        Trigger a data pipeline refresh for a metric.
        Some refreshes surface new values — agent must re-query after refreshing.
        """
        if self._random_failure():
            return {"error": "Refresh pipeline busy. Retry."}

        metric = self.metrics.get(metric_name)
        if not metric:
            return {"error": f"Metric '{metric_name}' not found."}

        was_stale = metric["stale_days"] > STALE_DAYS
        metric["stale_days"] = 0

        if metric_name in REFRESH_OVERRIDES:
            for k, v in REFRESH_OVERRIDES[metric_name].items():
                metric[k] = v

        self._log("refresh_data", {"metric_name": metric_name}, "refreshed")
        return {
            "success": True,
            "metric_name": metric_name,
            "was_stale": was_stale,
            "message": f"'{metric_name}' refreshed. Re-query to get updated value.",
        }

    def get_approver(self, role: str) -> dict[str, Any]:
        """Look up who approves for a given role. Surfaces OOO/delegate info."""
        if self._random_failure():
            return {"error": "DataHub read error. Retry."}

        directory = self.metrics.get("approver_directory", {}).get("value", {})
        role_upper = role.upper()
        approver = directory.get(role_upper)

        if not approver:
            return {
                "error": f"No approver for role '{role}'.",
                "hint": f"Available roles: {list(directory.keys())}",
            }

        result: dict[str, Any] = {"role": role_upper, "approver": approver}
        if role_upper == "CFO":
            result["warning"] = "CFO is OOO. Use CFO_DELEGATE instead."
            result["delegate"] = directory.get("CFO_DELEGATE")

        self._log("get_approver", {"role": role}, approver)
        return result

    # ------------------------------------------------------------------ helpers

    def _random_failure(self) -> bool:
        return random.random() < self.noise_prob

    def _log(self, method: str, args: dict, result_summary: Any) -> None:
        self.action_log.append(
            {"app": "DataHub", "method": method, "args": args, "result": result_summary}
        )

    def get_state_snapshot(self) -> dict:
        stale = sum(1 for m in self.metrics.values() if m["stale_days"] > STALE_DAYS)
        return {"total_metrics": len(self.metrics), "stale_metrics": stale}