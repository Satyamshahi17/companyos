"""
CompanyOS test suite.
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from apps import TicketDesk, DataHub, ApprovalFlow
from env.company_env import CompanyOSEnv
from env.task_generator import TaskGenerator


# ─────────────────────────────────────────────────────────────────────────────
# TicketDesk
# ─────────────────────────────────────────────────────────────────────────────

class TestTicketDesk:
    def setup_method(self):
        self.td = TicketDesk(noise_prob=0)
        self.td.reset(seed=42)

    def test_list_tickets_returns_all(self):
        result = self.td.list_tickets()
        assert len(result["tickets"]) == 5

    def test_search_finds_ticket(self):
        result = self.td.search_tickets("vendor")
        assert any(t["id"] == "T-001" for t in result["results"])

    def test_search_miss_returns_hint(self):
        result = self.td.search_tickets("xyznonexistent")
        assert result["results"] == []
        assert "hint" in result

    def test_get_ticket_full_details(self):
        result = self.td.get_ticket("T-001")
        assert "ticket" in result
        assert result["ticket"]["title"] == "Onboard vendor ACME Corp"

    def test_get_ticket_missing_priority(self):
        result = self.td.get_ticket("T-001")
        assert result["ticket"]["priority"] is None   # intentionally missing

    def test_get_ticket_not_found(self):
        result = self.td.get_ticket("T-999")
        assert "error" in result

    def test_update_priority(self):
        result = self.td.update_ticket("T-001", "priority", "high")
        assert result["success"] is True
        ticket = self.td.get_ticket("T-001")["ticket"]
        assert ticket["priority"] == "high"

    def test_update_verified(self):
        result = self.td.update_ticket("T-001", "verified", True)
        assert result["success"] is True

    def test_cannot_link_approval_without_verified(self):
        result = self.td.update_ticket("T-001", "linked_approval", "APR-001")
        assert "error" in result
        assert "verified" in result["error"].lower()

    def test_can_link_approval_after_verified(self):
        self.td.update_ticket("T-001", "verified", True)
        result = self.td.update_ticket("T-001", "linked_approval", "APR-001")
        assert result["success"] is True

    def test_cannot_close_blocked_ticket(self):
        result = self.td.update_ticket("T-003", "status", "closed")
        assert "error" in result

    def test_can_unblock_ticket(self):
        result = self.td.update_ticket("T-003", "status", "open")
        assert result["success"] is True

    def test_invalid_field_rejected(self):
        result = self.td.update_ticket("T-001", "secret_field", "hack")
        assert "error" in result

    def test_state_snapshot(self):
        snap = self.td.get_state_snapshot()
        assert "open" in snap and "blocked" in snap and "verified" in snap


# ─────────────────────────────────────────────────────────────────────────────
# DataHub
# ─────────────────────────────────────────────────────────────────────────────

class TestDataHub:
    def setup_method(self):
        self.dh = DataHub(noise_prob=0)
        self.dh.reset(seed=42)

    def test_list_metrics(self):
        result = self.dh.list_metrics()
        assert "vendor_compliance_score" in result["available_metrics"]

    def test_query_stale_metric_warns(self):
        result = self.dh.query_metric("vendor_compliance_score")
        assert result["is_stale"] is True
        assert "warning" in result

    def test_query_fresh_metric_no_warning(self):
        result = self.dh.query_metric("expense_report_447_amount")
        assert result["is_stale"] is False
        assert "warning" not in result

    def test_refresh_clears_stale(self):
        self.dh.refresh_data("vendor_compliance_score")
        result = self.dh.query_metric("vendor_compliance_score")
        assert result["is_stale"] is False

    def test_refresh_updates_value(self):
        before = self.dh.query_metric("vendor_compliance_score")["value"]
        self.dh.refresh_data("vendor_compliance_score")
        after = self.dh.query_metric("vendor_compliance_score")["value"]
        assert after != before   # refresh surfaced new data

    def test_query_missing_metric(self):
        result = self.dh.query_metric("nonexistent_metric")
        assert "error" in result

    def test_get_approver_cfo_warns_ooo(self):
        result = self.dh.get_approver("CFO")
        assert "warning" in result
        assert "delegate" in result

    def test_get_approver_cfo_delegate(self):
        result = self.dh.get_approver("CFO_DELEGATE")
        assert result["approver"] == "david.kim"

    def test_get_approver_invalid_role(self):
        result = self.dh.get_approver("JANITOR")
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# ApprovalFlow
# ─────────────────────────────────────────────────────────────────────────────

class TestApprovalFlow:
    def setup_method(self):
        self.af = ApprovalFlow(noise_prob=0)
        self.af.reset(seed=42)

    def _submit_valid(self):
        return self.af.submit_approval(
            approval_type="vendor_onboarding",
            approver="david.kim",
            data={"vendor_name": "ACME", "compliance_score": 85},
        )

    def test_list_approval_types(self):
        result = self.af.list_approval_types()
        assert "vendor_onboarding" in result["approval_types"]
        assert "expense_report" in result["approval_types"]

    def test_submit_valid_approval(self):
        result = self._submit_valid()
        assert result["success"] is True
        assert "approval_id" in result

    def test_submit_unknown_type(self):
        result = self.af.submit_approval("fake_type", "david.kim", {})
        assert "error" in result

    def test_submit_missing_data_keys(self):
        result = self.af.submit_approval(
            "vendor_onboarding", "david.kim", {"vendor_name": "ACME"}
            # missing compliance_score
        )
        assert "error" in result
        assert "missing" in result["error"].lower()

    def test_cfo_ooo_blocked(self):
        result = self.af.submit_approval(
            "vendor_onboarding",
            "sarah.chen",    # CFO — OOO
            {"vendor_name": "ACME", "compliance_score": 85},
        )
        assert "error" in result
        assert "OOO" in result["error"]

    def test_check_status_pending_then_resolves(self):
        sub = self._submit_valid()
        apr_id = sub["approval_id"]
        # Poll until resolved (max 5 polls, steps_until_decision ≤ 3)
        for _ in range(5):
            status_result = self.af.check_status(apr_id)
            if status_result["status"] != "pending":
                break
        assert status_result["status"] in ("approved", "rejected")

    def test_escalate_resets_to_pending(self):
        sub = self._submit_valid()
        apr_id = sub["approval_id"]
        # Force to rejected by draining steps
        for _ in range(5):
            r = self.af.check_status(apr_id)
            if r["status"] == "rejected":
                break
        if r["status"] == "rejected":
            esc = self.af.escalate(apr_id, "Urgent — vendor deadline today")
            assert esc["success"] is True
            assert self.af.approvals[apr_id]["status"] == "pending"

    def test_cannot_escalate_approved(self):
        sub = self._submit_valid()
        apr_id = sub["approval_id"]
        # Force approval by setting status directly (white-box for test)
        self.af.approvals[apr_id]["status"] = "approved"
        result = self.af.escalate(apr_id, "trying anyway")
        assert "error" in result

    def test_list_approvals(self):
        self._submit_valid()
        result = self.af.list_approvals()
        assert len(result["approvals"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# CompanyOSEnv — full environment
# ─────────────────────────────────────────────────────────────────────────────

class TestCompanyOSEnv:
    def setup_method(self):
        self.env = CompanyOSEnv(noise_prob=0, seed=42)

    def _step(self, app, method, params=None):
        return self.env.step({"app": app, "method": method, "params": params or {}})

    def test_reset_returns_observation(self):
        obs = self.env.reset(task_id="task_vendor_onboarding")
        assert "task" in obs
        assert "progress" in obs
        assert "tools" in obs
        assert obs["step"] == 0

    def test_reset_initialises_progress_false(self):
        obs = self.env.reset(task_id="task_vendor_onboarding")
        assert all(v is False for v in obs["progress"].values())

    def test_step_returns_correct_shape(self):
        self.env.reset(task_id="task_vendor_onboarding")
        obs, reward, done, info = self._step("ticketdesk", "list_tickets")
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "progress" in info

    def test_invalid_action_penalised(self):
        self.env.reset(task_id="task_vendor_onboarding")
        _, reward, _, _ = self._step("fakapp", "fakemethod")
        assert reward < 0

    def test_step_increments_counter(self):
        self.env.reset(task_id="task_vendor_onboarding")
        self._step("ticketdesk", "list_tickets")
        self._step("ticketdesk", "list_tickets")
        assert self.env.step_count == 2

    def test_timeout_ends_episode(self):
        self.env.reset(task_id="task_vendor_onboarding")
        self.env.task.max_steps = 3
        for _ in range(3):
            _, _, done, info = self._step("ticketdesk", "list_tickets")
        assert done is True
        assert info["timeout"] is True

    def test_full_vendor_onboarding_episode_succeeds(self):
        """End-to-end: complete vendor onboarding correctly."""
        self.env.reset(task_id="task_vendor_onboarding")
        setup_steps = [
            ("ticketdesk",    "update_ticket",   {"ticket_id": "T-001", "field": "priority", "value": "high"}),
            ("ticketdesk",    "update_ticket",   {"ticket_id": "T-001", "field": "verified",  "value": True}),
            ("datahub",       "refresh_data",    {"metric_name": "vendor_compliance_score"}),
            ("datahub",       "query_metric",    {"metric_name": "vendor_compliance_score"}),
            ("approvalflow",  "submit_approval", {
                "approval_type": "vendor_onboarding",
                "approver": "david.kim",
                "data": {"vendor_name": "ACME", "compliance_score": 85},
            }),
        ]
        done, info = False, {}
        for app, method, params in setup_steps:
            _, _, done, info = self._step(app, method, params)
            if done:
                break
        # Poll approval until resolved (max 5 polls handles random steps_until_decision)
        for _ in range(5):
            if done:
                break
            _, _, done, info = self._step("approvalflow", "check_status", {"approval_id": "APR-001"})
        assert info["success"] is True

    def test_render_returns_state(self):
        self.env.reset(task_id="task_vendor_onboarding")
        snap = self.env.render()
        assert "task" in snap
        assert "ticketdesk_state" in snap
        assert "datahub_state" in snap
        assert "approvalflow_state" in snap

    def test_all_five_tasks_are_resettable(self):
        tg = TaskGenerator(seed=0)
        for tid in tg.all_task_ids:
            obs = self.env.reset(task_id=tid)
            assert obs["step"] == 0
            assert obs["task"]

    def test_reward_positive_on_good_action(self):
        self.env.reset(task_id="task_vendor_onboarding")
        _, reward, _, _ = self._step(
            "ticketdesk", "update_ticket",
            {"ticket_id": "T-001", "field": "priority", "value": "high"}
        )
        assert reward > 0

    def test_terminal_bonus_on_success(self):
        """Completing the task should give a large positive reward on the final step."""
        self.env.reset(task_id="task_vendor_onboarding")
        setup_steps = [
            ("ticketdesk",   "update_ticket",   {"ticket_id": "T-001", "field": "priority", "value": "high"}),
            ("ticketdesk",   "update_ticket",   {"ticket_id": "T-001", "field": "verified",  "value": True}),
            ("datahub",      "refresh_data",    {"metric_name": "vendor_compliance_score"}),
            ("datahub",      "query_metric",    {"metric_name": "vendor_compliance_score"}),
            ("approvalflow", "submit_approval", {
                "approval_type": "vendor_onboarding",
                "approver": "david.kim",
                "data": {"vendor_name": "ACME", "compliance_score": 85},
            }),
        ]
        last_reward, done = 0.0, False
        for app, method, params in setup_steps:
            _, last_reward, done, _ = self._step(app, method, params)
        for _ in range(5):
            if done:
                break
            _, last_reward, done, _ = self._step("approvalflow", "check_status", {"approval_id": "APR-001"})
        assert last_reward > 10.0