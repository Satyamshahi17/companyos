"""
spaces/app.py — Gradio demo UI for CompanyOS on HuggingFace Spaces.

This runs alongside the FastAPI server and gives judges a visual,
interactive way to watch an agent (or themselves) complete tasks.

The FastAPI env server runs on port 7860.
Gradio runs on port 7861 in local dev; on Spaces it's the main UI.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import gradio as gr
except ImportError:
    raise SystemExit("Install gradio: pip install gradio")

from env.company_env import CompanyOSEnv

# One shared env instance for the demo
env = CompanyOSEnv(noise_prob=0.1)
_state = {"obs": None, "done": False, "log": [], "total_reward": 0.0}


def _fmt_obs(obs: dict) -> str:
    if not obs:
        return "No active episode. Click 'New Episode' to start."
    lines = [
        f"📋 TASK: {obs['task']}",
        f"",
        f"Step {obs['step']} / {obs['max_steps']}  |  Steps remaining: {obs['steps_remaining']}",
        f"",
        "Progress:",
    ]
    for k, v in obs["progress"].items():
        icon = "✅" if v else "⬜"
        lines.append(f"  {icon} {k}")
    if obs.get("last_result"):
        lines.append(f"")
        lines.append(f"Last result: {json.dumps(obs['last_result'], indent=2)}")
    if obs.get("message"):
        lines.append(f"")
        lines.append(f"ℹ️  {obs['message']}")
    return "\n".join(lines)


def _fmt_log() -> str:
    if not _state["log"]:
        return "No actions yet."
    return "\n".join(_state["log"][-20:])  # last 20 entries


def reset_episode(task_id: str):
    task_map = {
        "Vendor Onboarding":  "task_vendor_onboarding",
        "Expense Report":     "task_expense_report",
        "Bug Escalation":     "task_bug_escalation",
        "License Renewal":    "task_license_renewal",
        "Handbook Update":    "task_handbook_update",
        "Random":             None,
    }
    tid = task_map.get(task_id)
    obs = env.reset(task_id=tid)
    _state["obs"]          = obs
    _state["done"]         = False
    _state["log"]          = [f"── New episode: {task_id} ──"]
    _state["total_reward"] = 0.0
    return _fmt_obs(obs), _fmt_log(), f"Total reward: 0.00", "🟢 Running"


def take_action(app: str, method: str, params_json: str):
    if _state["done"]:
        return _fmt_obs(_state["obs"]), _fmt_log(), \
               f"Total reward: {_state['total_reward']:.2f}", "🔴 Done — start new episode"
    if _state["obs"] is None:
        return "Start an episode first.", _fmt_log(), "Total reward: 0.00", "⚪ Idle"

    # Parse params
    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        _state["log"].append(f"❌ Bad JSON params: {e}")
        return _fmt_obs(_state["obs"]), _fmt_log(), \
               f"Total reward: {_state['total_reward']:.2f}", "🟢 Running"

    action = {"app": app.strip(), "method": method.strip(), "params": params}
    obs, reward, done, info = env.step(action)

    _state["obs"]           = obs
    _state["done"]          = done
    _state["total_reward"] += reward

    status_icon = "✅" if info.get("success") else ("⏱️" if info.get("timeout") else "")
    log_line = (
        f"Step {info['step']:>2} | {app}.{method}({params_json[:40]}) "
        f"→ r={reward:+.2f}  {status_icon}"
    )
    _state["log"].append(log_line)

    if done:
        outcome = "✅ SUCCESS!" if info.get("success") else "⏱️ TIMEOUT"
        _state["log"].append(f"── Episode ended: {outcome}  Total reward: {_state['total_reward']:.2f} ──")
        status_label = f"🔴 {outcome}"
    else:
        status_label = "🟢 Running"

    return (
        _fmt_obs(obs),
        _fmt_log(),
        f"Total reward: {_state['total_reward']:.2f}",
        status_label,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

TOOL_HINT = """
Available tools:
  ticketdesk   → list_tickets | search_tickets | get_ticket | update_ticket
  datahub      → list_metrics | query_metric | refresh_data | get_approver
  approvalflow → list_approval_types | submit_approval | check_status | escalate | list_approvals
"""

EXAMPLE_ACTIONS = [
    ["ticketdesk",    "list_tickets",    "{}"],
    ["ticketdesk",    "search_tickets",  '{"query": "vendor"}'],
    ["ticketdesk",    "get_ticket",      '{"ticket_id": "T-001"}'],
    ["ticketdesk",    "update_ticket",   '{"ticket_id": "T-001", "field": "priority", "value": "high"}'],
    ["ticketdesk",    "update_ticket",   '{"ticket_id": "T-001", "field": "verified", "value": true}'],
    ["datahub",       "list_metrics",    "{}"],
    ["datahub",       "query_metric",    '{"metric_name": "vendor_compliance_score"}'],
    ["datahub",       "refresh_data",    '{"metric_name": "vendor_compliance_score"}'],
    ["datahub",       "get_approver",    '{"role": "CFO"}'],
    ["datahub",       "get_approver",    '{"role": "CFO_DELEGATE"}'],
    ["approvalflow",  "list_approval_types", "{}"],
    ["approvalflow",  "submit_approval", '{"approval_type": "vendor_onboarding", "approver": "david.kim", "data": {"vendor_name": "ACME", "compliance_score": 85}}'],
    ["approvalflow",  "check_status",    '{"approval_id": "APR-001"}'],
]

with gr.Blocks(title="CompanyOS — Enterprise RL Environment") as demo:
    gr.Markdown("# 🏢 CompanyOS\n### Enterprise Workflow RL Environment")
    gr.Markdown(
        "An OpenEnv-compliant RL benchmark where agents learn to navigate "
        "partially observable enterprise chaos across **TicketDesk**, **DataHub**, and **ApprovalFlow**."
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Episode Control")
            task_dropdown = gr.Dropdown(
                choices=["Random", "Vendor Onboarding", "Expense Report",
                         "Bug Escalation", "License Renewal", "Handbook Update"],
                value="Vendor Onboarding",
                label="Task",
            )
            reset_btn = gr.Button("🔄 New Episode", variant="primary")
            status_box = gr.Textbox(label="Status", value="⚪ Idle", interactive=False)
            reward_box = gr.Textbox(label="Total Reward", value="Total reward: 0.00", interactive=False)

        with gr.Column(scale=5):
            obs_box = gr.Textbox(
                label="Observation",
                value="Start a new episode to begin.",
                lines=18,
                interactive=False,
            )

    gr.Markdown("### Take an Action")
    gr.Markdown(f"```{TOOL_HINT}```")

    with gr.Row():
        app_input    = gr.Textbox(label="App",    placeholder="ticketdesk",     scale=1)
        method_input = gr.Textbox(label="Method", placeholder="list_tickets",   scale=1)
        params_input = gr.Textbox(label="Params (JSON)", placeholder='{}',       scale=3)
    step_btn = gr.Button("▶ Take Action", variant="secondary")

    gr.Examples(
        examples=EXAMPLE_ACTIONS,
        inputs=[app_input, method_input, params_input],
        label="Quick action examples",
    )

    log_box = gr.Textbox(label="Action Log", lines=10, interactive=False)

    reset_btn.click(
        fn=reset_episode,
        inputs=[task_dropdown],
        outputs=[obs_box, log_box, reward_box, status_box],
    )
    step_btn.click(
        fn=take_action,
        inputs=[app_input, method_input, params_input],
        outputs=[obs_box, log_box, reward_box, status_box],
    )

    gr.Markdown(
        "---\n"
        "**API:** Also available as REST at `/reset`, `/step`, `/render`, `/health`  \n"
        "**Training:** See `training/train.ipynb` for the Unsloth/TRL GRPO training script  \n"
        "**Repo:** [GitHub](https://github.com/your-username/companyos)"
    )


if __name__ == "__main__":
    demo.launch(server_port=7861, share=False)