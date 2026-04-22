"""
random_baseline.py — Run a random agent against CompanyOS.

This produces the BEFORE curve that you compare against your trained agent.
Run this locally or in Colab against your HF Spaces deployment.

Usage (local):
    python training/random_baseline.py --episodes 100

Usage (against HF Spaces):
    python training/random_baseline.py --episodes 100 --env-url https://your-space.hf.space
"""

import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import requests
    REMOTE_MODE = True
except ImportError:
    REMOTE_MODE = False

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ── Action space the random agent samples from ────────────────────────────────

RANDOM_ACTIONS = [
    # TicketDesk
    {"app": "ticketdesk", "method": "list_tickets",  "params": {}},
    {"app": "ticketdesk", "method": "search_tickets", "params": {"query": "vendor"}},
    {"app": "ticketdesk", "method": "search_tickets", "params": {"query": "expense"}},
    {"app": "ticketdesk", "method": "search_tickets", "params": {"query": "bug"}},
    {"app": "ticketdesk", "method": "get_ticket",     "params": {"ticket_id": "T-001"}},
    {"app": "ticketdesk", "method": "get_ticket",     "params": {"ticket_id": "T-002"}},
    {"app": "ticketdesk", "method": "get_ticket",     "params": {"ticket_id": "T-003"}},
    {"app": "ticketdesk", "method": "update_ticket",  "params": {"ticket_id": "T-001", "field": "priority",  "value": "high"}},
    {"app": "ticketdesk", "method": "update_ticket",  "params": {"ticket_id": "T-001", "field": "verified",  "value": True}},
    {"app": "ticketdesk", "method": "update_ticket",  "params": {"ticket_id": "T-003", "field": "status",    "value": "open"}},
    {"app": "ticketdesk", "method": "update_ticket",  "params": {"ticket_id": "T-003", "field": "assignee",  "value": "mike.ross"}},
    # DataHub
    {"app": "datahub",    "method": "list_metrics",   "params": {}},
    {"app": "datahub",    "method": "query_metric",   "params": {"metric_name": "vendor_compliance_score"}},
    {"app": "datahub",    "method": "query_metric",   "params": {"metric_name": "expense_report_447_amount"}},
    {"app": "datahub",    "method": "query_metric",   "params": {"metric_name": "auth_service_error_rate"}},
    {"app": "datahub",    "method": "refresh_data",   "params": {"metric_name": "vendor_compliance_score"}},
    {"app": "datahub",    "method": "get_approver",   "params": {"role": "CFO"}},
    {"app": "datahub",    "method": "get_approver",   "params": {"role": "CFO_DELEGATE"}},
    {"app": "datahub",    "method": "get_approver",   "params": {"role": "CTO"}},
    # ApprovalFlow
    {"app": "approvalflow", "method": "list_approval_types", "params": {}},
    {"app": "approvalflow", "method": "submit_approval", "params": {
        "approval_type": "vendor_onboarding", "approver": "david.kim",
        "data": {"vendor_name": "ACME", "compliance_score": 85}}},
    {"app": "approvalflow", "method": "submit_approval", "params": {
        "approval_type": "expense_report", "approver": "david.kim",
        "data": {"amount": 4350, "employee_id": "john.doe", "report_id": "447"}}},
    {"app": "approvalflow", "method": "check_status", "params": {"approval_id": "APR-001"}},
    {"app": "approvalflow", "method": "list_approvals", "params": {}},
    # Intentionally bad actions (random agent doesn't know better)
    {"app": "ticketdesk",   "method": "update_ticket", "params": {"ticket_id": "T-001", "field": "linked_approval", "value": "APR-001"}},
    {"app": "ticketdesk",   "method": "update_ticket", "params": {"ticket_id": "T-003", "field": "status", "value": "closed"}},
    {"app": "approvalflow", "method": "submit_approval", "params": {
        "approval_type": "vendor_onboarding", "approver": "sarah.chen",  # OOO
        "data": {"vendor_name": "ACME", "compliance_score": 85}}},
]


def random_action() -> dict:
    return random.choice(RANDOM_ACTIONS)


# ── Local mode runner ─────────────────────────────────────────────────────────

def run_local(n_episodes: int, task_id: str | None = None):
    from env.company_env import CompanyOSEnv
    env = CompanyOSEnv(noise_prob=0.1)
    rewards, successes = [], []

    for ep in range(n_episodes):
        obs = env.reset(task_id=task_id)
        done, ep_reward = False, 0.0
        while not done:
            action = random_action()
            _, reward, done, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        successes.append(info.get("success", False))
        if (ep + 1) % 10 == 0:
            mean_r  = sum(rewards[-10:]) / 10
            sr      = sum(successes[-10:]) / 10
            print(f"  Ep {ep+1:>4} | avg_reward={mean_r:+.2f} | success_rate={sr:.0%}")

    return rewards, successes


# ── Remote mode runner ────────────────────────────────────────────────────────

def run_remote(n_episodes: int, env_url: str, task_id: str | None = None):
    rewards, successes = [], []

    for ep in range(n_episodes):
        body = {"task_id": task_id} if task_id else {}
        obs = requests.post(f"{env_url}/reset", json=body, timeout=30).json()["observation"]
        done, ep_reward = False, 0.0
        while not done:
            action = random_action()
            r = requests.post(f"{env_url}/step", json=action, timeout=30).json()
            ep_reward += r["reward"]
            done = r["done"]
            info = r["info"]
        rewards.append(ep_reward)
        successes.append(info.get("success", False))
        if (ep + 1) % 10 == 0:
            mean_r = sum(rewards[-10:]) / 10
            sr     = sum(successes[-10:]) / 10
            print(f"  Ep {ep+1:>4} | avg_reward={mean_r:+.2f} | success_rate={sr:.0%}")

    return rewards, successes


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_baseline(rewards: list, successes: list, output_path: str = "baseline_curve.png"):
    if not HAS_PLOT:
        print("matplotlib not available — skipping plot.")
        return

    window = min(10, len(rewards))
    smoothed_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smoothed_s = np.convolve(
        [1.0 if s else 0.0 for s in successes],
        np.ones(window) / window, mode="valid"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CompanyOS — Random Baseline", fontsize=13)

    ax1.plot(rewards, alpha=0.25, color="steelblue", label="Raw")
    ax1.plot(range(window - 1, len(rewards)), smoothed_r,
             color="steelblue", linewidth=2, label=f"Smoothed (w={window})")
    ax1.axhline(y=sum(rewards) / len(rewards), color="red",
                linestyle="--", linewidth=1, label=f"Mean={sum(rewards)/len(rewards):.1f}")
    ax1.set_title("Episode Reward — Random Agent")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(window - 1, len(successes)), smoothed_s,
             color="tomato", linewidth=2)
    ax2.set_title(f"Success Rate — Random Agent\n(overall: {sum(successes)/len(successes):.1%})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved baseline curve → {output_path}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random baseline on CompanyOS")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--env-url",  type=str, default=None,
                        help="HF Spaces URL — if omitted, runs locally")
    parser.add_argument("--task-id",  type=str, default=None,
                        help="Pin a specific task ID for reproducibility")
    parser.add_argument("--output",   type=str, default="baseline_curve.png")
    args = parser.parse_args()

    print(f"Running random baseline — {args.episodes} episodes "
          f"({'remote: ' + args.env_url if args.env_url else 'local'})\n")

    if args.env_url:
        rewards, successes = run_remote(args.episodes, args.env_url, args.task_id)
    else:
        rewards, successes = run_local(args.episodes, args.task_id)

    print(f"\n── Results ──────────────────────────────")
    print(f"  Mean reward:  {sum(rewards)/len(rewards):+.2f}")
    print(f"  Best episode: {max(rewards):+.2f}")
    print(f"  Success rate: {sum(successes)/len(successes):.1%}")
    print(f"─────────────────────────────────────────")

    plot_baseline(rewards, successes, args.output)