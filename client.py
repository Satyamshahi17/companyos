"""
CompanyOS — OpenEnv client for remote interaction.

Install the package and point at a running CompanyOS Space:

    from companyos import CompanyAction, CompanyOSEnv

    with CompanyOSEnv(base_url="https://your-space.hf.space").sync() as client:
        result = client.reset()
        result = client.step(CompanyAction(app="ticketdesk", command="list"))
"""

from __future__ import annotations

from openenv.core.env_client import EnvClient

from companyos.models import CompanyAction, CompanyObservation, CompanyState


class CompanyOSEnv(EnvClient[CompanyAction, CompanyObservation, CompanyState]):
    """
    Async-first client for the CompanyOS environment.

    Connects over WebSocket to a running CompanyOS server
    (local Docker or remote HF Space).

    Usage (async)::

        async with CompanyOSEnv(base_url=url) as client:
            obs = await client.reset()
            obs = await client.step(CompanyAction(...))

    Usage (sync)::

        with CompanyOSEnv(base_url=url).sync() as client:
            obs = client.reset()
            obs = client.step(CompanyAction(...))
    """

    OBSERVATION_CLASS = CompanyObservation
    STATE_CLASS = CompanyState
