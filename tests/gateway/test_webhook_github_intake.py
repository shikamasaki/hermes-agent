"""GitHub App intake is reachable through the always-on WebhookAdapter."""

import asyncio
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from hermes_cli.github_sync import GitHubIntakeError, IntakeResult


def _adapter() -> WebhookAdapter:
    return WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "routes": {}},
        )
    )


def _app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/github/events", adapter._handle_github_event)
    return app


def test_public_webhook_lane_accepts_authenticated_intake():
    async def scenario():
        adapter = _adapter()
        with patch(
            "hermes_cli.github_sync.process_configured_delivery",
            return_value=IntakeResult("created", "t_123"),
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post(
                    "/api/github/events",
                    data=b"{}",
                    headers={"X-GitHub-Delivery": "delivery-1"},
                )
                assert response.status == 202
                assert await response.json() == {
                    "status": "created",
                    "task_id": "t_123",
                }

    asyncio.run(scenario())


def test_public_webhook_lane_hides_authentication_details():
    async def scenario():
        adapter = _adapter()
        with patch(
            "hermes_cli.github_sync.process_configured_delivery",
            side_effect=GitHubIntakeError("repository route does not exist"),
        ):
            async with TestClient(TestServer(_app(adapter))) as client:
                response = await client.post("/api/github/events", data=b"{}")
                assert response.status == 401
                assert await response.json() == {
                    "error": "invalid GitHub delivery"
                }

    asyncio.run(scenario())
