"""Google Antigravity provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


google_antigravity = ProviderProfile(
    name="google-antigravity",
    aliases=("antigravity", "google_antigravity"),
    api_mode="chat_completions",
    display_name="Google Antigravity",
    description="Google Antigravity via OAuth PKCE — no API key required",
    env_vars=(),
    base_url="https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
    auth_type="oauth_external",
)

register_provider(google_antigravity)
