"""``hermes usage`` subcommand parser.

Top-level provider usage listing. Distinct from the ``-z --usage-file``
oneshot flag (a per-run token/cost report) and the in-chat ``/usage``
slash command (both unrelated surfaces) — no namespace collision, since
argparse subparsers scope flags independently.
"""

from __future__ import annotations

from typing import Callable


def build_usage_parser(subparsers, *, cmd_usage: Callable) -> None:
    """Attach the ``usage`` subcommand to ``subparsers``."""
    usage_parser = subparsers.add_parser(
        "usage",
        help="Show cached usage status for every configured provider",
        description=(
            "Read-only usage listing across every provider discovered from "
            "active config (main model, delegation routes, auxiliary task "
            "assignments). Renders whatever "
            "is safely cached; never makes a live provider call unless "
            "--refresh is passed."
        ),
    )
    usage_parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Force a bounded, synchronous refresh for the selected "
            "providers before rendering (makes a live provider call)"
        ),
    )
    usage_parser.add_argument(
        "--provider",
        action="append",
        default=None,
        metavar="PROVIDER",
        help="Limit output to this provider; repeatable",
    )
    usage_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text",
    )
    usage_parser.set_defaults(func=cmd_usage)
