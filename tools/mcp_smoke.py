#!/usr/bin/env python3
"""Start MCP servers the way a client does and check that they answer.

Each server is launched with the same command the plugin uses
(``venv/run --server <name>``), so this exercises the launcher, the
environment and the server together: the MCP handshake, ``tools/list``, and
optionally a sequence of tool calls in one session (servers are stateful, so a
``load_model`` followed by ``relax_structure`` works as it does for an agent).

Usage:
    venv/run cpu python tools/mcp_smoke.py base smol drugdisc atomate2
    venv/run cpu python tools/mcp_smoke.py mace \\
        --call load_model '{}' --call relax_structure '{"structure_data": "Si.cif"}'
    venv/run cpu python tools/mcp_smoke.py base --server-env ATOMISTIC_RUNTIME=docker

To test a container server from a native client, set the runtime with
``--server-env`` as above. Exporting ``ATOMISTIC_RUNTIME=docker`` for the whole
command puts this client inside the CPU image too, which cannot launch Docker
servers. If a container runtime is already configured globally, prefix the
client command with ``ATOMISTIC_RUNTIME=uv``.

Exit status is non-zero if any server fails to start, lists no tools, or a
call errors.

Requirements:
    - the ``mcp`` package (present in every uv project)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

LAUNCHER = Path(__file__).resolve().parents[1] / "venv" / "run"


def summarise(result) -> str:
    """Return a short text form of a CallToolResult."""
    texts = [getattr(block, "text", "") for block in result.content]
    text = " ".join(t for t in texts if t)
    return text if len(text) <= 300 else text[:300] + "..."


def returned_error(result) -> bool:
    """True when a tool reports failure in its payload instead of raising."""
    for block in result.content:
        text = getattr(block, "text", "") or ""
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        if isinstance(value, dict) and value.get("error"):
            return True
        if isinstance(value, str) and value.lstrip().lower().startswith("error"):
            return True
    return False


async def check(
    server: str,
    calls: list[tuple[str, dict]],
    timeout: float,
    server_env: dict[str, str],
) -> bool:
    params = StdioServerParameters(
        command=str(LAUNCHER),
        args=["--server", server],
        env={**os.environ, **server_env},
    )
    started = time.monotonic()
    with anyio.fail_after(timeout):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                print(
                    f"[{server}] up in {time.monotonic() - started:.1f}s, "
                    f"{len(tools)} tools: {', '.join(t.name for t in tools)}"
                )
                if not tools:
                    return False
                ok = True
                for name, arguments in calls:
                    t0 = time.monotonic()
                    result = await session.call_tool(name, arguments)
                    summary = summarise(result)
                    failed = result.isError or returned_error(result)
                    ok &= not failed
                    print(
                        f"[{server}] {name} -> {'ERROR' if failed else 'ok'} "
                        f"({time.monotonic() - t0:.1f}s): {summary}"
                    )
                return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("servers", nargs="+", help="server names from venv/servers.tsv")
    parser.add_argument(
        "--call",
        nargs=2,
        action="append",
        default=[],
        metavar=("TOOL", "JSON"),
        help="tool call to make after listing tools (repeatable; applies to every server)",
    )
    # A first start may build the environment, which venv/run allows
    # ATOMISTIC_SERVER_SETUP_WAIT seconds for: wait at least that long.
    setup_wait = float(os.environ.get("ATOMISTIC_SERVER_SETUP_WAIT", 0) or 0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=max(600.0, setup_wait + 120),
        help="seconds per server (default: 600, or ATOMISTIC_SERVER_SETUP_WAIT + 120)",
    )
    parser.add_argument(
        "--server-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment for the servers only, e.g. ATOMISTIC_RUNTIME=docker (repeatable)",
    )
    args = parser.parse_args()
    calls = [(tool, json.loads(arguments)) for tool, arguments in args.call]
    server_env = dict(item.split("=", 1) for item in args.server_env)

    failures = []
    for server in args.servers:
        try:
            ok = anyio.run(check, server, calls, args.timeout, server_env)
        except (
            Exception
        ) as exc:  # a server that dies or times out is a result, not a crash
            print(f"[{server}] FAILED: {type(exc).__name__}: {exc}")
            ok = False
        if not ok:
            failures.append(server)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"all {len(args.servers)} servers answered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
