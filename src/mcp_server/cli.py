"""Call any AtomisticSkills MCP tool from the shell, without an MCP client.

Skills name MCP tools as ``<server>.<tool>`` (for example ``mace.relax_structure``).
An agent with the server connected calls the tool directly. This module is the
fallback for every other case -- an agent that does not load MCP servers, a
server that failed to connect, a batch script -- and goes through the same
FastMCP machinery, so arguments are validated against the same typed schema.

Servers are stateful: ``load_model`` keeps a model in memory for later tools.
Calls listed in one invocation therefore run in sequence inside one process.

Usage:
    python -m src.mcp_server.cli <server> --list
    python -m src.mcp_server.cli <server> <tool> [key=value ...] [<tool> ...]
    python -m src.mcp_server.cli <server> <tool> '{"key": "value"}'

Each value is parsed as JSON where possible (``fmax=0.02`` is a number,
``relax_cell=false`` a boolean, ``fixed_atoms=[0,1]`` a list) and is otherwise
taken as a string. Run it in the server's environment, e.g.:

    venv/run mlip python -m src.mcp_server.cli mace \\
        load_model model_name=MACE-OMAT-0-small \\
        relax_structure structure_data=POSCAR fmax=0.02

Results are printed to stdout as JSON, one object per call; library output goes
to stderr. The exit status is 1 if a call fails or returns an error, and the
calls after it are skipped.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SERVERS = sorted(
    p.stem.removesuffix("_server") for p in Path(__file__).parent.glob("*_server.py")
)
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_value(text: str) -> Any:
    """Return text as JSON when it parses, otherwise as the string itself."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_calls(tokens: list[str], tool_names: set[str]) -> list[tuple[str, dict]]:
    """Split ``tool [args] tool [args] ...`` into (tool, arguments) pairs."""
    calls: list[tuple[str, dict]] = []
    for token in tokens:
        if IDENTIFIER.match(token) and token in tool_names:
            calls.append((token, {}))
            continue
        if not calls:
            raise SystemExit(
                f"expected a tool name, got {token!r}; tools: {', '.join(sorted(tool_names))}"
            )
        if token.lstrip().startswith("{"):
            try:
                calls[-1][1].update(json.loads(token))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"invalid JSON arguments for {calls[-1][0]}: {exc}"
                ) from exc
        elif "=" in token:
            key, value = token.split("=", 1)
            calls[-1][1][key] = parse_value(value)
        else:
            raise SystemExit(
                f"cannot parse {token!r}: not a tool of this server and not key=value "
                f"(tools: {', '.join(sorted(tool_names))})"
            )
    return calls


def to_jsonable(result: Any) -> Any:
    """Turn what FastMCP returns for a tool call into plain JSON data.

    Depending on the tool's return annotation, FastMCP hands back content blocks,
    a structured dict, or both as a tuple. The structured form is preferred, and
    text blocks are decoded when they carry JSON.
    """
    if isinstance(result, tuple) and len(result) == 2:
        content, structured = result
        if structured is not None:
            if isinstance(structured, dict) and set(structured) == {"result"}:
                return structured["result"]
            return structured
        result = content
    if isinstance(result, dict):
        return result
    values = []
    for block in result or []:
        text = getattr(block, "text", None)
        if text is None:
            values.append(
                block.model_dump(mode="json")
                if hasattr(block, "model_dump")
                else str(block)
            )
        else:
            values.append(parse_value(text))
    return values[0] if len(values) == 1 else values


def failed(value: Any) -> bool:
    """Return True for the error shapes the servers use instead of raising."""
    if isinstance(value, dict) and value.get("error"):
        return True
    return isinstance(value, str) and value.lstrip().lower().startswith("error")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.mcp_server.cli",
        description="Call AtomisticSkills MCP tools from the shell.",
        epilog="Example: python -m src.mcp_server.cli base create_research_dir research_topic=LiFePO4",
    )
    parser.add_argument("server", choices=SERVERS, help="MCP server name")
    parser.add_argument(
        "--list", action="store_true", help="list the server's tools and arguments"
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="run every call even if one fails"
    )
    parser.add_argument("calls", nargs="*", help="tool [key=value ...] ...")
    args = parser.parse_args(argv)

    # Importing a server redirects file descriptor 1 to stderr, so that library
    # chatter cannot corrupt the MCP stream, and keeps the real stdout as
    # mcp_pipe_binary. Results go there; everything else stays on stderr.
    module = importlib.import_module(f"src.mcp_server.{args.server}_server")
    out = getattr(module, "mcp_pipe_binary", None) or sys.stdout.buffer
    server = module.mcp

    import anyio

    tools = anyio.run(server.list_tools)
    by_name = {tool.name: tool for tool in tools}

    def emit(value: Any) -> None:
        out.write((json.dumps(value, indent=2, default=str) + "\n").encode())
        out.flush()

    if args.list or not args.calls:
        emit(
            [
                {
                    "tool": tool.name,
                    "description": (tool.description or "").strip().split("\n")[0],
                    "arguments": {
                        name: {
                            "type": spec.get("type", spec.get("anyOf", "any")),
                            **(
                                {"default": spec["default"]}
                                if "default" in spec
                                else {}
                            ),
                        }
                        for name, spec in tool.inputSchema.get("properties", {}).items()
                    },
                    "required": tool.inputSchema.get("required", []),
                }
                for tool in tools
            ]
        )
        return 0

    status = 0
    for name, arguments in parse_calls(args.calls, set(by_name)):
        try:
            value = to_jsonable(anyio.run(server.call_tool, name, arguments))
        except Exception as exc:  # the tool raised: report it like an MCP client would
            value = {"error": f"{type(exc).__name__}: {exc}"}
        emit({"tool": name, "result": value})
        if failed(value):
            status = 1
            if not args.keep_going:
                break
    return status


if __name__ == "__main__":
    sys.exit(main())
