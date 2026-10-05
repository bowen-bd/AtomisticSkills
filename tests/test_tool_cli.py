"""Tests for src/mcp_server/cli.py, the shell fallback for every MCP tool.

Skills write MCP tools as ``server.tool``. When the server is not connected,
the agent runs the same tool through this CLI, so it must parse arguments the
way the note in each SKILL.md shows, keep server state across calls in one
command, and report errors the way the servers do.

Requirements:
    - Environment: cpu (run with: venv/run cpu python -m pytest tests/test_tool_cli.py)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.mcp_server import cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS = {"load_model", "relax_structure", "create_research_dir"}


class TestParseCalls:
    def test_key_value_pairs_are_json_where_possible(self):
        calls = cli.parse_calls(
            [
                "relax_structure",
                "structure_data=POSCAR",
                "fmax=0.02",
                "relax_cell=false",
                "fixed_atoms=[0,1]",
            ],
            TOOLS,
        )
        assert calls == [
            (
                "relax_structure",
                {
                    "structure_data": "POSCAR",
                    "fmax": 0.02,
                    "relax_cell": False,
                    "fixed_atoms": [0, 1],
                },
            )
        ]

    def test_several_tools_in_one_command(self):
        calls = cli.parse_calls(
            [
                "load_model",
                "model_name=MACE-OMAT-0-small",
                "relax_structure",
                "fmax=0.05",
            ],
            TOOLS,
        )
        assert [c[0] for c in calls] == ["load_model", "relax_structure"]
        assert calls[0][1] == {"model_name": "MACE-OMAT-0-small"}

    def test_json_object_arguments(self):
        calls = cli.parse_calls(
            ["relax_structure", '{"structure_data": "a.cif", "steps": 10}'], TOOLS
        )
        assert calls == [("relax_structure", {"structure_data": "a.cif", "steps": 10})]

    def test_value_may_contain_equals_signs(self):
        calls = cli.parse_calls(["create_research_dir", "research_topic=a=b"], TOOLS)
        assert calls[0][1] == {"research_topic": "a=b"}

    def test_arguments_before_any_tool_are_rejected(self):
        with pytest.raises(SystemExit, match="expected a tool name"):
            cli.parse_calls(["fmax=0.1"], TOOLS)

    def test_unknown_words_are_rejected_with_the_tool_list(self):
        with pytest.raises(SystemExit, match="not a tool of this server"):
            cli.parse_calls(["load_model", "relax"], TOOLS)


class TestResults:
    def test_structured_result_is_unwrapped(self):
        assert cli.to_jsonable(([], {"result": {"energy": -1.0}})) == {"energy": -1.0}

    def test_text_content_is_decoded(self):
        class Block:
            text = '{"a": 1}'

        assert cli.to_jsonable([Block()]) == {"a": 1}

    @pytest.mark.parametrize(
        "value, expected",
        [
            ({"error": "Model not loaded"}, True),
            ("Error loading model: x", True),
            ({"energy": 1.0}, False),
            ("Successfully loaded", False),
            ({"error": None}, False),
        ],
    )
    def test_error_shapes(self, value, expected):
        assert cli.failed(value) is expected


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, ATOMISTIC_WORKSPACE=str(cwd))
    return subprocess.run(
        [sys.executable, "-m", "src.mcp_server.cli", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestAgainstARealServer:
    def test_lists_tools_with_their_arguments(self, tmp_path):
        result = run_cli("base", "--list", cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        tools = {t["tool"]: t for t in json.loads(result.stdout)}
        assert "create_research_dir" in tools
        assert tools["create_research_dir"]["required"] == ["research_topic"]

    def test_runs_a_tool_and_writes_into_the_workspace(self, tmp_path):
        result = run_cli(
            "base", "create_research_dir", "research_topic=cli_unit", cwd=tmp_path
        )
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["tool"] == "create_research_dir"
        assert str(tmp_path) in out["result"]
        assert any(
            p.name.endswith("_cli_unit") for p in (tmp_path / "research").iterdir()
        )

    def test_library_chatter_stays_off_stdout(self, tmp_path):
        """stdout carries only JSON, so the output can be piped into jq."""
        result = run_cli(
            "base", "create_research_dir", "research_topic=quiet", cwd=tmp_path
        )
        json.loads(result.stdout)

    def test_invalid_arguments_fail_like_an_mcp_call(self, tmp_path):
        result = run_cli(
            "base", "create_research_dir", "no_such_argument=1", cwd=tmp_path
        )
        assert result.returncode == 1
        assert "error" in json.loads(result.stdout)["result"]
