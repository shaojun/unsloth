# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.battleground.agent_harness import (
    _instructions,
    validate_and_sanitize_harness,
)
from storage import battleground_db


def _harness(**changes):
    value = {
        "enabled": True,
        "version": 1,
        "instructions": "Be accurate.",
        "skills": [],
        "web_search": {
            "enabled": True,
            "provider": "brave",
            "api_key": "brave-secret",
            "max_results": 5,
        },
        "mcp_config": {"mcpServers": {}},
        "max_turns": 8,
        "tool_timeout_seconds": 20,
        "run_timeout_seconds": 120,
    }
    value.update(changes)
    return value


def test_harness_snapshot_redacts_all_credentials():
    config = _harness(
        mcp_config = {
            "mcpServers": {
                "docs": {
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer mcp-secret"},
                    "allowedTools": ["search_docs"],
                }
            }
        }
    )
    snapshot, secrets = validate_and_sanitize_harness(config)

    assert snapshot["web_search"]["api_key_set"] is True
    assert "api_key" not in snapshot["web_search"]
    assert snapshot["mcp_servers"] == [
        {
            "name": "docs",
            "url": "https://example.com/mcp",
            "transport": "streamable_http",
            "enabled": True,
            "allowed_tools": ["search_docs"],
            "headers_set": True,
        }
    ]
    assert "brave-secret" not in str(snapshot)
    assert "mcp-secret" not in str(snapshot)
    assert secrets["web_search"] == "brave-secret"
    assert "mcp-secret" in secrets["mcp_config"]


def test_harness_rejects_stdio_and_missing_remote_allowlist():
    with pytest.raises(ValueError, match = "stdio"):
        validate_and_sanitize_harness(
            _harness(
                mcp_config = {
                    "mcpServers": {
                        "local": {"command": "npx", "args": ["server"], "allowedTools": ["read"]}
                    }
                }
            )
        )

    with pytest.raises(ValueError, match = "allowedTools"):
        validate_and_sanitize_harness(
            _harness(
                mcp_config = {
                    "mcpServers": {"remote": {"url": "https://example.com/mcp"}}
                }
            )
        )


def test_ddgs_is_keyless_and_skills_compile_into_shared_instructions():
    snapshot, secrets = validate_and_sanitize_harness(
        _harness(
            web_search = {"enabled": True, "provider": "ddgs", "api_key": None},
            skills = [{"name": "Research", "instructions": "Prefer primary sources."}],
        )
    )
    assert secrets == {}
    assert snapshot["web_search"]["api_key_set"] is False
    text = _instructions(snapshot, "Answer briefly.")
    assert "Answer briefly." in text
    assert "# Skill: Research" in text
    assert "Prefer primary sources." in text


def test_storage_round_trips_harness_and_agent_metadata(tmp_path, monkeypatch):
    db_path = tmp_path / "studio.db"
    monkeypatch.setattr(battleground_db, "studio_db_path", lambda: db_path)
    battleground_db.reset_schema_cache()
    source = battleground_db.create_source(
        kind = "external_openai", name = "A", ref = "http://example/v1", external_model = "m"
    )
    harness = {"enabled": True, "version": 1, "web_search": {"provider": "ddgs"}}
    test = battleground_db.create_test(
        name = "Harness",
        slots = [{"source_id": source["id"]}],
        harness = harness,
    )
    session = battleground_db.create_session(mode = "public", test_id = test["id"])
    turn = battleground_db.create_turn(session["id"], "hello")
    response = battleground_db.create_response(
        turn_id = turn["id"],
        session_id = session["id"],
        source_id = source["id"],
        model_identity = "A · m",
        content = "hi",
        agent_meta = {"model_turns": 2, "web_search_calls": [{"query": "hello"}]},
    )

    assert battleground_db.get_test(test["id"])["harness"] == harness
    assert battleground_db.get_response(response["id"])["agent_meta"]["model_turns"] == 2
    battleground_db.reset_schema_cache()
