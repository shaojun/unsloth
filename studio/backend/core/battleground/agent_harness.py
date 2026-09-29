# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared Agents SDK harness for Battleground A/B participants.

Source-model play deliberately does not import or use this module. Every new
A/B test snapshots one harness and passes that same configuration to every
participant model.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from core.inference.mcp_config_import import parse_mcp_config
from storage.credential_secrets import get_secret

from .proxy import ProxyTarget, resolve_source_target

BATTLEGROUND_WEB_SEARCH_KEY_KIND = "battleground_web_search_key"
BATTLEGROUND_MCP_CONFIG_KIND = "battleground_mcp_config"

_MAX_MCP_JSON_BYTES = 64 * 1024
_MAX_RECORDED_TOOL_OUTPUT = 12000
_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class BattlegroundAgentError(RuntimeError):
    pass


@dataclass
class AgentHarnessResult:
    content: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    agent_meta: dict[str, Any] = field(default_factory = dict)


def validate_and_sanitize_harness(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate a request harness, returning its safe snapshot and secrets.

    MCP headers and the Brave key are encrypted outside ``battleground_tests``.
    Public A/B tests accept remote MCP transports only and require an explicit
    allow-list for each enabled server.
    """
    harness = json.loads(json.dumps(raw))
    secrets: dict[str, str] = {}
    search = harness.get("web_search") or {}
    if search.get("enabled") and search.get("provider") == "brave":
        key = str(search.pop("api_key", "") or "").strip()
        if not key:
            raise ValueError("Brave Search requires an API key")
        secrets["web_search"] = key
        search["api_key_set"] = True
    else:
        search.pop("api_key", None)
        search["api_key_set"] = False
    harness["web_search"] = search

    mcp_config = harness.pop("mcp_config", None) or {"mcpServers": {}}
    encoded = json.dumps(mcp_config, ensure_ascii = False)
    if len(encoded.encode("utf-8")) > _MAX_MCP_JSON_BYTES:
        raise ValueError("MCP JSON must be 64 KiB or smaller")
    entries, errors = parse_mcp_config(mcp_config)
    if errors:
        raise ValueError("Invalid MCP JSON: " + "; ".join(errors))
    raw_servers = mcp_config.get("mcpServers", mcp_config.get("servers", {}))
    safe_servers = []
    for entry in entries:
        if entry.is_stdio:
            raise ValueError(
                f"MCP server '{entry.display_name}' uses stdio; public A/B tests allow remote HTTP/SSE only"
            )
        if entry.use_oauth:
            raise ValueError(
                f"MCP server '{entry.display_name}' uses interactive OAuth, which public tests cannot complete"
            )
        spec = raw_servers.get(entry.display_name, {}) if isinstance(raw_servers, dict) else {}
        allowed = spec.get("allowedTools") if isinstance(spec, dict) else None
        if entry.is_enabled and (
            not isinstance(allowed, list)
            or not allowed
            or not all(isinstance(name, str) and name.strip() for name in allowed)
        ):
            raise ValueError(
                f"MCP server '{entry.display_name}' needs a non-empty allowedTools array"
            )
        safe_servers.append(
            {
                "name": entry.display_name,
                "url": entry.url,
                "transport": "sse" if entry.url.rstrip("/").endswith("/sse") else "streamable_http",
                "enabled": entry.is_enabled,
                "allowed_tools": allowed or [],
                "headers_set": bool(entry.headers),
            }
        )
    if entries:
        secrets["mcp_config"] = encoded
    harness["mcp_servers"] = safe_servers
    return harness, secrets


def _instructions(harness: dict[str, Any], system_prompt: Optional[str]) -> str:
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    if harness.get("instructions"):
        parts.append(str(harness["instructions"]).strip())
    for skill in harness.get("skills") or []:
        name = str(skill.get("name") or "Skill").strip()
        instructions = str(skill.get("instructions") or "").strip()
        if instructions:
            parts.append(f"# Skill: {name}\n{instructions}")
    return "\n\n".join(part for part in parts if part) or "Answer the user helpfully."


async def _brave_search(
    query: str,
    *,
    api_key: str,
    max_results: int,
    country: Optional[str],
    language: Optional[str],
) -> list[dict[str, str]]:
    params: dict[str, Any] = {"q": query, "count": max_results}
    if country:
        params["country"] = country
    if language:
        params["search_lang"] = language
    async with httpx.AsyncClient(timeout = 20.0, trust_env = False) as client:
        response = await client.get(
            _BRAVE_SEARCH_URL,
            params = params,
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
        )
        response.raise_for_status()
        payload = response.json()
    results = (payload.get("web") or {}).get("results") or []
    return [
        {
            "title": str(item.get("title") or "")[:500],
            "url": str(item.get("url") or "")[:2048],
            "snippet": str(item.get("description") or "")[:2000],
        }
        for item in results[:max_results]
        if isinstance(item, dict)
    ]


async def _ddgs_search(query: str, *, max_results: int) -> list[dict[str, str]]:
    def search() -> list[dict[str, str]]:
        from ddgs import DDGS

        rows = DDGS(timeout = 20).text(query, max_results = max_results)
        return [
            {
                "title": str(item.get("title") or "")[:500],
                "url": str(item.get("href") or item.get("url") or "")[:2048],
                "snippet": str(item.get("body") or item.get("description") or "")[:2000],
            }
            for item in rows or []
            if isinstance(item, dict)
        ]

    return await asyncio.to_thread(search)


def _safe_item(item: Any) -> dict[str, Any]:
    raw = getattr(item, "raw_item", None)
    if hasattr(raw, "model_dump"):
        value = raw.model_dump(mode = "json")
    elif hasattr(item, "to_input_item"):
        value = item.to_input_item()
    else:
        value = {"type": type(item).__name__}
    text = json.dumps(value, ensure_ascii = False, default = str)
    if len(text) > _MAX_RECORDED_TOOL_OUTPUT:
        return {"type": type(item).__name__, "truncated": True, "preview": text[:_MAX_RECORDED_TOOL_OUTPUT]}
    return value if isinstance(value, dict) else {"value": value}


def _usage(result: Any) -> tuple[Optional[int], Optional[int], int]:
    prompt = completion = 0
    responses = list(getattr(result, "raw_responses", None) or [])
    found = False
    for response in responses:
        usage = getattr(response, "usage", None)
        if usage is None:
            continue
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is not None:
            prompt += int(input_tokens)
            found = True
        if output_tokens is not None:
            completion += int(output_tokens)
            found = True
    return (prompt if found else None, completion if found else None, len(responses))


def _load_mcp_specs(test_id: str) -> list[dict[str, Any]]:
    saved = get_secret(BATTLEGROUND_MCP_CONFIG_KIND, test_id)
    if not saved:
        return []
    config = json.loads(saved)
    entries, errors = parse_mcp_config(config)
    if errors:
        raise BattlegroundAgentError("Saved MCP configuration is invalid")
    raw_servers = config.get("mcpServers", config.get("servers", {}))
    out = []
    for entry in entries:
        if entry.is_stdio or not entry.is_enabled:
            continue
        spec = raw_servers.get(entry.display_name, {}) if isinstance(raw_servers, dict) else {}
        allowed = spec.get("allowedTools", []) if isinstance(spec, dict) else []
        out.append(
            {
                "name": entry.display_name,
                "url": entry.url,
                "headers": entry.headers or {},
                "allowed_tools": allowed,
            }
        )
    return out


async def _run_agent_harness(
    *,
    source: dict,
    messages: list[dict[str, str]],
    harness: dict[str, Any],
    test_id: str,
    system_prompt: Optional[str],
    sampling: dict[str, Any],
) -> AgentHarnessResult:
    """Run one participant inside the test's snapshotted shared harness."""
    if not harness.get("enabled"):
        raise BattlegroundAgentError("Agent harness is not enabled")
    try:
        from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner
        from agents.decorators import tool
        from agents.mcp import MCPServerSse, MCPServerStreamableHttp, create_static_tool_filter
    except ImportError as exc:
        raise BattlegroundAgentError(
            "The A/B agent harness requires Python 3.10+ and the openai-agents package"
        ) from exc

    target: ProxyTarget = resolve_source_target(source)
    if not target.model:
        raise BattlegroundAgentError(f"Source '{source['name']}' needs a model ID for agent runs")

    calls: list[dict[str, Any]] = []
    tools = []
    search = harness.get("web_search") or {}
    if search.get("enabled"):
        provider = search.get("provider", "brave")
        api_key = get_secret(BATTLEGROUND_WEB_SEARCH_KEY_KIND, test_id) or ""
        max_results = int(search.get("max_results") or 5)
        tool_timeout = float(harness.get("tool_timeout_seconds") or 20.0)

        @tool(timeout = tool_timeout)
        async def web_search(query: str) -> str:
            """Search the public web and return ranked titles, URLs, and snippets."""
            started = time.perf_counter()
            error = None
            results: list[dict[str, str]] = []
            try:
                if provider == "brave":
                    if not api_key:
                        raise RuntimeError("Brave Search API key is unavailable")
                    results = await _brave_search(
                        query,
                        api_key = api_key,
                        max_results = max_results,
                        country = search.get("country"),
                        language = search.get("language"),
                    )
                else:
                    results = await _ddgs_search(query, max_results = max_results)
                return json.dumps({"query": query, "results": results}, ensure_ascii = False)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                calls.append(
                    {
                        "tool": "web_search",
                        "provider": provider,
                        "query": query,
                        "result_count": len(results),
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        "error": error,
                    }
                )

        tools.append(web_search)

    client = AsyncOpenAI(api_key = target.api_key or "not-needed", base_url = target.base_url)
    model = OpenAIChatCompletionsModel(model = target.model, openai_client = client)
    settings_kwargs = {
        key: sampling[key]
        for key in ("temperature", "top_p")
        if sampling.get(key) is not None
    }
    if sampling.get("max_tokens") is not None:
        settings_kwargs["max_tokens"] = sampling["max_tokens"]

    started = time.perf_counter()
    async with AsyncExitStack() as stack:
        mcp_servers = []
        for spec in _load_mcp_specs(test_id):
            server_cls = MCPServerSse if spec["url"].rstrip("/").endswith("/sse") else MCPServerStreamableHttp
            server = server_cls(
                name = spec["name"],
                params = {
                    "url": spec["url"],
                    "headers": spec["headers"],
                    "timeout": float(harness.get("tool_timeout_seconds") or 20.0),
                },
                cache_tools_list = True,
                tool_filter = create_static_tool_filter(
                    allowed_tool_names = spec["allowed_tools"]
                ),
                require_approval = "never",
            )
            mcp_servers.append(await stack.enter_async_context(server))

        agent = Agent(
            name = f"Battleground participant {source['id']}",
            instructions = _instructions(harness, system_prompt),
            model = model,
            tools = tools,
            mcp_servers = mcp_servers,
            mcp_config = {
                "convert_schemas_to_strict": True,
                "include_server_in_tool_names": True,
            },
            model_settings = ModelSettings(**settings_kwargs),
        )
        try:
            result = await asyncio.wait_for(
                Runner.run(
                    agent,
                    input = messages,
                    max_turns = int(harness.get("max_turns") or 8),
                    run_config = RunConfig(
                        tracing_disabled = True,
                        trace_include_sensitive_data = False,
                    ),
                ),
                timeout = float(harness.get("run_timeout_seconds") or 120.0),
            )
        except asyncio.TimeoutError as exc:
            raise BattlegroundAgentError("Agent run timed out") from exc
        except Exception as exc:
            raise BattlegroundAgentError("Agent run failed") from exc

    prompt_tokens, completion_tokens, model_turns = _usage(result)
    items = [_safe_item(item) for item in (getattr(result, "new_items", None) or [])]
    return AgentHarnessResult(
        content = str(result.final_output or ""),
        latency_ms = round((time.perf_counter() - started) * 1000, 1),
        prompt_tokens = prompt_tokens,
        completion_tokens = completion_tokens,
        agent_meta = {
            "harness_version": harness.get("version", 1),
            "model_turns": model_turns,
            "web_search_calls": calls,
            "run_items": items,
        },
    )


async def run_agent_harness(**kwargs: Any) -> AgentHarnessResult:
    """Run the harness while converting SDK/MCP setup failures into safe errors."""
    try:
        return await _run_agent_harness(**kwargs)
    except BattlegroundAgentError:
        raise
    except Exception as exc:
        raise BattlegroundAgentError("Agent harness setup failed") from exc
