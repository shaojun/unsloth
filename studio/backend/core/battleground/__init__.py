# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Battleground package: source proxying and LLM-as-judge evaluation."""

from core.battleground.proxy import (
    BattlegroundProxyError,
    SourceDownError,
    chat_completion,
    chat_completion_aggregate,
    stream_chat_completion,
    test_source_connection,
)

__all__ = [
    "BattlegroundProxyError",
    "SourceDownError",
    "chat_completion",
    "chat_completion_aggregate",
    "stream_chat_completion",
    "test_source_connection",
]
