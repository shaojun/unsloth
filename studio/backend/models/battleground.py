# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for the Battleground API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class BattlegroundSourceCreate(BaseModel):
    kind: Literal["external_openai"] = "external_openai"
    name: str = Field(..., min_length = 1, max_length = 120)
    ref: str = Field(..., min_length = 1, max_length = 2048)
    external_model: Optional[str] = Field(
        None,
        max_length = 300,
        description = "Model id to call on an external OpenAI-compatible endpoint",
    )
    api_key: Optional[str] = Field(
        None,
        max_length = 4096,
        description = "Optional API key; stored encrypted server-side (credential_secrets)",
    )
    notes: Optional[str] = Field(None, max_length = 2000)


class BattlegroundSourceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length = 1, max_length = 120)
    ref: Optional[str] = Field(None, min_length = 1, max_length = 2048)
    external_model: Optional[str] = Field(None, max_length = 300)
    api_key: Optional[str] = Field(None, max_length = 4096)
    notes: Optional[str] = Field(None, max_length = 2000)


class BattlegroundTestSlot(BaseModel):
    source_id: str = Field(..., min_length = 1, max_length = 64)


class BattlegroundTestCreate(BaseModel):
    name: str = Field(..., min_length = 1, max_length = 160)
    slots: list[BattlegroundTestSlot] = Field(..., min_length = 1, max_length = 4)
    show_model_cards: bool = Field(
        False, description = "Show model identity up front (non-blind mode)"
    )
    reveal_after_vote: bool = Field(
        True, description = "Reveal model identities after the tester votes on a turn"
    )
    system_prompt: Optional[str] = Field(None, max_length = 8000)
    sampling: Optional[dict[str, Any]] = None


class BattlegroundTestUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length = 1, max_length = 160)
    status: Optional[Literal["active", "archived"]] = None
    show_model_cards: Optional[bool] = None
    reveal_after_vote: Optional[bool] = None
    system_prompt: Optional[str] = Field(None, max_length = 8000)
    sampling: Optional[dict[str, Any]] = None
    slots: Optional[list[BattlegroundTestSlot]] = Field(None, min_length = 1, max_length = 4)


class BattlegroundChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length = 0, max_length = 128000)


class BattlegroundChatRequest(BaseModel):
    messages: list[BattlegroundChatMessage] = Field(..., min_length = 1, max_length = 200)
    session_id: Optional[str] = Field(None, min_length = 1, max_length = 64)
    stream: bool = Field(True, description = "SSE stream deltas when true, single JSON when false")
    temperature: Optional[float] = Field(None, ge = 0.0, le = 2.0)
    top_p: Optional[float] = Field(None, ge = 0.0, le = 1.0)
    max_tokens: Optional[int] = Field(None, ge = 1, le = 32768)


class BattlegroundFeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length = 1, max_length = 64)
    kind: Literal["rating", "pick_best"]
    turn_id: Optional[str] = Field(None, min_length = 1, max_length = 64)
    response_id: Optional[str] = Field(None, min_length = 1, max_length = 64)
    rating: Optional[Literal["good", "bad"]] = None
    tags: Optional[list[str]] = Field(None, max_length = 12)
    comment: Optional[str] = Field(None, max_length = 8000)
    chosen_response_id: Optional[str] = Field(None, min_length = 1, max_length = 64)


class BattlegroundPromptSetCreate(BaseModel):
    name: str = Field(..., min_length = 1, max_length = 160)
    description: Optional[str] = Field(None, max_length = 2000)
    prompts: list[dict[str, Any]] = Field(
        default_factory = list,
        description = "Items: {prompt, reference_answer?, tags?}",
    )


class BattlegroundPromptSetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length = 1, max_length = 160)
    description: Optional[str] = Field(None, max_length = 2000)


class BattlegroundPromptAdd(BaseModel):
    prompt: str = Field(..., min_length = 1, max_length = 32000)
    reference_answer: Optional[str] = Field(None, max_length = 32000)
    tags: Optional[list[str]] = Field(None, max_length = 12)


class BattlegroundPromptUpdate(BaseModel):
    prompt: Optional[str] = Field(None, min_length = 1, max_length = 32000)
    reference_answer: Optional[str] = Field(None, max_length = 32000)


class BattlegroundJudgeRunCreate(BaseModel):
    """Standalone LLM-as-judge auto evaluation.

    Participants are picked directly (not from an A/B test): pairwise mode
    compares exactly two models, rubric mode scores one or more.
    """

    source_ids: list[str] = Field(..., min_length = 1, max_length = 8)
    prompt_set_id: str = Field(..., min_length = 1, max_length = 64)
    judge_source_id: str = Field(..., min_length = 1, max_length = 64)
    mode: Literal["pairwise", "rubric"] = "pairwise"
    max_prompts: Optional[int] = Field(None, ge = 1, le = 10000)


class BattlegroundPlaySessionCreate(BaseModel):
    source_id: str = Field(..., min_length = 1, max_length = 64)
