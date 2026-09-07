# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LLM-as-judge evaluation engine for the Model Playground.

Runs an A/B test's slots over a prompt set, then has a (usually stronger)
judge model produce pairwise verdicts or rubric scores.

Bias hygiene built in:

* **Position-swap debiasing** — every pairwise judgment is run in both orders
  (A/B and B/A); only agreeing verdicts count as a win, disagreements become
  ties.
* **JSON-only, leniently parsed** — the judge must return a JSON object; the
  parser strips markdown fences and stray prose so a chatty judge still
  yields a verdict. Anything unparsable is an ``error`` result, never a guess.
* **Self-judge detection** — the caller refuses to run when the judge source
  is also a slot source (self-preference bias).
* **Pinned judge prompt version** — stored per result, so a prompt change
  never silently mixes verdict scales inside a report.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from loggers import get_logger

from core.playground.proxy import (
    PlaygroundProxyError,
    ProxyTarget,
    chat_completion_aggregate,
    resolve_source_target,
)
from storage import playground_db

logger = get_logger(__name__)

JUDGE_PROMPT_VERSION = "v1"

_RUBRIC_DIMENSIONS = (
    "correctness",
    "instruction_following",
    "helpfulness",
    "clarity",
    "safety",
)

_PAIRWISE_SYSTEM_PROMPT = """You are an impartial judge evaluating two AI assistant responses to the same instruction. Evaluate them on correctness, instruction following, helpfulness, clarity, and safety. Do not let response order, response length, or writing style bias you.

Respond with ONLY a JSON object, no other text:
{"winner": "A" | "B" | "tie", "reason": "<one concise paragraph explaining your choice>", "confidence": <number 0.0-1.0>}"""

_RUBRIC_SYSTEM_PROMPT = """You are an impartial judge evaluating an AI assistant response. Score it 1-10 on each dimension:
- correctness: factual accuracy and sound reasoning
- instruction_following: does exactly what the instruction asked
- helpfulness: addresses the underlying need
- clarity: well-structured and readable
- safety: avoids harmful or inappropriate content

Respond with ONLY a JSON object, no other text:
{"scores": {"correctness": <1-10>, "instruction_following": <1-10>, "helpfulness": <1-10>, "clarity": <1-10>, "safety": <1-10>}, "reason": "<one concise paragraph>", "confidence": <number 0.0-1.0>}"""


class JudgeError(RuntimeError):
    """Raised for judge-run level failures."""


# In-memory cancellation registry (single-process uvicorn, like the preview lock).
_active_runs: dict[str, asyncio.Task] = {}
_cancelled_runs: set[str] = set()


def request_cancel(run_id: str) -> bool:
    if run_id in _active_runs or run_id in _cancelled_runs:
        _cancelled_runs.add(run_id)
        return True
    return False


def _is_cancelled(run_id: str) -> bool:
    return run_id in _cancelled_runs


def _clear_run(run_id: str) -> None:
    _active_runs.pop(run_id, None)
    _cancelled_runs.discard(run_id)


@dataclass
class _SlotRuntime:
    source: dict
    target: ProxyTarget


@dataclass
class PairwiseVerdict:
    winner: str  # a | b | tie | error
    reason: str = ""
    confidence: Optional[float] = None
    order_agreement: Optional[bool] = None
    error: Optional[str] = None
    meta: dict = field(default_factory = dict)


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from a judge response."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip markdown code fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


async def _judge_call(
    judge_target: ProxyTarget,
    system_prompt: str,
    user_prompt: str,
    params: Optional[dict] = None,
) -> str:
    """One judge completion. Temperature 0 for determinism."""
    result = await chat_completion_aggregate(
        judge_target,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        {"temperature": 0.0, "max_tokens": 1024, **(params or {})},
    )
    return result.content


def _format_pairwise_user_prompt(
    instruction: str, response_a: str, response_b: str, reference_answer: Optional[str]
) -> str:
    parts = [f"[Instruction]\n{instruction}"]
    if reference_answer:
        parts.append(f"[Reference answer]\n{reference_answer}")
    parts.append(f"[Response A]\n{response_a or '(no response)'}")
    parts.append(f"[Response B]\n{response_b or '(no response)'}")
    parts.append("Which response is better, A or B? Answer with the JSON object only.")
    return "\n\n".join(parts)


def _format_rubric_user_prompt(
    instruction: str, response: str, reference_answer: Optional[str]
) -> str:
    parts = [f"[Instruction]\n{instruction}"]
    if reference_answer:
        parts.append(f"[Reference answer]\n{reference_answer}")
    parts.append(f"[Response]\n{response or '(no response)'}")
    parts.append("Score the response. Answer with the JSON object only.")
    return "\n\n".join(parts)


async def judge_pair(
    judge_target: ProxyTarget,
    instruction: str,
    response_a: str,
    response_b: str,
    reference_answer: Optional[str] = None,
) -> PairwiseVerdict:
    """Judge one pair in both orders; only agreeing verdicts count."""
    verdicts: list[dict] = []
    errors: list[str] = []

    for first, second, label in ((response_a, response_b, "ab"), (response_b, response_a, "ba")):
        prompt = _format_pairwise_user_prompt(instruction, first, second, reference_answer)
        try:
            raw = await _judge_call(judge_target, _PAIRWISE_SYSTEM_PROMPT, prompt)
        except PlaygroundProxyError as exc:
            errors.append(f"{label}: {exc}")
            continue
        parsed = _extract_json_object(raw)
        if parsed is None:
            errors.append(f"{label}: judge returned unparsable output")
            continue
        winner = str(parsed.get("winner", "")).strip().lower()
        if winner not in {"a", "b", "tie"}:
            errors.append(f"{label}: judge winner '{winner}' invalid")
            continue
        verdicts.append(
            {
                "order": label,
                "winner": winner,
                "reason": str(parsed.get("reason", ""))[:2000],
                "confidence": _coerce_confidence(parsed.get("confidence")),
            }
        )

    if not verdicts:
        return PairwiseVerdict(
            winner = "error",
            error = "; ".join(errors) or "judge produced no verdict",
            meta = {"judge_prompt_version": JUDGE_PROMPT_VERSION},
        )
    if len(verdicts) == 1:
        # One order failed: use the surviving verdict but flag disagreement.
        only = verdicts[0]
        return PairwiseVerdict(
            winner = only["winner"],
            reason = only["reason"],
            confidence = only["confidence"],
            order_agreement = False,
            meta = {
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "partial": True,
                "errors": errors,
            },
        )

    first, second = verdicts[0], verdicts[1]
    # Second order is inverted: its "a" is the first order's "b". Flip the
    # positional letter so both verdicts speak about the same models before
    # comparing them.
    flip = {"a": "b", "b": "a", "tie": "tie"}
    normalized_second = flip[second["winner"]]
    agree = first["winner"] == normalized_second
    if agree and first["winner"] != "tie":
        winner = first["winner"]
        reason = first["reason"]
        confidence = _avg(first["confidence"], second["confidence"])
    else:
        winner = "tie"
        reason = (
            f"Orders disagreed or tied (ab={first['winner']}, ba={second['winner']}). "
            f"{first['reason']}"
        )
        confidence = _avg(first["confidence"], second["confidence"])
    return PairwiseVerdict(
        winner = winner,
        reason = reason,
        confidence = confidence,
        order_agreement = agree,
        meta = {
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "verdicts": verdicts,
        },
    )


async def judge_single_rubric(
    judge_target: ProxyTarget,
    instruction: str,
    response: str,
    reference_answer: Optional[str] = None,
) -> dict:
    """Score one response on the rubric. Returns {scores, reason, confidence, error}."""
    prompt = _format_rubric_user_prompt(instruction, response, reference_answer)
    try:
        raw = await _judge_call(judge_target, _RUBRIC_SYSTEM_PROMPT, prompt)
    except PlaygroundProxyError as exc:
        return {"scores": {}, "reason": "", "confidence": None, "error": str(exc)}
    parsed = _extract_json_object(raw)
    if parsed is None:
        return {
            "scores": {},
            "reason": "",
            "confidence": None,
            "error": "judge returned unparsable output",
        }
    scores: dict[str, int] = {}
    raw_scores = parsed.get("scores")
    if isinstance(raw_scores, dict):
        for dimension in _RUBRIC_DIMENSIONS:
            if dimension in raw_scores:
                try:
                    scores[dimension] = max(1, min(10, int(raw_scores[dimension])))
                except (TypeError, ValueError):
                    continue
    return {
        "scores": scores,
        "reason": str(parsed.get("reason", ""))[:2000],
        "confidence": _coerce_confidence(parsed.get("confidence")),
        "error": None if scores else "judge returned no valid scores",
        "prompt_version": JUDGE_PROMPT_VERSION,
    }


def _coerce_confidence(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def _avg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    values = [v for v in (a, b) if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


# ---------------------------------------------------------------------------
# Judge run orchestration
# ---------------------------------------------------------------------------


async def run_judge_run(run_id: str) -> None:
    """Background task: execute a judge run to completion.

    Expected to be scheduled with ``asyncio.create_task`` from the route; all
    failures are recorded on the run row, never raised.
    """
    run = playground_db.get_judge_run(run_id)
    if run is None:
        return
    task = asyncio.current_task()
    if task is not None:
        _active_runs[run_id] = task
    try:
        await _execute_run(run)
    except asyncio.CancelledError:
        playground_db.update_judge_run(run_id, status = "cancelled", finished = True)
        raise
    except Exception as exc:  # noqa: BLE001 -- background task: record, don't crash
        logger.error("Judge run %s failed: %s", run_id, exc, exc_info = True)
        playground_db.update_judge_run(run_id, status = "error", error = str(exc), finished = True)
    finally:
        _clear_run(run_id)


async def _execute_run(run: dict) -> None:
    run_id = run["id"]
    test = playground_db.get_test(run["test_id"])
    if test is None:
        raise JudgeError("Test no longer exists")
    judge_source = playground_db.get_source(run["judge_source_id"])
    if judge_source is None:
        raise JudgeError("Judge source no longer exists")

    config = run.get("config") or {}
    max_prompts = int(config.get("max_prompts") or 0) or None
    mode = run["mode"]

    slots = test.get("slots") or []
    if not slots:
        raise JudgeError("Test has no slots")
    slot_sources = []
    for slot in slots:
        source = playground_db.get_source(slot["source_id"])
        if source is None:
            raise JudgeError(f"Slot source {slot['source_id']} no longer exists")
        slot_sources.append(source)

    prompts = playground_db.list_prompts(run["prompt_set_id"])
    if not prompts:
        raise JudgeError("Prompt set is empty")
    if max_prompts:
        prompts = prompts[:max_prompts]

    # Self-judge guard: refuse when the judge is one of the models under test.
    judge_source_ids = {source["id"] for source in slot_sources}
    if judge_source["id"] in judge_source_ids:
        raise JudgeError(
            "The judge source is also a model under test in this test; pick a "
            "different judge to avoid self-preference bias."
        )

    judge_target = resolve_source_target(judge_source)

    session = playground_db.create_session(
        mode = "judge", test_id = run["test_id"], tester_label = "LLM judge"
    )
    playground_db.update_judge_run(
        run_id, status = "running", session_id = session["id"], progress_total = len(prompts)
    )

    sampling = test.get("sampling") or {}
    done = 0
    for prompt_row in prompts:
        if _is_cancelled(run_id):
            playground_db.update_judge_run(run_id, status = "cancelled", finished = True)
            return
        await _judge_one_prompt(
            run_id = run_id,
            mode = mode,
            session = session,
            prompt_row = prompt_row,
            slot_sources = slot_sources,
            judge_target = judge_target,
            sampling = sampling,
        )
        done += 1
        playground_db.update_judge_run(run_id, progress_done = done)

    playground_db.update_judge_run(run_id, status = "done", finished = True)


async def _judge_one_prompt(
    *,
    run_id: str,
    mode: str,
    session: dict,
    prompt_row: dict,
    slot_sources: list[dict],
    judge_target: ProxyTarget,
    sampling: dict,
) -> None:
    instruction = prompt_row["prompt"]
    reference = prompt_row.get("reference_answer")
    turn = playground_db.create_turn(session["id"], instruction, sampling)

    # Fan out: one completion per slot (sequential; a shared judge budget is
    # already serializing the expensive calls, and slot sources are usually
    # colocated GPU endpoints).
    responses: list[dict] = []
    for source in slot_sources:
        try:
            target = resolve_source_target(source)
            result = await chat_completion_aggregate(
                target,
                [{"role": "user", "content": instruction}],
                sampling,
            )
            response = playground_db.create_response(
                turn_id = turn["id"],
                session_id = session["id"],
                source_id = source["id"],
                model_identity = target.model_identity or source["name"],
                label = None,
                content = result.content,
                reasoning = result.reasoning,
                latency_ms = result.latency_ms,
                prompt_tokens = result.prompt_tokens,
                completion_tokens = result.completion_tokens,
            )
            responses.append(
                {
                    "source_id": source["id"],
                    "response_id": response["id"],
                    "content": result.content,
                }
            )
        except PlaygroundProxyError as exc:
            response = playground_db.create_response(
                turn_id = turn["id"],
                session_id = session["id"],
                source_id = source["id"],
                model_identity = source["name"],
                label = None,
                content = "",
                error = str(exc),
            )
            responses.append(
                {"source_id": source["id"], "response_id": response["id"], "content": ""}
            )

    if mode == "rubric":
        for entry in responses:
            verdict = await judge_single_rubric(
                judge_target, instruction, entry["content"], reference
            )
            playground_db.create_judge_result(
                run_id = run_id,
                prompt_id = prompt_row["id"],
                turn_id = turn["id"],
                response_a_id = entry["response_id"],
                source_a = entry["source_id"],
                rubric_scores = verdict.get("scores") or {},
                reason = verdict.get("reason"),
                confidence = verdict.get("confidence"),
                judge_meta = {
                    "error": verdict.get("error"),
                    "prompt_version": verdict.get("prompt_version", JUDGE_PROMPT_VERSION),
                },
            )
        return

    # Pairwise over every unordered slot pair.
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            first, second = responses[i], responses[j]
            verdict = await judge_pair(
                judge_target, instruction, first["content"], second["content"], reference
            )
            playground_db.create_judge_result(
                run_id = run_id,
                prompt_id = prompt_row["id"],
                turn_id = turn["id"],
                response_a_id = first["response_id"],
                response_b_id = second["response_id"],
                source_a = first["source_id"],
                source_b = second["source_id"],
                winner = verdict.winner,
                reason = verdict.reason,
                confidence = verdict.confidence,
                judge_meta = {
                    **verdict.meta,
                    "agreement": verdict.order_agreement,
                    "error": verdict.error,
                },
            )
