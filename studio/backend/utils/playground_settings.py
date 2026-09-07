# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted kill switch + GPU budget settings for the Model Playground."""

from __future__ import annotations

from typing import Any

PLAYGROUND_SHARING_SETTING_KEY = "playground_public_sharing_enabled"
DEFAULT_PLAYGROUND_SHARING_ENABLED = True

PLAYGROUND_GPU_BUDGET_SETTING_KEY = "playground_gpu_budget"
# Default: leave 10% of the GPU for everything else when summing vLLM
# --gpu-memory-utilization fractions across hosted playground instances.
DEFAULT_PLAYGROUND_GPU_BUDGET = 0.9


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def get_playground_sharing_enabled() -> bool:
    """Read the public-playground-sharing kill switch. Fails closed."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(PLAYGROUND_SHARING_SETTING_KEY, None)
    except Exception:
        return False
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_PLAYGROUND_SHARING_ENABLED


def set_playground_sharing_enabled(value: Any) -> bool:
    """Persist whether public playground links are accepted."""
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("Public playground sharing must be true or false.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({PLAYGROUND_SHARING_SETTING_KEY: parsed})
    return parsed


def _coerce_fraction(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    # Clamp to (0, 1]: 0 would host nothing, >1 oversubscribes the GPU.
    return min(1.0, max(0.05, parsed))


def get_playground_gpu_budget() -> float:
    """Max summed ``--gpu-memory-utilization`` across playground instances."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(PLAYGROUND_GPU_BUDGET_SETTING_KEY, None)
    except Exception:
        return DEFAULT_PLAYGROUND_GPU_BUDGET
    if stored is None:
        return DEFAULT_PLAYGROUND_GPU_BUDGET
    return _coerce_fraction(stored, DEFAULT_PLAYGROUND_GPU_BUDGET)


def set_playground_gpu_budget(value: Any) -> float:
    from storage.studio_db import upsert_app_settings

    parsed = _coerce_fraction(value, DEFAULT_PLAYGROUND_GPU_BUDGET)
    upsert_app_settings({PLAYGROUND_GPU_BUDGET_SETTING_KEY: parsed})
    return parsed
