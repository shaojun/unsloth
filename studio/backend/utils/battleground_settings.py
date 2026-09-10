# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted kill switch for public Model Playground sharing links."""

from __future__ import annotations

from typing import Any

PLAYGROUND_SHARING_SETTING_KEY = "playground_public_sharing_enabled"
DEFAULT_PLAYGROUND_SHARING_ENABLED = True


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
