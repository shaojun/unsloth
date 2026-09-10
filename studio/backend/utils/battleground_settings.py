# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted kill switch for public Model Battleground sharing links."""

from __future__ import annotations

from typing import Any

BATTLEGROUND_SHARING_SETTING_KEY = "battleground_public_sharing_enabled"
# Stored key predates the playground → battleground rename; read it as a
# fallback so a previously disabled kill switch does not silently re-enable.
LEGACY_SHARING_SETTING_KEY = "playground_public_sharing_enabled"
DEFAULT_BATTLEGROUND_SHARING_ENABLED = True


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


def get_battleground_sharing_enabled() -> bool:
    """Read the public-battleground-sharing kill switch. Fails closed."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(BATTLEGROUND_SHARING_SETTING_KEY, None)
        if stored is None:
            stored = get_app_setting(LEGACY_SHARING_SETTING_KEY, None)
    except Exception:
        return False
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_BATTLEGROUND_SHARING_ENABLED


def set_battleground_sharing_enabled(value: Any) -> bool:
    """Persist whether public battleground links are accepted."""
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("Public battleground sharing must be true or false.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({BATTLEGROUND_SHARING_SETTING_KEY: parsed})
    return parsed
