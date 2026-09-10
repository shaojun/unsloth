# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HMAC capability tokens for public Model Battleground share links.

Mirrors ``utils/preview_token.py``: the test id is guessable, so every public
battleground request must carry a signed capability token. The token domain is
separate from preview tokens (``battleground:`` prefix) so neither surface can
accept the other's links, and rotating the preview secret does not revoke
battleground links (or vice versa).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Optional

from auth.storage import get_or_create_preview_link_secret

# Versioned so the token format can evolve without silently honoring old shapes.
_BATTLEGROUND_TOKEN_VERSION = "v1"


def _canonical_payload(ref: str) -> bytes:
    # Sign the canonical ref only, never host/path, so links stay portable across
    # localhost / LAN IP / tunnel host changes.
    return f"battleground:{_BATTLEGROUND_TOKEN_VERSION}:{ref}".encode("utf-8")


def sign_battleground_ref(ref: str) -> str:
    """Return the URL-safe HMAC capability token for a canonical battleground ref."""
    mac = hmac.new(
        get_or_create_preview_link_secret(),
        _canonical_payload(ref),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(mac).rstrip(b"=").decode("ascii")


def verify_battleground_ref(ref: str, token: Optional[str]) -> bool:
    """Constant-time check that ``token`` is a valid capability for ``ref``."""
    if not token:
        return False
    try:
        provided = token.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(sign_battleground_ref(ref).encode("ascii"), provided)
