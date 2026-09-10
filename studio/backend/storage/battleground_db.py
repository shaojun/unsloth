# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SQLite storage for the Model Battleground.

Same pattern as ``studio_db.py`` / ``providers_db.py``: module-level functions,
raw sqlite3, WAL, per-function connections, idempotent additive migrations.

Domain model:

* ``battleground_sources``     — model catalog entries (external OpenAI-compatible
  endpoints, typically a manually started ``vllm serve`` instance). External API
  keys are NOT stored here: they live in ``storage.credential_secrets`` under the
  ``battleground_api_key`` kind; this table only tracks whether one is set.
* ``battleground_tests``       — A/B (or single-model) test configuration.
* ``battleground_sessions``    — one tester's conversation on a test (or a
  single-model play session, or a synthetic judge session).
* ``battleground_turns``       — one user message, plus the effective sampling
  params so feedback stays reproducible.
* ``battleground_responses``   — per (turn, source): full response text, latency,
  token usage. The real model identity lives here server-side and is only sent
  to blind testers after the session is finished (every turn voted).
* ``battleground_feedback``    — human ratings / pick-best votes and judge
  verdicts (all four kinds share one table; reports split by ``kind``).
* ``battleground_prompt_sets`` / ``battleground_prompts`` — evaluation batteries.
* ``battleground_judge_runs`` / ``battleground_judge_results`` — LLM-as-judge jobs.
"""

from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from utils.paths import studio_db_path, ensure_dir

_schema_lock = threading.Lock()
_schema_ready = False

# Sentinel for "field not provided" in partial updates (None means "clear").
_UNSET = object()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_LEGACY_TABLE_RENAMES: tuple[tuple[str, str], ...] = (
    ("playground_sources", "battleground_sources"),
    ("playground_tests", "battleground_tests"),
    ("playground_sessions", "battleground_sessions"),
    ("playground_turns", "battleground_turns"),
    ("playground_responses", "battleground_responses"),
    ("playground_feedback", "battleground_feedback"),
    ("playground_prompt_sets", "battleground_prompt_sets"),
    ("playground_prompts", "battleground_prompts"),
    ("playground_judge_runs", "battleground_judge_runs"),
    ("playground_judge_results", "battleground_judge_results"),
)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _migrate_legacy_playground_tables(conn: sqlite3.Connection) -> None:
    """Rename pre-battleground playground_* tables, preserving all data.

    Standalone judge runs (created from a source list, not from an A/B test)
    need a nullable ``test_id``; legacy judge-runs tables declare it NOT NULL,
    so that one table is rebuilt instead of plain-renamed.
    """
    for legacy, modern in _LEGACY_TABLE_RENAMES:
        if not _table_exists(conn, legacy) or _table_exists(conn, modern):
            continue
        if legacy == "playground_judge_runs":
            # Rebuild with nullable test_id and copy the rows across.
            conn.execute(
                """
                CREATE TABLE battleground_judge_runs_migrated (
                    id TEXT NOT NULL PRIMARY KEY,
                    test_id TEXT,
                    prompt_set_id TEXT NOT NULL,
                    judge_source_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    session_id TEXT,
                    progress_done INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO battleground_judge_runs_migrated
                SELECT id, test_id, prompt_set_id, judge_source_id, mode, status,
                       config_json, session_id, progress_done, progress_total,
                       error, created_at, finished_at
                FROM playground_judge_runs
                """
            )
            conn.execute("DROP TABLE playground_judge_runs")
            conn.execute(
                "ALTER TABLE battleground_judge_runs_migrated RENAME TO battleground_judge_runs"
            )
        else:
            conn.execute(f"ALTER TABLE {legacy} RENAME TO {modern}")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create all battleground tables if absent. Additive migrations only."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    _migrate_legacy_playground_tables(conn)
    # The legacy-table migration contains DML (copy INSERT), which opens an
    # implicit transaction; commit so the rename survives the connection.
    conn.commit()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS battleground_sources (
            id TEXT NOT NULL PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            ref TEXT NOT NULL,
            external_model TEXT,
            api_key_set INTEGER NOT NULL DEFAULT 0,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS battleground_tests (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            show_model_cards INTEGER NOT NULL DEFAULT 0,
            reveal_after_vote INTEGER NOT NULL DEFAULT 1,
            system_prompt TEXT,
            sampling_json TEXT NOT NULL DEFAULT '{}',
            slots_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS battleground_sessions (
            id TEXT NOT NULL PRIMARY KEY,
            test_id TEXT,
            mode TEXT NOT NULL,
            source_id TEXT,
            tester_label TEXT,
            label_map_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT
        )
        """
    )
    # Additive migration: sessions may be explicitly finished by the tester
    # ("finish & reveal"); identities stay hidden until then.
    session_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(battleground_sessions)").fetchall()
    }
    if "finished_at" not in session_cols:
        conn.execute("ALTER TABLE battleground_sessions ADD COLUMN finished_at TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS battleground_turns (
            id TEXT NOT NULL PRIMARY KEY,
            session_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            params_json TEXT NOT NULL DEFAULT '{}',
            revealed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS battleground_responses (
            id TEXT NOT NULL PRIMARY KEY,
            turn_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            instance_id TEXT,
            model_identity TEXT NOT NULL,
            label TEXT,
            content TEXT NOT NULL DEFAULT '',
            reasoning TEXT,
            error TEXT,
            latency_ms REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS battleground_feedback (
            id TEXT NOT NULL PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT,
            response_id TEXT,
            kind TEXT NOT NULL,
            rating TEXT,
            tags_json TEXT,
            comment TEXT,
            chosen_response_id TEXT,
            judge_source_id TEXT,
            judge_meta_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS battleground_prompt_sets (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            origin TEXT NOT NULL DEFAULT 'manual',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS battleground_prompts (
            id TEXT NOT NULL PRIMARY KEY,
            set_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            reference_answer TEXT,
            tags_json TEXT,
            order_idx INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS battleground_judge_runs (
            id TEXT NOT NULL PRIMARY KEY,
            -- Nullable: standalone auto-eval runs are not tied to an A/B test.
            test_id TEXT,
            prompt_set_id TEXT NOT NULL,
            judge_source_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL DEFAULT '{}',
            session_id TEXT,
            progress_done INTEGER NOT NULL DEFAULT 0,
            progress_total INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS battleground_judge_results (
            id TEXT NOT NULL PRIMARY KEY,
            run_id TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            response_a_id TEXT,
            response_b_id TEXT,
            source_a TEXT,
            source_b TEXT,
            winner TEXT,
            reason TEXT,
            confidence REAL,
            rubric_scores_json TEXT,
            judge_meta_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    _conn_ensure_indexes(conn)


def _conn_ensure_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_battleground_turns_session ON battleground_turns(session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_battleground_responses_turn ON battleground_responses(turn_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_battleground_responses_session ON battleground_responses(session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_battleground_feedback_session ON battleground_feedback(session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_battleground_judge_results_run ON battleground_judge_results(run_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_battleground_prompts_set ON battleground_prompts(set_id)"
    )


def get_connection() -> sqlite3.Connection:
    """Open studio.db with WAL mode; create battleground schema once per process."""
    global _schema_ready
    db_path = studio_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    if not _schema_ready:
        with _schema_lock:
            if not _schema_ready:
                try:
                    _ensure_schema(conn)
                    _schema_ready = True
                except Exception:
                    conn.close()
                    raise
    return conn


def reset_schema_cache() -> None:
    """Test hook: force the next connection to re-run schema init."""
    global _schema_ready
    with _schema_lock:
        _schema_ready = False


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

_SOURCE_KINDS = {"external_openai"}


def create_source(
    kind: str,
    name: str,
    ref: str,
    external_model: Optional[str] = None,
    api_key_set: bool = False,
    notes: Optional[str] = None,
    source_id: Optional[str] = None,
) -> dict:
    if kind not in _SOURCE_KINDS:
        raise ValueError(f"Unknown source kind: {kind}")
    now = _now()
    source_id = source_id or _new_id("pgsrc")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO battleground_sources (
                id, kind, name, ref, external_model, api_key_set, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                kind,
                name,
                ref,
                external_model,
                1 if api_key_set else 0,
                notes,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return get_source(source_id) or {"id": source_id}


def update_source(
    source_id: str,
    name: Optional[str] = None,
    ref: Optional[str] = None,
    external_model: Optional[str] = None,
    api_key_set: Optional[bool] = None,
    notes: Optional[str] = None,
) -> Optional[dict]:
    updates = []
    params: list = []
    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if ref is not None:
        updates.append("ref = ?")
        params.append(ref)
    if external_model is not None:
        updates.append("external_model = ?")
        params.append(external_model)
    if api_key_set is not None:
        updates.append("api_key_set = ?")
        params.append(1 if api_key_set else 0)
    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)
    if not updates:
        return get_source(source_id)
    updates.append("updated_at = ?")
    params.append(_now())
    params.append(source_id)
    conn = get_connection()
    try:
        cursor = conn.execute(
            f"UPDATE battleground_sources SET {', '.join(updates)} WHERE id = ?", params
        )
        conn.commit()
        if cursor.rowcount == 0:
            return None
    finally:
        conn.close()
    return get_source(source_id)


def delete_source(source_id: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM battleground_sources WHERE id = ?", (source_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_source(source_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM battleground_sources WHERE id = ?", (source_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_sources() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM battleground_sources ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests (A/B or single-model)
# ---------------------------------------------------------------------------


def create_test(
    name: str,
    slots: list[dict],
    show_model_cards: bool = False,
    reveal_after_vote: bool = True,
    system_prompt: Optional[str] = None,
    sampling: Optional[dict] = None,
    test_id: Optional[str] = None,
) -> dict:
    now = _now()
    test_id = test_id or _new_id("pgtest")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO battleground_tests (
                id, name, status, show_model_cards, reveal_after_vote,
                system_prompt, sampling_json, slots_json, created_at, updated_at
            ) VALUES (?, ?, 'active', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                test_id,
                name,
                1 if show_model_cards else 0,
                1 if reveal_after_vote else 0,
                system_prompt,
                json.dumps(sampling or {}),
                json.dumps(slots),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return get_test(test_id) or {"id": test_id}


def update_test(
    test_id: str,
    name: Optional[str] = None,
    status: Optional[str] = None,
    show_model_cards: Optional[bool] = None,
    reveal_after_vote: Optional[bool] = None,
    system_prompt: Optional[str] = None,
    sampling: Optional[dict] = None,
    slots: Optional[list[dict]] = None,
) -> Optional[dict]:
    updates = []
    params: list = []
    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if show_model_cards is not None:
        updates.append("show_model_cards = ?")
        params.append(1 if show_model_cards else 0)
    if reveal_after_vote is not None:
        updates.append("reveal_after_vote = ?")
        params.append(1 if reveal_after_vote else 0)
    if system_prompt is not None:
        updates.append("system_prompt = ?")
        params.append(system_prompt)
    if sampling is not None:
        updates.append("sampling_json = ?")
        params.append(json.dumps(sampling))
    if slots is not None:
        updates.append("slots_json = ?")
        params.append(json.dumps(slots))
    if not updates:
        return get_test(test_id)
    updates.append("updated_at = ?")
    params.append(_now())
    params.append(test_id)
    conn = get_connection()
    try:
        cursor = conn.execute(
            f"UPDATE battleground_tests SET {', '.join(updates)} WHERE id = ?", params
        )
        conn.commit()
        if cursor.rowcount == 0:
            return None
    finally:
        conn.close()
    return get_test(test_id)


def delete_test(test_id: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM battleground_tests WHERE id = ?", (test_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_test(test_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM battleground_tests WHERE id = ?", (test_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["show_model_cards"] = bool(data["show_model_cards"])
        data["reveal_after_vote"] = bool(data["reveal_after_vote"])
        data["sampling"] = _load_json(data.pop("sampling_json", None), {})
        data["slots"] = _load_json(data.pop("slots_json", None), [])
        return data
    finally:
        conn.close()


def list_tests(include_archived: bool = False) -> list[dict]:
    conn = get_connection()
    try:
        query = "SELECT * FROM battleground_tests"
        if not include_archived:
            query += " WHERE status = 'active'"
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["show_model_cards"] = bool(data["show_model_cards"])
            data["reveal_after_vote"] = bool(data["reveal_after_vote"])
            data["sampling"] = _load_json(data.pop("sampling_json", None), {})
            data["slots"] = _load_json(data.pop("slots_json", None), [])
            out.append(data)
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Sessions / turns / responses
# ---------------------------------------------------------------------------


def create_session(
    mode: str,
    test_id: Optional[str] = None,
    source_id: Optional[str] = None,
    tester_label: Optional[str] = None,
    label_map: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> dict:
    now = _now()
    session_id = session_id or _new_id("pgses")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO battleground_sessions (
                id, test_id, mode, source_id, tester_label, label_map_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                test_id,
                mode,
                source_id,
                tester_label,
                json.dumps(label_map) if label_map is not None else None,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return get_session(session_id) or {"id": session_id}


def get_session(session_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM battleground_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["label_map"] = _load_json(data.pop("label_map_json", None), {})
        return data
    finally:
        conn.close()


def touch_session(session_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE battleground_sessions SET updated_at = ? WHERE id = ?",
            (_now(), session_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_sessions_for_test(test_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM battleground_sessions WHERE test_id = ? ORDER BY created_at",
            (test_id,),
        ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["label_map"] = _load_json(data.pop("label_map_json", None), {})
            out.append(data)
        return out
    finally:
        conn.close()


def create_turn(
    session_id: str,
    prompt: str,
    params: Optional[dict] = None,
    turn_id: Optional[str] = None,
) -> dict:
    now = _now()
    turn_id = turn_id or _new_id("pgturn")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO battleground_turns (id, session_id, prompt, params_json, revealed, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (turn_id, session_id, prompt, json.dumps(params or {}), now),
        )
        conn.commit()
    finally:
        conn.close()
    touch_session(session_id)
    return {
        "id": turn_id,
        "session_id": session_id,
        "prompt": prompt,
        "params": params or {},
        "revealed": False,
        "created_at": now,
    }


def get_turn(turn_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM battleground_turns WHERE id = ?", (turn_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["params"] = _load_json(data.pop("params_json", None), {})
        data["revealed"] = bool(data["revealed"])
        return data
    finally:
        conn.close()


def list_turns(session_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM battleground_turns WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["params"] = _load_json(data.pop("params_json", None), {})
            data["revealed"] = bool(data["revealed"])
            out.append(data)
        return out
    finally:
        conn.close()


def mark_turn_revealed(turn_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("UPDATE battleground_turns SET revealed = 1 WHERE id = ?", (turn_id,))
        conn.commit()
    finally:
        conn.close()


def finish_session(session_id: str) -> None:
    """Stamp the session as finished and reveal every turn in it.

    Idempotent: the finished_at stamp is only written once, and re-marking
    already-revealed turns is a no-op.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE battleground_sessions
            SET finished_at = COALESCE(finished_at, ?), updated_at = ?
            WHERE id = ?
            """,
            (_now(), _now(), session_id),
        )
        conn.execute(
            "UPDATE battleground_turns SET revealed = 1 WHERE session_id = ?", (session_id,)
        )
        conn.commit()
    finally:
        conn.close()


def create_response(
    turn_id: str,
    session_id: str,
    source_id: str,
    model_identity: str,
    label: Optional[str] = None,
    instance_id: Optional[str] = None,
    content: str = "",
    reasoning: Optional[str] = None,
    error: Optional[str] = None,
    latency_ms: Optional[float] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    response_id: Optional[str] = None,
) -> dict:
    now = _now()
    response_id = response_id or _new_id("pgresp")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO battleground_responses (
                id, turn_id, session_id, source_id, instance_id, model_identity,
                label, content, reasoning, error, latency_ms, prompt_tokens,
                completion_tokens, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response_id,
                turn_id,
                session_id,
                source_id,
                instance_id,
                model_identity,
                label,
                content,
                reasoning,
                error,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": response_id,
        "turn_id": turn_id,
        "session_id": session_id,
        "source_id": source_id,
        "model_identity": model_identity,
        "label": label,
        "content": content,
        "created_at": now,
    }


def get_response(response_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM battleground_responses WHERE id = ?", (response_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_responses_for_turn(turn_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM battleground_responses WHERE turn_id = ? ORDER BY created_at",
            (turn_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def list_responses_for_session(session_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM battleground_responses WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

_FEEDBACK_KINDS = {"rating", "pick_best", "judge_pairwise", "judge_rubric"}


def create_feedback(
    session_id: str,
    kind: str,
    turn_id: Optional[str] = None,
    response_id: Optional[str] = None,
    rating: Optional[str] = None,
    tags: Optional[list[str]] = None,
    comment: Optional[str] = None,
    chosen_response_id: Optional[str] = None,
    judge_source_id: Optional[str] = None,
    judge_meta: Optional[dict] = None,
    feedback_id: Optional[str] = None,
) -> dict:
    if kind not in _FEEDBACK_KINDS:
        raise ValueError(f"Unknown feedback kind: {kind}")
    now = _now()
    feedback_id = feedback_id or _new_id("pgfb")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO battleground_feedback (
                id, session_id, turn_id, response_id, kind, rating, tags_json,
                comment, chosen_response_id, judge_source_id, judge_meta_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                session_id,
                turn_id,
                response_id,
                kind,
                rating,
                json.dumps(tags) if tags else None,
                comment,
                chosen_response_id,
                judge_source_id,
                json.dumps(judge_meta) if judge_meta else None,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": feedback_id, "created_at": now}


def list_feedback(
    session_id: Optional[str] = None, kinds: Optional[list[str]] = None
) -> list[dict]:
    conn = get_connection()
    try:
        query = "SELECT * FROM battleground_feedback"
        conditions = []
        params: list = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            conditions.append(f"kind IN ({placeholders})")
            params.extend(kinds)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at"
        rows = conn.execute(query, params).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["tags"] = _load_json(data.pop("tags_json", None), [])
            data["judge_meta"] = _load_json(data.pop("judge_meta_json", None), {})
            out.append(data)
        return out
    finally:
        conn.close()


def list_feedback_for_tests(test_ids: list[str]) -> list[dict]:
    """All feedback rows belonging to any session of the given tests."""
    if not test_ids:
        return []
    conn = get_connection()
    try:
        placeholders = ", ".join("?" for _ in test_ids)
        rows = conn.execute(
            f"""
            SELECT f.* FROM battleground_feedback f
            JOIN battleground_sessions s ON s.id = f.session_id
            WHERE s.test_id IN ({placeholders})
            ORDER BY f.created_at
            """,
            test_ids,
        ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["tags"] = _load_json(data.pop("tags_json", None), [])
            data["judge_meta"] = _load_json(data.pop("judge_meta_json", None), {})
            out.append(data)
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Prompt sets
# ---------------------------------------------------------------------------


def create_prompt_set(
    name: str,
    description: Optional[str] = None,
    origin: str = "manual",
    set_id: Optional[str] = None,
) -> dict:
    now = _now()
    set_id = set_id or _new_id("pgset")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO battleground_prompt_sets (id, name, description, origin, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (set_id, name, description, origin, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return get_prompt_set(set_id) or {"id": set_id}


def get_prompt_set(set_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM battleground_prompt_sets WHERE id = ?", (set_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["prompt_count"] = conn.execute(
            "SELECT COUNT(*) FROM battleground_prompts WHERE set_id = ?", (set_id,)
        ).fetchone()[0]
        return data
    finally:
        conn.close()


def list_prompt_sets() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM battleground_prompt_sets ORDER BY created_at DESC"
        ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["prompt_count"] = conn.execute(
                "SELECT COUNT(*) FROM battleground_prompts WHERE set_id = ?", (row["id"],)
            ).fetchone()[0]
            out.append(data)
        return out
    finally:
        conn.close()


def delete_prompt_set(set_id: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM battleground_prompt_sets WHERE id = ?", (set_id,))
        conn.execute("DELETE FROM battleground_prompts WHERE set_id = ?", (set_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_prompt_set(
    set_id: str,
    name: Optional[str] = None,
    description: object = _UNSET,
) -> Optional[dict]:
    """Partial update of a prompt set's metadata.

    ``None`` clears a field; the ``_UNSET`` sentinel leaves it unchanged.
    """
    conn = get_connection()
    try:
        assignments = ["updated_at = ?"]
        params: list = [_now()]
        if name is not None:
            assignments.append("name = ?")
            params.append(name)
        if description is not _UNSET:
            assignments.append("description = ?")
            params.append(description)  # type: ignore[arg-type]
        params.append(set_id)
        cursor = conn.execute(
            f"UPDATE battleground_prompt_sets SET {', '.join(assignments)} WHERE id = ?",
            params,
        )
        conn.commit()
        if cursor.rowcount == 0:
            return None
    finally:
        conn.close()
    return get_prompt_set(set_id)


def add_prompt(
    set_id: str,
    prompt: str,
    reference_answer: Optional[str] = None,
    tags: Optional[list[str]] = None,
    order_idx: int = 0,
    prompt_id: Optional[str] = None,
) -> dict:
    now = _now()
    prompt_id = prompt_id or _new_id("pgprm")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO battleground_prompts (id, set_id, prompt, reference_answer, tags_json, order_idx, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prompt_id,
                set_id,
                prompt,
                reference_answer,
                json.dumps(tags) if tags else None,
                order_idx,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": prompt_id,
        "set_id": set_id,
        "prompt": prompt,
        "reference_answer": reference_answer,
        "tags": tags or [],
        "order_idx": order_idx,
    }


def add_prompts_bulk(set_id: str, prompts: list[dict]) -> int:
    """Add many prompts at once. Each item: {prompt, reference_answer?, tags?}."""
    conn = get_connection()
    now = _now()
    try:
        base_idx = conn.execute(
            "SELECT COALESCE(MAX(order_idx), -1) + 1 FROM battleground_prompts WHERE set_id = ?",
            (set_id,),
        ).fetchone()[0]
        rows = []
        for offset, item in enumerate(prompts):
            rows.append(
                (
                    _new_id("pgprm"),
                    set_id,
                    item.get("prompt", ""),
                    item.get("reference_answer"),
                    json.dumps(item.get("tags")) if item.get("tags") else None,
                    base_idx + offset,
                    now,
                )
            )
        conn.executemany(
            """
            INSERT INTO battleground_prompts (id, set_id, prompt, reference_answer, tags_json, order_idx, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def list_prompts(set_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM battleground_prompts WHERE set_id = ? ORDER BY order_idx, created_at",
            (set_id,),
        ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["tags"] = _load_json(data.pop("tags_json", None), [])
            out.append(data)
        return out
    finally:
        conn.close()


def delete_prompt(prompt_id: str) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM battleground_prompts WHERE id = ?", (prompt_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_prompt(prompt_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM battleground_prompts WHERE id = ?", (prompt_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["tags"] = _load_json(data.pop("tags_json", None), [])
        return data
    finally:
        conn.close()


def update_prompt(
    prompt_id: str,
    prompt: Optional[str] = None,
    reference_answer: object = _UNSET,
) -> Optional[dict]:
    """Partial update of one prompt row.

    ``None`` clears a field; the ``_UNSET`` sentinel leaves it unchanged.
    """
    conn = get_connection()
    try:
        assignments = []
        params: list = []
        if prompt is not None:
            assignments.append("prompt = ?")
            params.append(prompt)
        if reference_answer is not _UNSET:
            assignments.append("reference_answer = ?")
            params.append(reference_answer)  # type: ignore[arg-type]
        if not assignments:
            return get_prompt(prompt_id)
        params.append(prompt_id)
        cursor = conn.execute(
            f"UPDATE battleground_prompts SET {', '.join(assignments)} WHERE id = ?",
            params,
        )
        conn.commit()
        if cursor.rowcount == 0:
            return None
    finally:
        conn.close()
    return get_prompt(prompt_id)


# ---------------------------------------------------------------------------
# Judge runs
# ---------------------------------------------------------------------------


def create_judge_run(
    prompt_set_id: str,
    judge_source_id: str,
    mode: str,
    config: Optional[dict] = None,
    test_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Create a judge run. ``test_id`` is legacy (A/B-test-scoped) runs only;
    standalone auto-eval runs carry their participants in ``config.source_ids``."""
    if mode not in {"pairwise", "rubric"}:
        raise ValueError(f"Unknown judge mode: {mode}")
    now = _now()
    run_id = run_id or _new_id("pgjudge")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO battleground_judge_runs (
                id, test_id, prompt_set_id, judge_source_id, mode, status,
                config_json, progress_done, progress_total, created_at
            ) VALUES (?, ?, ?, ?, ?, 'queued', ?, 0, 0, ?)
            """,
            (run_id, test_id, prompt_set_id, judge_source_id, mode, json.dumps(config or {}), now),
        )
        conn.commit()
    finally:
        conn.close()
    return get_judge_run(run_id) or {"id": run_id}


def update_judge_run(
    run_id: str,
    status: Optional[str] = None,
    session_id: Optional[str] = None,
    progress_done: Optional[int] = None,
    progress_total: Optional[int] = None,
    error: Optional[str] = None,
    finished: bool = False,
) -> Optional[dict]:
    updates = []
    params: list = []
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if session_id is not None:
        updates.append("session_id = ?")
        params.append(session_id)
    if progress_done is not None:
        updates.append("progress_done = ?")
        params.append(progress_done)
    if progress_total is not None:
        updates.append("progress_total = ?")
        params.append(progress_total)
    if error is not None:
        updates.append("error = ?")
        params.append(error)
    if finished:
        updates.append("finished_at = ?")
        params.append(_now())
    if not updates:
        return get_judge_run(run_id)
    conn = get_connection()
    try:
        cursor = conn.execute(
            f"UPDATE battleground_judge_runs SET {', '.join(updates)} WHERE id = ?",
            params + [run_id],
        )
        conn.commit()
        if cursor.rowcount == 0:
            return None
    finally:
        conn.close()
    return get_judge_run(run_id)


def get_judge_run(run_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM battleground_judge_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["config"] = _load_json(data.pop("config_json", None), {})
        return data
    finally:
        conn.close()


def list_judge_runs(test_id: Optional[str] = None, limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        if test_id is not None:
            rows = conn.execute(
                "SELECT * FROM battleground_judge_runs WHERE test_id = ? ORDER BY created_at DESC LIMIT ?",
                (test_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM battleground_judge_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["config"] = _load_json(data.pop("config_json", None), {})
            out.append(data)
        return out
    finally:
        conn.close()


def create_judge_result(
    run_id: str,
    prompt_id: str,
    turn_id: str,
    response_a_id: Optional[str] = None,
    response_b_id: Optional[str] = None,
    source_a: Optional[str] = None,
    source_b: Optional[str] = None,
    winner: Optional[str] = None,
    reason: Optional[str] = None,
    confidence: Optional[float] = None,
    rubric_scores: Optional[dict] = None,
    judge_meta: Optional[dict] = None,
    result_id: Optional[str] = None,
) -> str:
    now = _now()
    result_id = result_id or _new_id("pgjr")
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO battleground_judge_results (
                id, run_id, prompt_id, turn_id, response_a_id, response_b_id,
                source_a, source_b, winner, reason, confidence, rubric_scores_json,
                judge_meta_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                run_id,
                prompt_id,
                turn_id,
                response_a_id,
                response_b_id,
                source_a,
                source_b,
                winner,
                reason,
                confidence,
                json.dumps(rubric_scores) if rubric_scores else None,
                json.dumps(judge_meta) if judge_meta else None,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return result_id


def list_judge_results(run_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM battleground_judge_results WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["rubric_scores"] = _load_json(data.pop("rubric_scores_json", None), {})
            data["judge_meta"] = _load_json(data.pop("judge_meta_json", None), {})
            out.append(data)
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(raw: Optional[str], default):
    if not raw:
        return default
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return default
    return parsed
