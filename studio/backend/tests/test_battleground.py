# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model Battleground backend tests: storage, tokens, and route-level security
for the public /bg tester pages.

Exercises the route layer with the real routers while stubbing the proxy layer
(the mock OpenAI server in battleground_mock_openai.py covers the proxy itself
end-to-end), following the test_preview_routes.py pattern: HMAC token gating,
rate limiting, blind-identity guarantees, feedback validation, and report math.
"""

import asyncio
import json
import sqlite3
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Mirror test_preview.py: the real `loggers` package pulls in heavy handlers.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.battleground as battleground
import utils.battleground_token as battleground_token
from storage import battleground_db

_TEST_SECRET = b"unit-test-battleground-secret-0123456789abcdef"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point studio.db at a temp file and reset the schema cache."""
    db_path = tmp_path / "studio.db"
    monkeypatch.setattr("utils.paths.studio_db_path", lambda: db_path, raising = True)
    # battleground_db imported studio_db_path by value; patch its reference too.
    monkeypatch.setattr(battleground_db, "studio_db_path", lambda: db_path)
    battleground_db.reset_schema_cache()
    yield db_path
    battleground_db.reset_schema_cache()


@pytest.fixture
def app_client(isolated_db, monkeypatch):
    """FastAPI app with the battleground routers mounted, auth stubbed open."""
    monkeypatch.setattr(battleground_token, "get_or_create_preview_link_secret", lambda: _TEST_SECRET)
    import utils.preview_rate_limit as _rl

    _rl.reset()
    monkeypatch.setattr(battleground, "get_battleground_sharing_enabled", lambda: True)

    app = FastAPI()
    app.include_router(battleground.router, prefix = "/api/battleground")
    app.include_router(battleground.public_router, prefix = "/bg")
    # Auth: every Depends(get_current_subject) resolves to a test user.
    from auth.authentication import get_current_subject

    app.dependency_overrides[get_current_subject] = lambda: "tester"
    client = TestClient(app)
    yield client
    _rl.reset()


def _make_source(
    kind = "external_openai",
    name = "Mock A",
    ref = "http://127.0.0.1:1/v1",
):
    return battleground_db.create_source(kind = kind, name = name, ref = ref, external_model = "m")


# ---------------------------------------------------------------------------
# Token / security
# ---------------------------------------------------------------------------


class TestBattlegroundTokens:
    def test_sign_verify_roundtrip(self):
        token = battleground_token.sign_battleground_ref("t/abc")
        assert battleground_token.verify_battleground_ref("t/abc", token)

    def test_wrong_ref_rejected(self):
        token = battleground_token.sign_battleground_ref("t/abc")
        assert not battleground_token.verify_battleground_ref("t/other", token)

    def test_missing_token_rejected(self):
        assert not battleground_token.verify_battleground_ref("t/abc", None)

    def test_not_preview_token_compatible(self, monkeypatch):
        """Preview tokens must not unlock battleground pages (domain separation)."""
        import utils.preview_token as preview_token

        monkeypatch.setattr(
            preview_token, "get_or_create_preview_link_secret", lambda: _TEST_SECRET
        )
        preview_token_value = preview_token.sign_preview_ref("t/abc")
        assert not battleground_token.verify_battleground_ref("t/abc", preview_token_value)

    def test_garbage_token_rejected(self):
        assert not battleground_token.verify_battleground_ref("t/abc", "nonsense")
        assert not battleground_token.verify_battleground_ref("t/abc", "üñïçödé")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestBattlegroundStorage:
    def test_source_crud(self):
        source = battleground_db.create_source(kind = "external_openai", name = "S", ref = "http://x/v1")
        assert battleground_db.get_source(source["id"])["name"] == "S"
        battleground_db.update_source(source["id"], name = "S2")
        assert battleground_db.get_source(source["id"])["name"] == "S2"
        assert battleground_db.delete_source(source["id"])
        assert battleground_db.get_source(source["id"]) is None

    def test_unknown_source_kind_rejected(self):
        with pytest.raises(ValueError):
            battleground_db.create_source(kind = "local_dir", name = "S", ref = "/tmp/x")

    def test_feedback_roundtrip(self):
        source = _make_source()
        session = battleground_db.create_session(mode = "play", source_id = source["id"])
        turn = battleground_db.create_turn(session["id"], "hi")
        response = battleground_db.create_response(
            turn_id = turn["id"],
            session_id = session["id"],
            source_id = source["id"],
            model_identity = "x",
        )
        battleground_db.create_feedback(
            session_id = session["id"],
            kind = "rating",
            turn_id = turn["id"],
            response_id = response["id"],
            rating = "good",
            tags = ["correct"],
        )
        rows = battleground_db.list_feedback(session_id = session["id"])
        assert len(rows) == 1
        assert rows[0]["tags"] == ["correct"]


# ---------------------------------------------------------------------------
# Manual vLLM hosting hints
# ---------------------------------------------------------------------------


class TestTrainedModels:
    def test_completed_runs_with_output_paths(self, app_client, monkeypatch, tmp_path):
        from storage import studio_db as studio_db_mod

        def fake_list_runs(limit, offset):
            assert offset == 0
            return {
                "runs": [
                    {
                        "id": "run_1",
                        "status": "completed",
                        "display_name": "My Run",
                        "model_name": "org/model",
                        "output_dir": str(tmp_path / "out"),
                        "ended_at": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "id": "run_2",
                        "status": "error",
                        "display_name": "Bad",
                        "model_name": "org/model",
                        "output_dir": str(tmp_path / "bad"),
                        "ended_at": None,
                    },
                ],
                "total": 2,
            }

        monkeypatch.setattr(studio_db_mod, "list_runs", fake_list_runs)
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "config.json").write_text("{}", encoding = "utf-8")

        response = app_client.get("/api/battleground/trained-models")
        assert response.status_code == 200
        models = response.json()["models"]
        # Only successfully trained (completed) runs are listed.
        assert [m["run_id"] for m in models] == ["run_1"]
        assert models[0]["output_dir"] == str(tmp_path / "out")
        assert models[0]["artifact"] == "model"
        assert models[0]["name"] == "My Run"

    def test_adapter_dir_annotated(self, app_client, monkeypatch, tmp_path):
        from storage import studio_db as studio_db_mod

        monkeypatch.setattr(
            studio_db_mod,
            "list_runs",
            lambda limit, offset: {
                "runs": [
                    {
                        "id": "run_1",
                        "status": "completed",
                        "display_name": None,
                        "model_name": "org/model",
                        "output_dir": str(tmp_path / "adapter"),
                        "ended_at": None,
                    }
                ],
                "total": 1,
            },
        )
        (tmp_path / "adapter").mkdir()
        (tmp_path / "adapter" / "adapter_config.json").write_text("{}", encoding = "utf-8")

        models = app_client.get("/api/battleground/trained-models").json()["models"]
        assert models[0]["artifact"] == "adapter"
        assert models[0]["name"] == "org/model"


# ---------------------------------------------------------------------------
# Routes: public blind-test guarantees
# ---------------------------------------------------------------------------


def _make_test_with_two_sources():
    a = battleground_db.create_source(
        kind = "external_openai", name = "A", ref = "http://x/v1", external_model = "a"
    )
    b = battleground_db.create_source(
        kind = "external_openai", name = "B", ref = "http://y/v1", external_model = "b"
    )
    test = battleground_db.create_test(
        name = "T",
        slots = [{"source_id": a["id"]}, {"source_id": b["id"]}],
        show_model_cards = False,
        reveal_after_vote = True,
    )
    return test, a, b


class TestPublicRoutes:
    def test_page_requires_valid_token(self, app_client):
        test, _, _ = _make_test_with_two_sources()
        no_token = app_client.get(f"/bg/t/{test['id']}")
        assert no_token.status_code == 404
        bad_token = app_client.get(f"/bg/t/{test['id']}?k=wrong")
        assert bad_token.status_code == 404
        ref = f"t/{test['id']}"
        good = app_client.get(f"/bg/t/{test['id']}?k={battleground_token.sign_battleground_ref(ref)}")
        assert good.status_code == 200
        assert "default-src 'self'" in good.headers["content-security-policy"]

    def test_sharing_kill_switch(self, app_client, monkeypatch):
        test, _, _ = _make_test_with_two_sources()
        token = battleground_token.sign_battleground_ref(f"t/{test['id']}")
        monkeypatch.setattr(battleground, "get_battleground_sharing_enabled", lambda: False)
        response = app_client.get(f"/bg/t/{test['id']}?k={token}")
        assert response.status_code == 404

    def test_config_creates_session_without_leaking_models(self, app_client):
        test, _, _ = _make_test_with_two_sources()
        token = battleground_token.sign_battleground_ref(f"t/{test['id']}")
        response = app_client.get(f"/bg/t/{test['id']}/config?k={token}")
        assert response.status_code == 200
        config = response.json()
        assert config["session_id"]
        assert sorted(config["labels"]) == ["Model A", "Model B"]
        # Blind test: no model identity may leave the server before the vote.
        assert "model_names" not in config

    def test_rate_limit_kicks_in(self, app_client):
        test, _, _ = _make_test_with_two_sources()
        token = battleground_token.sign_battleground_ref(f"t/{test['id']}")
        import utils.preview_rate_limit as rl

        rl.reset()
        codes = []
        for _ in range(25):
            response = app_client.get(f"/bg/t/{test['id']}/config?k={token}")
            codes.append(response.status_code)
        assert 429 in codes
        assert codes[-1] == 429

    def test_feedback_pick_best_does_not_reveal(self, app_client):
        """Votes alone never reveal: identities wait for the explicit finish."""
        test, a, b = _make_test_with_two_sources()
        token = battleground_token.sign_battleground_ref(f"t/{test['id']}")
        session = battleground_db.create_session(
            mode = "public", test_id = test["id"], label_map = {a["id"]: "Model A", b["id"]: "Model B"}
        )
        turn = battleground_db.create_turn(session["id"], "prompt")
        resp_a = battleground_db.create_response(
            turn_id = turn["id"],
            session_id = session["id"],
            source_id = a["id"],
            model_identity = "A · identity",
            label = "Model A",
        )
        resp_b = battleground_db.create_response(
            turn_id = turn["id"],
            session_id = session["id"],
            source_id = b["id"],
            model_identity = "B · identity",
            label = "Model B",
        )
        response = app_client.post(
            f"/bg/t/{test['id']}/feedback?k={token}",
            json = {
                "session_id": session["id"],
                "kind": "pick_best",
                "turn_id": turn["id"],
                "chosen_response_id": resp_a["id"],
            },
        )
        assert response.status_code == 200
        body = response.json()
        # Blind until the tester finishes: no identities on the vote response…
        assert "revealed" not in body
        # …and the turn stays unrevealed.
        assert not battleground_db.get_turn(turn["id"])["revealed"]

    def test_finish_requires_vote_on_every_turn(self, app_client):
        test, a, b = _make_test_with_two_sources()
        token = battleground_token.sign_battleground_ref(f"t/{test['id']}")
        session = battleground_db.create_session(
            mode = "public", test_id = test["id"], label_map = {a["id"]: "Model A", b["id"]: "Model B"}
        )
        turn_1 = battleground_db.create_turn(session["id"], "q1")
        turn_2 = battleground_db.create_turn(session["id"], "q2")
        for turn in (turn_1, turn_2):
            for source, label, identity in ((a, "Model A", "A · identity"), (b, "Model B", "B · identity")):
                battleground_db.create_response(
                    turn_id = turn["id"],
                    session_id = session["id"],
                    source_id = source["id"],
                    model_identity = identity,
                    label = label,
                )

        def _vote(turn, response_id):
            return app_client.post(
                f"/bg/t/{test['id']}/feedback?k={token}",
                json = {
                    "session_id": session["id"],
                    "kind": "pick_best",
                    "turn_id": turn["id"],
                    "chosen_response_id": response_id,
                },
            )

        def _finish():
            return app_client.post(
                f"/bg/t/{test['id']}/finish?k={token}",
                json = {"session_id": session["id"]},
            )

        # No votes yet: nothing may be revealed.
        assert _finish().json() == {"finished": False, "pending_turns": 2}
        # Voting on one of two turns is still not enough.
        assert _vote(turn_1, battleground_db.list_responses_for_turn(turn_1["id"])[0]["id"]).status_code == 200
        assert _finish().json() == {"finished": False, "pending_turns": 1}
        # Last vote → the finish succeeds and reveals every turn at once.
        assert _vote(turn_2, battleground_db.list_responses_for_turn(turn_2["id"])[0]["id"]).status_code == 200
        body = _finish().json()
        assert body["finished"] is True
        assert body["reveals"] == {
            turn_1["id"]: {"Model A": "A · identity", "Model B": "B · identity"},
            turn_2["id"]: {"Model A": "A · identity", "Model B": "B · identity"},
        }
        # The session is stamped finished and every turn counts as revealed.
        assert battleground_db.get_session(session["id"])["finished_at"]
        for turn in (turn_1, turn_2):
            assert battleground_db.get_turn(turn["id"])["revealed"]

    def test_finish_validates_session_and_config(self, app_client):
        test, _, _ = _make_test_with_two_sources()
        token = battleground_token.sign_battleground_ref(f"t/{test['id']}")
        # Foreign / play-mode session ids are rejected.
        foreign = battleground_db.create_session(mode = "play")
        response = app_client.post(
            f"/bg/t/{test['id']}/finish?k={token}",
            json = {"session_id": foreign["id"]},
        )
        assert response.status_code == 400
        # With public model cards there is nothing to reveal: finish is off.
        a = battleground_db.create_source(
            kind = "external_openai", name = "A", ref = "http://x/v1", external_model = "a"
        )
        b = battleground_db.create_source(
            kind = "external_openai", name = "B", ref = "http://x/v1", external_model = "b"
        )
        open_test = battleground_db.create_test(
            name = "Open",
            slots = [{"source_id": a["id"]}, {"source_id": b["id"]}],
            show_model_cards = True,
            reveal_after_vote = True,
        )
        open_session = battleground_db.create_session(
            mode = "public", test_id = open_test["id"], label_map = {a["id"]: "Model A", b["id"]: "Model B"}
        )
        open_token = battleground_token.sign_battleground_ref(f"t/{open_test['id']}")
        response = app_client.post(
            f"/bg/t/{open_test['id']}/finish?k={open_token}",
            json = {"session_id": open_session["id"]},
        )
        assert response.status_code == 400
        assert "not enabled" in response.json()["detail"]

    def test_finished_session_rejects_chat(self, app_client):
        test, a, b = _make_test_with_two_sources()
        token = battleground_token.sign_battleground_ref(f"t/{test['id']}")
        session = battleground_db.create_session(
            mode = "public", test_id = test["id"], label_map = {a["id"]: "Model A", b["id"]: "Model B"}
        )
        turn = battleground_db.create_turn(session["id"], "prompt")
        resp_a = battleground_db.create_response(
            turn_id = turn["id"],
            session_id = session["id"],
            source_id = a["id"],
            model_identity = "A · identity",
            label = "Model A",
        )
        app_client.post(
            f"/bg/t/{test['id']}/feedback?k={token}",
            json = {
                "session_id": session["id"],
                "kind": "pick_best",
                "turn_id": turn["id"],
                "chosen_response_id": resp_a["id"],
            },
        )
        finish = app_client.post(
            f"/bg/t/{test['id']}/finish?k={token}",
            json = {"session_id": session["id"]},
        )
        assert finish.json()["finished"] is True
        # Post-reveal votes would be biased: no more chat on this session.
        response = app_client.post(
            f"/bg/t/{test['id']}/chat?k={token}",
            json = {"session_id": session["id"], "message": "one more?"},
        )
        assert response.status_code == 400
        assert "finished" in response.json()["detail"]

    def test_feedback_validates_ownership(self, app_client):
        test, a, _ = _make_test_with_two_sources()
        token = battleground_token.sign_battleground_ref(f"t/{test['id']}")
        other_session = battleground_db.create_session(mode = "public")
        turn = battleground_db.create_turn(other_session["id"], "x")
        resp = battleground_db.create_response(
            turn_id = turn["id"],
            session_id = other_session["id"],
            source_id = a["id"],
            model_identity = "x",
            label = "Model A",
        )
        response = app_client.post(
            f"/bg/t/{test['id']}/feedback?k={token}",
            json = {
                "session_id": other_session["id"],
                "kind": "rating",
                "turn_id": turn["id"],
                "response_id": resp["id"],
                "rating": "good",
            },
        )
        assert response.status_code == 400

    def test_public_chat_rejects_foreign_session(self, app_client):
        test, _, _ = _make_test_with_two_sources()
        token = battleground_token.sign_battleground_ref(f"t/{test['id']}")
        foreign = battleground_db.create_session(mode = "play")
        response = app_client.post(
            f"/bg/t/{test['id']}/chat?k={token}",
            json = {"session_id": foreign["id"], "message": "hi"},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Report math
# ---------------------------------------------------------------------------


class TestJudgePairOrderNormalization:
    """The second judging order swaps the candidates: its 'a' is the first
    order's 'b'. Verdict letters must be flipped before comparing, else real
    agreement reads as disagreement (and position bias reads as agreement)."""

    @staticmethod
    def _stub_judge(responses):
        """responses: list of raw judge outputs, one per call, in call order."""
        import core.battleground.judge as judge_mod

        calls = {"n": 0, "prompts": []}

        async def fake_call(_target, _system, prompt):
            calls["prompts"].append(prompt)
            out = responses[calls["n"]]
            calls["n"] += 1
            return out

        return judge_mod, calls, fake_call

    def test_agreement_across_orders_picks_the_model(self, monkeypatch):
        import core.battleground.judge as judge_mod

        # Order ab: response_a first, judge picks "a" (response_a wins).
        # Order ba: response_b first, judge picks "b" (response_a wins again).
        responses = [
            '{"winner": "a", "reason": "A is better", "confidence": 0.9}',
            '{"winner": "b", "reason": "B slot holds the better answer", "confidence": 0.8}',
        ]
        judge_mod, calls, fake = self._stub_judge(responses)
        monkeypatch.setattr(judge_mod, "_judge_call", fake)

        verdict = asyncio.run(
            judge_mod.judge_pair(None, "instr", "resp-a", "resp-b")
        )
        assert verdict.winner == "a"
        assert verdict.order_agreement is True
        assert calls["n"] == 2

    def test_position_bias_becomes_a_tie(self, monkeypatch):
        import core.battleground.judge as judge_mod

        # Judge always picks whichever answer is first: "a" in both orders.
        responses = [
            '{"winner": "a", "reason": "first looks better", "confidence": 0.9}',
            '{"winner": "a", "reason": "first looks better again", "confidence": 0.9}',
        ]
        judge_mod, calls, fake = self._stub_judge(responses)
        monkeypatch.setattr(judge_mod, "_judge_call", fake)

        verdict = asyncio.run(
            judge_mod.judge_pair(None, "instr", "resp-a", "resp-b")
        )
        assert verdict.winner == "tie"
        assert verdict.order_agreement is False

    def test_both_orders_tie_is_a_tie(self, monkeypatch):
        import core.battleground.judge as judge_mod

        responses = [
            '{"winner": "tie", "reason": "even", "confidence": 0.7}',
            '{"winner": "tie", "reason": "still even", "confidence": 0.7}',
        ]
        judge_mod, _, fake = self._stub_judge(responses)
        monkeypatch.setattr(judge_mod, "_judge_call", fake)

        verdict = asyncio.run(
            judge_mod.judge_pair(None, "instr", "resp-a", "resp-b")
        )
        assert verdict.winner == "tie"
        assert verdict.order_agreement is True


class TestReportMath:
    def test_wilson_ci_bounds(self):
        # Wilson interval: 0/10 -> [0, ~0.2775]; 10/10 -> [~0.7225, 1].
        low = battleground._wilson_ci(0, 10)
        high = battleground._wilson_ci(10, 10)
        assert low == [0.0, pytest.approx(0.2775, abs = 1e-3)]
        assert high == [pytest.approx(0.7225, abs = 1e-3), 1.0]
        assert battleground._wilson_ci(0, 0) is None
        # Symmetry check: p and 1-p mirror each other.
        five = battleground._wilson_ci(5, 10)
        assert five is not None
        assert five[0] + five[1] == pytest.approx(1.0, abs = 1e-6)

    def test_percentile(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert battleground._percentile(values, 50) == 3.0
        assert battleground._percentile([], 50) is None


# ---------------------------------------------------------------------------
# Legacy playground_* → battleground_* migration
# ---------------------------------------------------------------------------


class TestLegacyTableMigration:
    def _install_legacy_schema(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE playground_sources (
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
            # Legacy judge runs declare test_id NOT NULL; the migration must
            # rebuild this table so standalone auto-eval rows can be added.
            conn.execute(
                """
                CREATE TABLE playground_judge_runs (
                    id TEXT NOT NULL PRIMARY KEY,
                    test_id TEXT NOT NULL,
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
                INSERT INTO playground_sources (id, kind, name, ref, created_at, updated_at)
                VALUES ('pgsrc_old', 'external_openai', 'Old', 'http://x/v1', 't', 't')
                """
            )
            conn.execute(
                """
                INSERT INTO playground_judge_runs (id, test_id, prompt_set_id, judge_source_id, mode, status, created_at)
                VALUES ('pgjudge_old', 'pgtest_old', 'pgset_old', 'pgsrc_old', 'pairwise', 'done', 't')
                """
            )
            conn.commit()
        finally:
            conn.close()

    def test_legacy_tables_renamed_and_preserved(self, tmp_path, monkeypatch):
        db_path = tmp_path / "studio.db"
        self._install_legacy_schema(db_path)
        monkeypatch.setattr("utils.paths.studio_db_path", lambda: db_path, raising = True)
        # battleground_db imported studio_db_path by value; patch its reference too.
        monkeypatch.setattr(battleground_db, "studio_db_path", lambda: db_path)
        battleground_db.reset_schema_cache()
        try:
            conn = battleground_db.get_connection()
            try:
                names = {
                    row["name"]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                assert "battleground_sources" in names
                assert "battleground_judge_runs" in names
                assert "playground_sources" not in names
                assert "playground_judge_runs" not in names
                # Data survived the rename/rebuild.
                source = conn.execute(
                    "SELECT name FROM battleground_sources WHERE id = 'pgsrc_old'"
                ).fetchone()
                assert source is not None and source["name"] == "Old"
                legacy_run = conn.execute(
                    "SELECT test_id FROM battleground_judge_runs WHERE id = 'pgjudge_old'"
                ).fetchone()
                assert legacy_run is not None and legacy_run["test_id"] == "pgtest_old"
            finally:
                conn.close()

            # Standalone judge runs (test_id NULL) are now accepted.
            run = battleground_db.create_judge_run(
                prompt_set_id = "pgset_old",
                judge_source_id = "pgsrc_other",
                mode = "pairwise",
                config = {"source_ids": ["pgsrc_a", "pgsrc_b"]},
            )
            assert run["test_id"] is None
        finally:
            battleground_db.reset_schema_cache()


# ---------------------------------------------------------------------------
# Standalone judge runs (auto eval decoupled from A/B tests)
# ---------------------------------------------------------------------------


class TestStandaloneJudgeRuns:
    def test_pairwise_requires_exactly_two_sources(self, app_client):
        a = _make_source(name = "A")
        prompt_set = battleground_db.create_prompt_set(name = "P")
        battleground_db.add_prompts_bulk(prompt_set["id"], [{"prompt": "hi"}])
        judge = _make_source(name = "J")
        response = app_client.post(
            "/api/battleground/judge/runs",
            json = {
                "source_ids": [a["id"]],
                "prompt_set_id": prompt_set["id"],
                "judge_source_id": judge["id"],
                "mode": "pairwise",
            },
        )
        assert response.status_code == 400

    def test_judge_cannot_be_participant(self, app_client):
        a = _make_source(name = "A")
        b = _make_source(name = "B")
        prompt_set = battleground_db.create_prompt_set(name = "P")
        response = app_client.post(
            "/api/battleground/judge/runs",
            json = {
                "source_ids": [a["id"], b["id"]],
                "prompt_set_id": prompt_set["id"],
                "judge_source_id": a["id"],
                "mode": "pairwise",
            },
        )
        assert response.status_code == 400

    def test_unknown_participant_rejected(self, app_client):
        b = _make_source(name = "B")
        judge = _make_source(name = "J")
        prompt_set = battleground_db.create_prompt_set(name = "P")
        response = app_client.post(
            "/api/battleground/judge/runs",
            json = {
                "source_ids": ["pgsrc_missing", b["id"]],
                "prompt_set_id": prompt_set["id"],
                "judge_source_id": judge["id"],
                "mode": "pairwise",
            },
        )
        assert response.status_code == 404

    def test_run_created_with_participants(self, app_client):
        a = _make_source(name = "A")
        b = _make_source(name = "B")
        judge = _make_source(name = "J")
        prompt_set = battleground_db.create_prompt_set(name = "P")
        battleground_db.add_prompts_bulk(prompt_set["id"], [{"prompt": "hi"}])
        response = app_client.post(
            "/api/battleground/judge/runs",
            json = {
                "source_ids": [a["id"], b["id"]],
                "prompt_set_id": prompt_set["id"],
                "judge_source_id": judge["id"],
                "mode": "pairwise",
                "max_prompts": 5,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["test_id"] is None
        assert body["config"]["source_ids"] == [a["id"], b["id"]]
        runs = app_client.get("/api/battleground/judge/runs").json()
        entry = next(r for r in runs if r["id"] == body["id"])
        assert entry["source_names"] == ["A", "B"]
        assert entry["test_name"] is None


# ---------------------------------------------------------------------------
# Prompt set editing
# ---------------------------------------------------------------------------


class TestPromptEditing:
    def test_rename_prompt_set(self, app_client):
        prompt_set = battleground_db.create_prompt_set(name = "Old", description = "d")
        response = app_client.put(
            f"/api/battleground/prompt-sets/{prompt_set['id']}",
            json = {"name": "New"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "New"
        assert response.json()["description"] == "d"

    def test_clear_description(self, app_client):
        prompt_set = battleground_db.create_prompt_set(name = "S", description = "d")
        response = app_client.put(
            f"/api/battleground/prompt-sets/{prompt_set['id']}",
            json = {"description": None},
        )
        assert response.status_code == 200
        assert response.json()["description"] is None

    def test_update_prompt_text_and_reference(self, app_client):
        prompt_set = battleground_db.create_prompt_set(name = "S")
        row = battleground_db.add_prompt(prompt_set["id"], prompt = "q", reference_answer = "a")
        response = app_client.put(
            f"/api/battleground/prompt-sets/{prompt_set['id']}/prompts/{row['id']}",
            json = {"prompt": "q2", "reference_answer": None},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prompt"] == "q2"
        assert body["reference_answer"] is None

    def test_update_prompt_missing(self, app_client):
        prompt_set = battleground_db.create_prompt_set(name = "S")
        response = app_client.put(
            f"/api/battleground/prompt-sets/{prompt_set['id']}/prompts/pgprm_missing",
            json = {"prompt": "x"},
        )
        assert response.status_code == 404
