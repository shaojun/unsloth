# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model Playground backend tests: storage, tokens, and route-level security
for the public /pg tester pages.

Exercises the route layer with the real routers while stubbing the proxy layer
(the mock OpenAI server in playground_mock_openai.py covers the proxy itself
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

import routes.playground as playground
import utils.playground_token as playground_token
from storage import playground_db

_TEST_SECRET = b"unit-test-playground-secret-0123456789abcdef"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point studio.db at a temp file and reset the schema cache."""
    db_path = tmp_path / "studio.db"
    monkeypatch.setattr("utils.paths.studio_db_path", lambda: db_path, raising = True)
    # playground_db imported studio_db_path by value; patch its reference too.
    monkeypatch.setattr(playground_db, "studio_db_path", lambda: db_path)
    playground_db.reset_schema_cache()
    yield db_path
    playground_db.reset_schema_cache()


@pytest.fixture
def app_client(isolated_db, monkeypatch):
    """FastAPI app with the playground routers mounted, auth stubbed open."""
    monkeypatch.setattr(playground_token, "get_or_create_preview_link_secret", lambda: _TEST_SECRET)
    import utils.preview_rate_limit as _rl

    _rl.reset()
    monkeypatch.setattr(playground, "get_playground_sharing_enabled", lambda: True)

    app = FastAPI()
    app.include_router(playground.router, prefix = "/api/playground")
    app.include_router(playground.public_router, prefix = "/pg")
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
    return playground_db.create_source(kind = kind, name = name, ref = ref, external_model = "m")


# ---------------------------------------------------------------------------
# Token / security
# ---------------------------------------------------------------------------


class TestPlaygroundTokens:
    def test_sign_verify_roundtrip(self):
        token = playground_token.sign_playground_ref("t/abc")
        assert playground_token.verify_playground_ref("t/abc", token)

    def test_wrong_ref_rejected(self):
        token = playground_token.sign_playground_ref("t/abc")
        assert not playground_token.verify_playground_ref("t/other", token)

    def test_missing_token_rejected(self):
        assert not playground_token.verify_playground_ref("t/abc", None)

    def test_not_preview_token_compatible(self, monkeypatch):
        """Preview tokens must not unlock playground pages (domain separation)."""
        import utils.preview_token as preview_token

        monkeypatch.setattr(
            preview_token, "get_or_create_preview_link_secret", lambda: _TEST_SECRET
        )
        preview_token_value = preview_token.sign_preview_ref("t/abc")
        assert not playground_token.verify_playground_ref("t/abc", preview_token_value)

    def test_garbage_token_rejected(self):
        assert not playground_token.verify_playground_ref("t/abc", "nonsense")
        assert not playground_token.verify_playground_ref("t/abc", "üñïçödé")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestPlaygroundStorage:
    def test_source_crud(self):
        source = playground_db.create_source(kind = "external_openai", name = "S", ref = "http://x/v1")
        assert playground_db.get_source(source["id"])["name"] == "S"
        playground_db.update_source(source["id"], name = "S2")
        assert playground_db.get_source(source["id"])["name"] == "S2"
        assert playground_db.delete_source(source["id"])
        assert playground_db.get_source(source["id"]) is None

    def test_unknown_source_kind_rejected(self):
        with pytest.raises(ValueError):
            playground_db.create_source(kind = "local_dir", name = "S", ref = "/tmp/x")

    def test_feedback_roundtrip(self):
        source = _make_source()
        session = playground_db.create_session(mode = "play", source_id = source["id"])
        turn = playground_db.create_turn(session["id"], "hi")
        response = playground_db.create_response(
            turn_id = turn["id"],
            session_id = session["id"],
            source_id = source["id"],
            model_identity = "x",
        )
        playground_db.create_feedback(
            session_id = session["id"],
            kind = "rating",
            turn_id = turn["id"],
            response_id = response["id"],
            rating = "good",
            tags = ["correct"],
        )
        rows = playground_db.list_feedback(session_id = session["id"])
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

        response = app_client.get("/api/playground/trained-models")
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

        models = app_client.get("/api/playground/trained-models").json()["models"]
        assert models[0]["artifact"] == "adapter"
        assert models[0]["name"] == "org/model"


# ---------------------------------------------------------------------------
# Routes: public blind-test guarantees
# ---------------------------------------------------------------------------


def _make_test_with_two_sources():
    a = playground_db.create_source(
        kind = "external_openai", name = "A", ref = "http://x/v1", external_model = "a"
    )
    b = playground_db.create_source(
        kind = "external_openai", name = "B", ref = "http://y/v1", external_model = "b"
    )
    test = playground_db.create_test(
        name = "T",
        slots = [{"source_id": a["id"]}, {"source_id": b["id"]}],
        show_model_cards = False,
        reveal_after_vote = True,
    )
    return test, a, b


class TestPublicRoutes:
    def test_page_requires_valid_token(self, app_client):
        test, _, _ = _make_test_with_two_sources()
        no_token = app_client.get(f"/pg/t/{test['id']}")
        assert no_token.status_code == 404
        bad_token = app_client.get(f"/pg/t/{test['id']}?k=wrong")
        assert bad_token.status_code == 404
        ref = f"t/{test['id']}"
        good = app_client.get(f"/pg/t/{test['id']}?k={playground_token.sign_playground_ref(ref)}")
        assert good.status_code == 200
        assert "default-src 'self'" in good.headers["content-security-policy"]

    def test_sharing_kill_switch(self, app_client, monkeypatch):
        test, _, _ = _make_test_with_two_sources()
        token = playground_token.sign_playground_ref(f"t/{test['id']}")
        monkeypatch.setattr(playground, "get_playground_sharing_enabled", lambda: False)
        response = app_client.get(f"/pg/t/{test['id']}?k={token}")
        assert response.status_code == 404

    def test_config_creates_session_without_leaking_models(self, app_client):
        test, _, _ = _make_test_with_two_sources()
        token = playground_token.sign_playground_ref(f"t/{test['id']}")
        response = app_client.get(f"/pg/t/{test['id']}/config?k={token}")
        assert response.status_code == 200
        config = response.json()
        assert config["session_id"]
        assert sorted(config["labels"]) == ["Model A", "Model B"]
        # Blind test: no model identity may leave the server before the vote.
        assert "model_names" not in config

    def test_rate_limit_kicks_in(self, app_client):
        test, _, _ = _make_test_with_two_sources()
        token = playground_token.sign_playground_ref(f"t/{test['id']}")
        import utils.preview_rate_limit as rl

        rl.reset()
        codes = []
        for _ in range(25):
            response = app_client.get(f"/pg/t/{test['id']}/config?k={token}")
            codes.append(response.status_code)
        assert 429 in codes
        assert codes[-1] == 429

    def test_feedback_pick_best_reveals(self, app_client):
        test, a, b = _make_test_with_two_sources()
        token = playground_token.sign_playground_ref(f"t/{test['id']}")
        session = playground_db.create_session(
            mode = "public", test_id = test["id"], label_map = {a["id"]: "Model A", b["id"]: "Model B"}
        )
        turn = playground_db.create_turn(session["id"], "prompt")
        resp_a = playground_db.create_response(
            turn_id = turn["id"],
            session_id = session["id"],
            source_id = a["id"],
            model_identity = "A · identity",
            label = "Model A",
        )
        resp_b = playground_db.create_response(
            turn_id = turn["id"],
            session_id = session["id"],
            source_id = b["id"],
            model_identity = "B · identity",
            label = "Model B",
        )
        response = app_client.post(
            f"/pg/t/{test['id']}/feedback?k={token}",
            json = {
                "session_id": session["id"],
                "kind": "pick_best",
                "turn_id": turn["id"],
                "chosen_response_id": resp_a["id"],
            },
        )
        assert response.status_code == 200
        body = response.json()
        # Reveal-after-vote: identities arrive with the vote confirmation.
        assert body["revealed"] == {"Model A": "A · identity", "Model B": "B · identity"}

    def test_feedback_validates_ownership(self, app_client):
        test, a, _ = _make_test_with_two_sources()
        token = playground_token.sign_playground_ref(f"t/{test['id']}")
        other_session = playground_db.create_session(mode = "public")
        turn = playground_db.create_turn(other_session["id"], "x")
        resp = playground_db.create_response(
            turn_id = turn["id"],
            session_id = other_session["id"],
            source_id = a["id"],
            model_identity = "x",
            label = "Model A",
        )
        response = app_client.post(
            f"/pg/t/{test['id']}/feedback?k={token}",
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
        token = playground_token.sign_playground_ref(f"t/{test['id']}")
        foreign = playground_db.create_session(mode = "play")
        response = app_client.post(
            f"/pg/t/{test['id']}/chat?k={token}",
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
        import core.playground.judge as judge_mod

        calls = {"n": 0, "prompts": []}

        async def fake_call(_target, _system, prompt):
            calls["prompts"].append(prompt)
            out = responses[calls["n"]]
            calls["n"] += 1
            return out

        return judge_mod, calls, fake_call

    def test_agreement_across_orders_picks_the_model(self, monkeypatch):
        import core.playground.judge as judge_mod

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
        import core.playground.judge as judge_mod

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
        import core.playground.judge as judge_mod

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
        low = playground._wilson_ci(0, 10)
        high = playground._wilson_ci(10, 10)
        assert low == [0.0, pytest.approx(0.2775, abs = 1e-3)]
        assert high == [pytest.approx(0.7225, abs = 1e-3), 1.0]
        assert playground._wilson_ci(0, 0) is None
        # Symmetry check: p and 1-p mirror each other.
        five = playground._wilson_ci(5, 10)
        assert five is not None
        assert five[0] + five[1] == pytest.approx(1.0, abs = 1e-6)

    def test_percentile(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert playground._percentile(values, 50) == 3.0
        assert playground._percentile([], 50) is None
