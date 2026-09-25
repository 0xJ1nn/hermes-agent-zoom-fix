"""Regression: a profile quarantined out of the Nous grant must not latch dead.

Nous Portal refresh tokens are single-use and the Portal revokes the whole session
chain when one is redeemed twice, so a profile that loses the race ends up with a
``providers.nous`` block whose token keys were popped by ``_quarantine_nous_oauth_state``
while the block itself survives (the routing metadata and ``last_auth_error`` are still
worth persisting).

Before the fix, ``_load_provider_state_with_source`` treated "block present" as
"credential available", so the dead husk shadowed the global-root grant the
``read_credential_pool``/root fallback is supposed to reach. The profile then failed
forever with "No access token found for Nous Portal login" and demanded its own manual
re-login, even though a valid login was one directory up -- which is what made every
profile and workspace look permanently expired.

Real imports, real temp HERMES_HOME root + named profile, real auth.json I/O.
"""
from __future__ import annotations

import base64
import json
import time

import pytest


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(epoch))


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _jwt(exp: float, scope: str = "inference:invoke") -> str:
    """Unsigned JWT with the claims the invoke-JWT gate reads (scope + exp)."""
    head = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps({"scope": scope, "exp": int(exp), "aud": "hermes"}).encode())
    return f"{head}.{body}.sig"


@pytest.fixture
def nous_fleet(tmp_path, monkeypatch):
    """Root holding a LIVE Nous grant; profile 'appdev' holding a quarantined husk.

    The profile's grant was spent and rejected by the Portal, so it is exactly the
    post-quarantine shape: routing metadata + ``last_auth_error``, no token material.
    """
    root = tmp_path / "hermes-root"
    prof = root / "profiles" / "appdev"
    prof.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]

    now = time.time()
    live = {
        "client_id": "hermes-cli",
        "portal_base_url": "https://portal.example",
        "inference_base_url": "https://inference.example/v1",
        "access_token": _jwt(now - 60),
        "refresh_token": "RT-live",
        "agent_key": _jwt(now - 60),
        "agent_key_expires_at": _iso(now - 60),
        "agent_key_expires_in": 3274,
        "scope": "inference:invoke",
        "token_type": "Bearer",
        "obtained_at": _iso(now - 3334),
        "expires_at": _iso(now - 60),
        "expires_in": 3274,
        "tls": {"insecure": False, "ca_bundle": None},
    }
    (root / "auth.json").write_text(json.dumps({
        "version": 1, "active_provider": "nous",
        "providers": {"nous": live},
        "credential_pool": {"nous": [{
            "id": "e9e2fd", "label": "acct", "auth_type": "oauth", "priority": 0,
            "source": "device_code", "access_token": _jwt(now - 60), "refresh_token": "RT-live",
            "agent_key": _jwt(now - 60), "agent_key_expires_at": _iso(now - 60),
            "agent_key_expires_in": 3274, "expires_at": _iso(now - 60),
            "client_id": "hermes-cli", "portal_base_url": "https://portal.example",
            "token_type": "Bearer", "scope": "inference:invoke",
        }]},
    }))

    # The husk: quarantine popped every token key but kept the block.
    (prof / "auth.json").write_text(json.dumps({
        "version": 1, "active_provider": "nous",
        "providers": {"nous": {
            "client_id": "hermes-cli",
            "portal_base_url": "https://portal.example",
            "inference_base_url": "https://inference.example/v1",
            "scope": "inference:invoke",
            "token_type": "Bearer",
            "tls": {"insecure": False, "ca_bundle": None},
            "last_auth_error": {
                "provider": "nous", "code": "invalid_grant",
                "message": "Refresh session has been revoked",
                "reason": "runtime_access_refresh_failure", "relogin_required": True,
                "at": _iso(now - 600),
            },
        }},
        "credential_pool": {"nous": []},
    }))

    server = {"n": 0}

    def fake_refresh_access_token(**kwargs):
        """Stand in for the Portal token POST (the real seam is httpx, not urllib)."""
        server["n"] += 1
        n = server["n"]
        assert kwargs["refresh_token"] == "RT-live", (
            "a waiter must re-read the token authority and never re-POST a spent token")
        return {
            "access_token": _jwt(time.time() + 3274), "refresh_token": f"RT-new{n}",
            "agent_key": _jwt(time.time() + 3274), "agent_key_expires_in": 3274, "expires_in": 3274,
            "token_type": "Bearer", "scope": "inference:invoke",
        }

    import hermes_cli.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_refresh_access_token", fake_refresh_access_token)

    def use(home):
        monkeypatch.setenv("HERMES_HOME", str(home))
        hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
        import hermes_cli.auth as auth_mod
        auth_mod._global_auth_store_cache = None
        auth_mod._oauth_heal_clean_marks.clear()
        auth_mod.invalidate_nous_auth_status_cache()

    return {"root": root, "prof": prof, "server": server, "use": use}


def test_quarantined_profile_block_does_not_shadow_the_root_grant(nous_fleet):
    """The husk must read as absent so the global-root fallback engages."""
    from hermes_cli.auth import _load_provider_state_with_source, _load_auth_store

    nous_fleet["use"](nous_fleet["prof"])
    state, source = _load_provider_state_with_source(_load_auth_store(), "nous")
    assert state is not None, "profile must fall back to the root grant, not report logged-out"
    assert state["refresh_token"] == "RT-live"
    assert _same(source, nous_fleet["root"] / "auth.json")


def test_profile_recovers_its_nous_login_without_a_manual_re_login(nous_fleet):
    """End-to-end: the profile resolves a usable credential from the healthy root."""
    from hermes_cli.auth_nous import get_nous_auth_status_local

    nous_fleet["use"](nous_fleet["prof"])
    status = get_nous_auth_status_local()
    assert status["logged_in"] is True
    assert not status.get("relogin_required"), (
        "a stale quarantine marker must not keep reporting 'log in again' once the root "
        "grant is reachable again")


def test_root_and_profile_agree_after_the_profile_refreshes(nous_fleet):
    """The rotation lands in the root (the authority) and the profile stops being a husk."""
    from hermes_cli.auth_nous import resolve_nous_runtime_credentials

    nous_fleet["use"](nous_fleet["prof"])
    creds = resolve_nous_runtime_credentials(timeout_seconds=15.0)
    assert creds["api_key"]

    root_state = json.loads((nous_fleet["root"] / "auth.json").read_text())["providers"]["nous"]
    assert root_state["refresh_token"] == "RT-new1"

    # Second tick in the same profile must keep working (no permanent latch), and must NOT
    # spend the freshly-rotated single-use token again.
    nous_fleet["use"](nous_fleet["prof"])
    creds2 = resolve_nous_runtime_credentials(timeout_seconds=15.0)
    assert creds2["api_key"] == creds["api_key"]
    assert nous_fleet["server"]["n"] == 1, "the rotated grant was re-POSTed on a healthy token"


def test_stale_pool_row_adopts_the_block_pair_instead_of_re_posting_it(nous_fleet):
    """A pool row left behind by an earlier rotation must not re-POST a spent token.

    This is the on-disk shape behind the live failure: ``providers.nous`` had rotated
    (expires_at 15:45) while ``credential_pool.nous[0]`` still held the previous,
    already-spent pair (expires_at 15:15) in the SAME auth.json. A pool refresh that
    trusts its own row POSTs a dead token, the Portal answers invalid_grant /
    "Refresh session has been revoked", and the whole login gets quarantined -- out of
    a file that also contained a perfectly valid grant.

    ``load_pool()`` happens to reseed a root row from the block, so this drives
    ``try_refresh_current()`` on a pool built directly from the stale row: that is the
    narrow window where the in-lock re-read is the only thing standing between a stale
    row and a revoked session chain.
    """
    from agent.credential_pool import CredentialPool, PooledCredential
    import agent.credential_pool as pool_mod

    stale_row = {
        "id": "e9e2fd", "label": "acct", "auth_type": "oauth", "priority": 0,
        "source": "device_code",
        "access_token": _jwt(time.time() - 120), "refresh_token": "RT-spent",
        "agent_key": _jwt(time.time() - 120), "agent_key_expires_at": _iso(time.time() - 60),
        "agent_key_expires_in": 3274, "expires_at": _iso(time.time() - 60),
        "client_id": "hermes-cli", "portal_base_url": "https://portal.example",
        "token_type": "Bearer", "scope": "inference:invoke",
        "tls": {"insecure": False, "ca_bundle": None}, "last_status": None,
    }
    nous_fleet["use"](nous_fleet["root"])
    pool = CredentialPool("nous", [PooledCredential.from_dict("nous", stale_row)])
    assert pool.select().refresh_token == "RT-spent", "fixture: the row must start out stale"

    # "nous" must be treated as single-use, or there is no cross-process flock and no
    # in-lock re-read at all.
    assert "nous" in pool_mod._SINGLE_USE_REFRESH_PROVIDERS, (
        "the Nous pool refresh is unsynchronized; a stale row can spend a live token")

    import hermes_cli.auth as auth_mod
    posted = []
    original = auth_mod._refresh_access_token

    def spy(**kwargs):
        posted.append(kwargs["refresh_token"])
        return original(**kwargs)

    auth_mod._refresh_access_token = spy
    try:
        pool.try_refresh_current()
    finally:
        auth_mod._refresh_access_token = original

    assert "RT-spent" not in posted, (
        "the pool re-POSTed an already-spent single-use refresh token")
    assert nous_fleet["server"]["n"] == 0, (
        "a fresher grant was on disk, so nothing should have been POSTed at all")


def test_guest_state_is_never_mistaken_for_a_husk(nous_fleet):
    """The free tier carries an anon_ credential and no token pair: still resolvable."""
    from hermes_cli.auth import _resolvable_provider_state
    from hermes_cli.anon_auth import ANON_AUTH_METHOD

    guest = {"auth_method": ANON_AUTH_METHOD, "anon_token": "anon_abc",
             "portal_base_url": "https://portal.example"}
    assert _resolvable_provider_state({"providers": {"nous": guest}}, "nous") is not None


def test_api_key_block_is_never_mistaken_for_a_husk(nous_fleet):
    from hermes_cli.auth import _resolvable_provider_state
    key = {"api_key": "sk-some-key"}
    assert _resolvable_provider_state({"providers": {"nous": key}}, "nous") is not None


def test_genuinely_absent_block_still_reports_logged_out(nous_fleet):
    from hermes_cli.auth import _resolvable_provider_state
    assert _resolvable_provider_state({"providers": {}}, "nous") is None
    assert _resolvable_provider_state(
        {"providers": {"nous": {"client_id": "hermes-cli"}}}, "nous") is None


def _same(a, b) -> bool:
    from pathlib import Path
    return Path(a).resolve() == Path(b).resolve()
