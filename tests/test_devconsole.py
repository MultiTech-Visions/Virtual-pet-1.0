"""The Dev tab: a terminal on the robot, and the gate in front of it."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from festival_pet.devconsole import DevConsole


def _console(tmp_path):
    return DevConsole(tmp_path, workdir=tmp_path)


def test_the_console_is_off_until_somebody_sets_a_passphrase(tmp_path):
    d = _console(tmp_path)
    assert not d.enabled and not d.allows("") and not d.allows(None)
    with pytest.raises(ValueError):
        d.enable("short")  # it is a shell: a two-character passphrase is not a gate
    assert not d.enabled
    d.enable("open sesame")
    assert d.enabled and d.allows("open sesame")
    assert not d.allows("open sesam") and not d.allows("") and not d.allows(None)
    d.disable()
    assert not d.enabled and not d.allows("open sesame")  # and the passphrase does not linger


def test_the_api_key_is_stored_tightly_and_never_handed_back(tmp_path):
    d = _console(tmp_path)
    assert d.key() is None and d.status()["key_set"] is False
    with pytest.raises(ValueError):
        d.set_key("hunter2")  # not a key
    d.set_key("sk-ant-secret-value-1234")
    st = d.status()
    assert st["key_set"] and st["key_tail"] == "1234"
    assert "secret" not in json.dumps(st)  # the page is never told the key itself
    assert oct(d.key_file.stat().st_mode)[-3:] == "600"
    assert d.key() == "sk-ant-secret-value-1234"
    d.clear_key()
    assert d.key() is None


def test_the_terminal_runs_a_command_and_the_key_is_in_its_environment(tmp_path):
    d = _console(tmp_path)
    d.set_key("sk-ant-test-key-abcd")
    d.start("echo HELLO-$ANTHROPIC_API_KEY; exit", cols=80, rows=24)
    assert d.running
    out = b""
    while True:
        chunk = d.read()
        if not chunk:
            break
        out += chunk
    assert b"HELLO-sk-ant-test-key-abcd" in out
    d.stop()
    assert not d.running and d.fd is None


def test_the_terminal_takes_input_and_a_window_size(tmp_path):
    d = _console(tmp_path)
    d.start("", cols=90, rows=20)
    d.resize(120, 40)
    d.write(b"echo COLS-IS-$COLUMNS\n")
    out = b""
    for _ in range(40):
        out += d.read()
        if b"COLS-IS-120" in out:
            break
    assert b"COLS-IS-120" in out
    d.stop()


def _pet_app():
    import sys

    sys.path.insert(0, "tests")
    from test_web import _pet

    from festival_pet.main import install_routes

    pet = _pet()
    app = FastAPI()
    install_routes(app, pet)
    return pet, TestClient(app)


def test_the_websocket_refuses_without_the_passphrase_and_works_with_it():
    pet, c = _pet_app()
    try:
        assert c.get("/api/dev").json()["enabled"] is False
        # It says why and then hangs up — a blank failed handshake in a terminal tells nobody anything.
        def refused(url):
            with c.websocket_connect(url) as ws:
                why = ws.receive_text()
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_text()
            return why

        assert "off" in refused("/ws/dev?token=anything")  # off: no terminal for anybody, passphrase or not
        assert not pet.dev.running
        assert c.post("/api/dev", json={"cmd": "enable", "value": "abc"}).status_code == 400
        assert c.post("/api/dev", json={"cmd": "enable", "value": "letmeinplease"}).json()["enabled"]
        assert "passphrase" in refused("/ws/dev?token=letmein")  # close, but no
        assert not pet.dev.running  # ...and a refused connection never starts a shell
        with c.websocket_connect("/ws/dev?token=letmeinplease&cols=80&rows=24&cmd=echo+MARCO-POLO") as ws:
            out = b""
            for _ in range(20):
                msg = ws.receive()
                out += msg.get("bytes") or (msg.get("text") or "").encode()
                if b"MARCO-POLO" in out:
                    break
            assert b"MARCO-POLO" in out
        assert c.post("/api/dev", json={"cmd": "disable", "value": True}).json()["enabled"] is False
        assert not pet.dev.running  # switching it off kills the session, it does not leave it behind
    finally:
        pet.dev.disable()
        pet.stop()


def test_the_vendored_terminal_is_served_and_nothing_else_is():
    pet, c = _pet_app()
    try:
        assert c.get("/static/vendor/xterm.js").status_code == 200
        assert c.get("/static/vendor/xterm.css").headers["content-type"].startswith("text/css")
        for sneaky in ("main.py", "../main.py", "../../pyproject.toml"):
            assert c.get(f"/static/vendor/{sneaky}").status_code == 404
    finally:
        pet.stop()


def test_petctl_turns_the_mind_blob_into_one_readable_line():
    """The line a session on the robot reads before it theorises about anything."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("petctl", Path(__file__).resolve().parents[1] / "scripts" / "petctl.py")
    petctl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(petctl)

    pet, c = _pet_app()
    try:
        line = petctl.summarise(c.get("/api/mind").json())
        assert "IDLE" in line and "no face" in line and "spots none" in line
        # values are JSON, as the page sends them: numbers stay numbers, bare words stay strings
        assert petctl._value("0.5") == 0.5 and petctl._value("true") is True
        assert petctl._value("left") == "left" and petctl._value(None) is True
    finally:
        pet.stop()


def test_a_session_that_dies_on_its_first_line_says_why(tmp_path):
    """The Dev tab printed nothing at all when a command failed — just "[session closed]" — so an npm
    EACCES or a missing binary was indistinguishable from the console being off."""
    import time

    from festival_pet.devconsole import DevConsole

    d = DevConsole(tmp_path, tmp_path)
    d.start("exit 42")
    end = time.monotonic() + 5.0
    while d.read() and time.monotonic() < end:
        pass
    while d.running and time.monotonic() < end:
        time.sleep(0.02)
    assert d.ended() == "[the session exited with status 42 — the last lines above say why]"

    d.start("true")
    end = time.monotonic() + 5.0
    while d.read() and time.monotonic() < end:
        pass
    while d.running and time.monotonic() < end:
        time.sleep(0.02)
    assert d.ended() == "[the session finished]"
    d.stop()
