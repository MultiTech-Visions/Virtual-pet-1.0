import random

import numpy as np

from festival_pet.behavior import Behavior, FaceObs, Observation, Timers
from festival_pet.memory import FaceMemory


def _brain(awake=True):
    mem = FaceMemory("/nonexistent/never-written.json")
    b = Behavior(mem, timers=Timers(), rng=random.Random(0))
    b.start(0.0, awake=awake)
    return b, mem


def _run(b, obs_fn, t0, t1, dt=0.05):
    acts = []
    t = t0
    while t < t1:
        acts += b.tick(obs_fn(t), t, dt)
        t += dt
    return acts


def test_stranger_greeting_then_search_then_idle():
    b, _ = _brain()
    face = FaceObs(track_id=1, yaw_deg=10.0, pitch_deg=-2.0, area_frac=0.05, person=None, similarity=0.0)
    acts = _run(b, lambda t: Observation(face=face), 0.0, 3.0)
    assert b.state == "ENGAGED"
    assert b.gaze == (10.0, -2.0)
    assert any(a.kind == "sound" and a.name == "hello_new" for a in acts)
    acts = _run(b, lambda t: Observation(), 3.0, 7.0)
    assert any(a.kind == "gesture" and a.name == "search" for a in acts)
    assert b.state == "SEARCHING"
    _run(b, lambda t: Observation(), 7.0, 13.0)
    assert b.state == "IDLE" and b.gaze is None


def test_friend_gets_friend_greeting_and_attention():
    b, mem = _brain()
    p = mem.enroll(np.ones(4), now=-1000.0)
    mem.sighted(p, now=-500.0)  # second encounter -> friend
    assert p.tier() == "friend"
    face = FaceObs(1, 0.0, 0.0, 0.1, p, 0.8)
    acts = _run(b, lambda t: Observation(face=face), 0.0, 5.0)
    names = [a.name for a in acts if a.kind == "sound"]
    assert "hello_friend" in names
    assert any(a.kind == "move" for a in acts)  # library move for known friends
    assert p.attention_seconds > 4.0
    assert p.encounters == 3


def test_pickup_purr_shake_dizzy_and_setdown():
    b, _ = _brain()
    acts = _run(b, lambda t: Observation(held=True), 0.0, 8.0)
    assert b.state == "HELD"
    names = [a.name for a in acts]
    assert "surprised" in names and "startle" in names
    assert any(n in ("purr", "content") for n in names)
    acts = _run(b, lambda t: Observation(held=True, shaken=(t < 8.1)), 8.0, 9.0)
    assert any(a.name == "dizzy" for a in acts)
    acts = _run(b, lambda t: Observation(held=False), 9.0, 10.0)
    assert b.state == "IDLE"
    assert any(a.name == "shake_off" for a in acts)


def test_lonely_then_falls_asleep_then_touch_wakes():
    b, _ = _brain()
    t = Timers()
    acts = _run(b, lambda t_: Observation(), 0.0, t.lonely_after + 5, dt=0.5)
    assert any(a.name == "lonely" for a in acts)
    acts = _run(b, lambda t_: Observation(), t.lonely_after + 5, t.sleep_after + 5, dt=0.5)
    assert b.state == "SLEEPING"
    assert any(a.kind == "sleep" for a in acts)
    acts = _run(b, lambda t_: Observation(touched=(t_ < t.sleep_after + 5.2)), t.sleep_after + 5, t.sleep_after + 9, dt=0.1)
    assert any(a.kind == "wake" for a in acts)
    assert b.state == "IDLE"


def test_sleeping_wakes_on_persistent_face_only():
    b, _ = _brain(awake=False)
    near = FaceObs(1, 0.0, 0.0, 0.05, None, 0.0)
    _run(b, lambda t: Observation(face=near), 0.0, 1.0)
    assert b.state == "SLEEPING"
    _run(b, lambda t: Observation(face=near), 1.0, 2.0)
    assert b.state == "WAKING"


def test_touch_while_engaged_counts_as_pet():
    b, mem = _brain()
    p = mem.enroll(np.ones(4), now=0.0)
    face = FaceObs(1, 0.0, 0.0, 0.1, p, 0.9)
    _run(b, lambda t: Observation(face=face), 0.0, 1.0)
    _run(b, lambda t: Observation(face=face, touched=(t < 1.1)), 1.0, 2.0)
    assert p.pets == 1
