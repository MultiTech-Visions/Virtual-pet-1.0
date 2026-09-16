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


def test_ear_touch_while_engaged_is_a_tickle_not_a_pet():
    b, mem = _brain()
    p = mem.enroll(np.ones(4), now=0.0)
    face = FaceObs(1, 0.0, 0.0, 0.1, p, 0.9)
    _run(b, lambda t: Observation(face=face), 0.0, 1.0)
    acts = _run(b, lambda t: Observation(face=face, touched=(t < 1.1), touched_side=0), 1.0, 2.0)
    assert any(a.name == "flinch:-" for a in acts)
    assert p.pets == 0


def test_name_call_perks_and_turns_toward_voice():
    b, _ = _brain()
    acts = _run(b, lambda t: Observation(name_heard=(t < 0.05), voice_yaw_deg=35.0), 0.0, 0.5)
    assert any(a.name == "name" for a in acts)
    assert b.state == "SEARCHING" and b.gaze == (35.0, 0.0)


def test_dance_trick_escalates():
    b, _ = _brain()
    face = FaceObs(1, 0.0, 0.0, 0.05, None, 0.0)
    _run(b, lambda t: Observation(face=face), 0.0, 2.0)
    acts = _run(b, lambda t: Observation(face=face, command=("dance" if t < 2.05 else None)), 2.0, 3.0)
    assert any(a.kind == "groove" and a.name == "0.90" for a in acts)
    assert not any(a.kind == "move" for a in acts)
    acts = _run(b, lambda t: Observation(face=face, command=("dance" if t < 6.05 else None)), 6.0, 7.0)
    assert any(a.kind == "move" and a.name.startswith("dance") for a in acts)


def test_command_ignored_when_not_addressed():
    b, _ = _brain()
    acts = _run(b, lambda t: Observation(command=("dance" if t < 0.05 else None)), 0.0, 1.0)
    assert not any(a.kind in ("groove", "move") for a in acts)


def test_scratch_is_ticklish_and_counts_as_pet():
    b, mem = _brain()
    p = mem.enroll(np.ones(4), now=0.0)
    face = FaceObs(1, 0.0, 0.0, 0.1, p, 0.9)
    _run(b, lambda t: Observation(face=face), 0.0, 1.0)
    acts = _run(b, lambda t: Observation(face=face, scratched=(t < 1.05)), 1.0, 2.0)
    assert any(a.name == "ticklish" for a in acts) and p.pets == 1


def test_music_grooves_after_settling_and_sings():
    b, _ = _brain()
    acts = _run(b, lambda t: Observation(music_bpm=120.0, music_confidence=0.6), 0.0, 3.0)
    assert not any(a.kind == "groove" for a in acts)
    acts = _run(b, lambda t: Observation(music_bpm=120.0, music_confidence=0.6), 3.0, 30.0)
    grooves = [a for a in acts if a.kind == "groove"]
    assert grooves and 0.2 < float(grooves[-1].name) < 0.7
    assert any(a.name == "sing" for a in acts)
    acts = _run(b, lambda t: Observation(), 30.0, 31.0)
    assert not any(a.kind == "groove" for a in acts)


def test_peekaboo_and_shy():
    b, _ = _brain()
    face = FaceObs(1, 0.0, 0.0, 0.05, None, 0.0)
    _run(b, lambda t: Observation(face=face), 0.0, 3.0)
    _run(b, lambda t: Observation(), 3.0, 4.2)  # hides behind hands for 1.2 s
    assert b.state == "ENGAGED"
    acts = _run(b, lambda t: Observation(face=face), 4.2, 4.5)
    assert any(a.name == "giggle" for a in acts)  # came right back
    close = FaceObs(1, 0.0, 0.0, 0.2, None, 0.0)
    acts = _run(b, lambda t: Observation(face=close), 4.5, 22.0)
    assert any(a.name == "shy" for a in acts)


def test_ear_tickle_flinches_then_gets_annoyed():
    b, _ = _brain()
    acts = _run(b, lambda t: Observation(touched=(t < 0.05), touched_side=1), 0.0, 1.5)
    flinch = [a for a in acts if a.kind == "gesture" and a.name.startswith("flinch")]
    assert flinch and flinch[0].name == "flinch:+"
    for k in range(3):
        t0 = 1.5 + k * 1.5
        acts = _run(b, lambda t, t0=t0: Observation(touched=(t < t0 + 0.05), touched_side=0), t0, t0 + 1.5)
    assert any(a.name == "annoyed" for a in acts)


def test_head_pet_leans_in_and_purrs_while_it_lasts():
    b, mem = _brain()
    p = mem.enroll(np.ones(4), now=0.0)
    face = FaceObs(1, 0.0, 0.0, 0.1, p, 0.9)
    _run(b, lambda t: Observation(face=face), 0.0, 1.0)
    acts = _run(b, lambda t: Observation(face=face, petted=(t < 1.05), petting=True), 1.0, 8.0)
    assert any(a.name == "lean" for a in acts) and p.pets == 1
    assert sum(1 for a in acts if a.name == "purr") >= 2


def test_new_voice_makes_it_look_and_name_overrides_a_face():
    b, _ = _brain()
    acts = _run(b, lambda t: Observation(voice_started=(t < 0.05), voice_yaw_deg=-40.0), 0.0, 0.5)
    assert b.state == "SEARCHING" and b.gaze == (-40.0, 0.0)
    assert any(a.name == "perk" for a in acts)
    face = FaceObs(1, 10.0, 0.0, 0.05, None, 0.0)
    _run(b, lambda t: Observation(face=face), 0.5, 3.0)
    assert b.state == "ENGAGED"
    _run(b, lambda t: Observation(face=face, name_heard=(t < 3.05), voice_yaw_deg=45.0), 3.0, 3.2)
    assert b.gaze == (45.0, 0.0)



def test_heard_speech_reacts_to_intent_and_logs_it():
    b, _ = _brain()
    acts = _run(b, lambda t: Observation(heard_text=("what a cute little robot" if t < 0.05 else None)), 0.0, 0.5)
    heard = [a for a in acts if a.kind == "heard"]
    assert heard and heard[0].name.startswith("what a cute little robot|")
    assert any(a.name == "shy" for a in acts)
    acts = _run(b, lambda t: Observation(heard_text=("the load air" if t < 0.55 else None)), 0.5, 1.0)
    assert not any(a.kind in ("sound", "gesture") for a in acts)


def test_body_makes_it_look_up_and_search():
    b, _ = _brain()
    body = FaceObs(-1, 15.0, -20.0, 0.2, None, 0.0)
    acts = _run(b, lambda t: Observation(body=body), 0.0, 1.0)
    assert b.state == "SEARCHING" and b.gaze == (15.0, -20.0)
    assert any(a.name == "perk" for a in acts)
    _run(b, lambda t: Observation(body=body), 1.0, 12.0)
    assert b.state == "SEARCHING"  # keeps looking as long as the body is there


def test_sleeping_ignores_faces_but_wakes_on_loud_or_name():
    b, _ = _brain(awake=False)
    face = FaceObs(1, 0.0, 0.0, 0.1, None, 0.0)
    _run(b, lambda t: Observation(face=face), 0.0, 5.0)
    assert b.state == "SLEEPING"
    acts = _run(b, lambda t: Observation(loud_yaw_deg=(20.0 if t < 5.05 else None)), 5.0, 5.5)
    assert any(a.kind == "wake" for a in acts) and b.state == "WAKING"
