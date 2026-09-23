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
    _run(b, lambda t: Observation(), 3.0, 5.0)  # hides behind hands for 2 s (longer than a detector dropout)
    assert b.state == "ENGAGED"
    acts = _run(b, lambda t: Observation(face=face), 5.0, 5.3)
    assert any(a.name == "giggle" for a in acts)  # came right back
    close = FaceObs(1, 0.0, 0.0, 0.2, None, 0.0)
    acts = _run(b, lambda t: Observation(face=close), 5.3, 22.0)
    assert any(a.name == "shy" for a in acts)


def test_ear_tickles_are_keep_away_then_not_in_the_mood_then_a_swat_and_a_nuzzle():
    b, _ = _brain()
    _run(b, lambda t: Observation(), 0.0, 1.0)
    acts, t = [], 1.0
    for _ in range(3):  # three tickles on the right ear: keep-away, with a giggle each time
        acts += b.tick(Observation(touched=True, touched_side=0), t, 0.05)
        t += 2.0
        acts += _run(b, lambda t: Observation(), t, t + 1.0)
        t += 1.0
    assert [a.name for a in acts if a.kind == "ears"] == ["away:0"] * 3
    assert sum(1 for a in acts if a.kind == "sound" and a.name in ("giggle", "ticklish")) == 3
    assert not any(a.name == "annoyed" for a in acts)
    # the fourth: not in the mood, the ear goes over the head
    acts = b.tick(Observation(touched=True, touched_side=0), t, 0.05)
    assert any(a.kind == "ears" and a.name == "tuck:0" for a in acts) and any(a.name == "annoyed" for a in acts)
    t += 2.0
    # disturb it there: the LEFT antenna swats, nuh-uh-uh
    acts = b.tick(Observation(touched=True, touched_side=0), t, 0.05)
    assert any(a.kind == "sound" and a.name == "no_no" for a in acts) and any(a.kind == "gesture" and a.name == "swat:+" for a in acts)
    # keep at it and it keeps batting (one swat per poke once the last one's taps have landed), no make-up yet
    acts = _run(b, lambda t: Observation(), t, t + 1.0) + b.tick(Observation(touched=True, touched_side=0), t + 1.0, 0.05)
    assert not any(a.kind == "gesture" and a.name == "swat:+" for a in acts)  # mid-swat: covered
    acts = _run(b, lambda t: Observation(), t + 1.0, t + 2.0) + b.tick(Observation(touched=True, touched_side=0), t + 2.0, 0.05)
    assert any(a.kind == "gesture" and a.name == "swat:+" for a in acts) and not any(a.name == "nuzzle" for a in acts)
    t += 2.0
    # ...only once they stop do both come back round and it asks for a pet instead
    acts = _run(b, lambda t: Observation(), t, t + 2.0)
    assert not any(a.name == "nuzzle" for a in acts)
    acts = _run(b, lambda t: Observation(), t + 2.0, t + 4.0)
    assert any(a.kind == "ears" and a.name == "clear" for a in acts) and any(a.kind == "gesture" and a.name == "nuzzle" for a in acts)
    assert b._ear_tucked is None and b._ear_tickles == 0


def test_the_sulk_wears_off_on_its_own():
    from festival_pet.behavior import EAR_TUCK_S
    from festival_pet.motion import MotionComposer

    assert EAR_TUCK_S == MotionComposer.EAR_TUCK_S == 26.0
    b, _ = _brain()
    _run(b, lambda t: Observation(), 0.0, 1.0)
    t = 1.0
    for _ in range(4):
        b.tick(Observation(touched=True, touched_side=1), t, 0.05)
        t += 3.0
    assert b._ear_tucked == 1
    _run(b, lambda t: Observation(), t, t + EAR_TUCK_S + 1.0)
    assert b._ear_tucked is None and b._ear_tickles == 0
    acts = b.tick(Observation(touched=True, touched_side=1), t + EAR_TUCK_S + 1.0, 0.05)
    assert any(a.name == "away:1" for a in acts)  # back to the game, not a swat


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


def test_a_torso_with_no_face_is_given_up_on_and_ignored():
    b, _ = _brain()
    body = FaceObs(-1, 40.0, -20.0, 0.2, None, 0.0)
    _run(b, lambda t: Observation(), 0.0, 3.0)
    acts = _run(b, lambda t: Observation(body=body), 3.0, 3.8)
    assert b.gaze == (40.0, -20.0) and b.state == "IDLE"  # paused on it, but not believed yet (no search, no thought)
    assert not any("a body!" in txt for _, txt in b.thoughts)
    _run(b, lambda t: Observation(body=body), 3.8, 5.0)
    assert b.gaze == (40.0, -20.0) and b.state == "SEARCHING"
    _run(b, lambda t: Observation(body=body), 5.0, 12.0)
    assert b.state == "IDLE" and any("not a person" in txt for _, txt in b.thoughts)
    _run(b, lambda t: Observation(body=body), 12.0, 17.0)
    assert b.gaze is None or abs(b.gaze[0] - 40.0) > 1.0  # that spot is ignored now
    other = FaceObs(-1, -80.0, -20.0, 0.2, None, 0.0)
    _run(b, lambda t: Observation(body=other), 17.0, 20.0)
    assert b.gaze == (-80.0, -20.0)  # a torso somewhere else is still worth a look
    assert sum(1 for _, txt in b.thoughts if "a body!" in txt) == 2  # one thought per torso, not per tick


def test_a_glimpse_of_a_torso_pauses_the_sweep_and_blinks_do_not_reset_it():
    b, _ = _brain()
    b._last_face_time = -10.0
    b.activity = "look_around"
    b._look_at, b._look_until, b._next_glance = (120.0, 0.0), 5.0, 5.0
    body = FaceObs(-1, 40.0, -20.0, 0.2, None, 0.0)
    b.tick(Observation(body=body), 1.0, 0.05)
    assert b.gaze == (40.0, -20.0) and b._next_glance >= 2.5  # stopped on it at once, before believing it
    # the detector blinks for half a second: still the same torso
    _run(b, lambda t: Observation(), 1.05, 1.5)
    _run(b, lambda t: Observation(body=body), 1.5, 2.3)
    assert b.state == "SEARCHING" and any("a body!" in txt for _, txt in b.thoughts)


def test_a_solo_gesture_does_not_lose_the_person():
    b, _ = _brain()
    face = FaceObs(1, 20.0, -10.0, 0.05, None, 0.0)
    _run(b, lambda t: Observation(face=face), 0.0, 3.0)
    assert b.state == "ENGAGED"
    _run(b, lambda t: Observation(busy="gesture"), 3.0, 9.5)  # a 6 s bow: the camera is everywhere but on them
    assert b.state == "ENGAGED" and b.gaze == (20.0, -10.0)  # still theirs, still looking where they were


def test_body_makes_it_look_up_and_search():
    b, _ = _brain()
    body = FaceObs(-1, 15.0, -20.0, 0.2, None, 0.0)
    _run(b, lambda t: Observation(), 0.0, 3.0)  # no face for a while first (bodies never override a recent face)
    acts = _run(b, lambda t: Observation(body=body), 3.0, 4.5)  # a second to believe it
    assert b.state == "SEARCHING" and b.gaze == (15.0, -20.0)
    assert any(a.name == "perk" for a in acts)
    _run(b, lambda t: Observation(body=body), 4.5, 9.0)
    assert b.state == "SEARCHING"  # keeps looking as long as the body is there (until it gives up on it)


def test_sleeping_ignores_faces_but_wakes_on_loud_or_name():
    b, _ = _brain(awake=False)
    face = FaceObs(1, 0.0, 0.0, 0.1, None, 0.0)
    _run(b, lambda t: Observation(face=face), 0.0, 5.0)
    assert b.state == "SLEEPING"
    acts = _run(b, lambda t: Observation(loud_yaw_deg=(20.0 if t < 5.05 else None)), 5.0, 5.5)
    assert any(a.kind == "wake" for a in acts) and b.state == "WAKING"


def test_no_flip_flop_between_face_and_body_and_no_regreet():
    b, _ = _brain()
    face = FaceObs(1, 10.0, 0.0, 0.05, None, 0.0)
    body = FaceObs(-1, 10.0, -25.0, 0.2, None, 0.0)
    acts = _run(b, lambda t: Observation(face=face), 0.0, 3.0)
    assert sum(1 for a in acts if a.name == "hello_new") == 1
    # detector dropout: only the body for 1 s, then the face again -> gaze must not jump to the body estimate
    _run(b, lambda t: Observation(body=body), 3.0, 4.0)
    assert b.state == "ENGAGED" and b.gaze == (10.0, 0.0)
    acts = _run(b, lambda t: Observation(face=face), 4.0, 5.0)
    assert not any(a.name in ("giggle", "hello_new") for a in acts)  # no peekaboo, no re-greeting
    # a new track id for the same stranger within 30 s: small acknowledgement, not the full hello
    face2 = FaceObs(2, 10.0, 0.0, 0.05, None, 0.0)
    _run(b, lambda t: Observation(), 5.0, 10.0)
    acts = _run(b, lambda t: Observation(face=face2), 10.0, 13.0)
    assert not any(a.name == "hello_new" for a in acts) and any(a.name == "curious" for a in acts)


def test_nods_back_when_the_person_nods():
    import math

    b, _ = _brain()
    still = FaceObs(1, 0.0, 0.0, 0.04, None, 0.0)
    _run(b, lambda t: Observation(face=still), 0.0, 3.0)
    # a nod is a short burst that ends: two nods, then the head comes to rest
    nodding = lambda t: Observation(face=FaceObs(1, 0.0, 6.0 * math.sin(2 * math.pi * 1.5 * t) if t < 4.4 else 0.0, 0.04, None, 0.0))
    acts = _run(b, nodding, 3.0, 5.5)
    assert any(a.kind == "gesture" and a.name == "nod" for a in acts)
    shaking = lambda t: Observation(face=FaceObs(1, 8.0 * math.sin(2 * math.pi * 1.5 * t) if t < 13.4 else 0.0, 0.0, 0.04, None, 0.0))
    acts = _run(b, shaking, 12.0, 14.5)
    assert any(a.kind == "gesture" and a.name == "shake" for a in acts)


def test_a_bob_that_keeps_going_is_not_a_nod():
    import math

    b, _ = _brain()
    still = FaceObs(1, 0.0, 0.0, 0.04, None, 0.0)
    _run(b, lambda t: Observation(face=still), 0.0, 3.0)
    bobbing = lambda t: Observation(face=FaceObs(1, 0.0, 6.0 * math.sin(2 * math.pi * 2.0 * t), 0.04, None, 0.0))
    acts = _run(b, bobbing, 3.0, 9.0)
    assert not any(a.kind == "gesture" and a.name in ("nod", "shake") for a in acts)
    # and once the dance detector has locked on, no nod-back even if they pause for a beat
    dancing = lambda t: Observation(face=FaceObs(1, 0.0, 0.0, 0.04, None, 0.0), dance_bpm=120.0)
    acts = _run(b, dancing, 9.0, 12.0)
    assert not any(a.kind == "gesture" and a.name in ("nod", "shake") for a in acts)


def test_mirror_game_starts_when_close_and_quiet_then_copies():
    b, _ = _brain()
    far = FaceObs(1, 0.0, 0.0, 0.03, None, 0.0)
    _run(b, lambda t: Observation(face=far), 0.0, 3.0)
    assert not b.mimicking
    close = FaceObs(1, 0.0, 0.0, 0.12, None, 0.0, roll_deg=10.0, head_yaw_deg=20.0, head_pitch_deg=-5.0)
    acts = _run(b, lambda t: Observation(face=close), 3.0, 6.0)
    assert b.mimicking
    mim = [a for a in acts if a.kind == "mimic"]
    assert mim and mim[-1].name == "20.0,-5.0,10.0"
    assert not any(a.kind == "sound" and a.name in ("giggle", "happy", "excited") for a in acts[-50:])
    _run(b, lambda t: Observation(face=far), 6.0, 7.0)
    assert not b.mimicking


def test_visual_dancing_makes_it_groove():
    b, _ = _brain()
    face = FaceObs(1, 0.0, 0.0, 0.05, None, 0.0)
    acts = _run(b, lambda t: Observation(face=face, dance_bpm=120.0), 0.0, 4.0)
    grooves = [a for a in acts if a.kind == "groove"]
    assert grooves and grooves[-1].name.endswith("|visual")
    assert any(a.name == "excited" for a in acts)


def test_dancing_is_not_interrupted_by_reactions_or_a_lost_face():
    b, _ = _brain()
    face = FaceObs(1, 0.0, 0.0, 0.05, None, 0.0)
    _run(b, lambda t: Observation(face=face), 0.0, 3.0)
    acts = _run(b, lambda t: Observation(face=face, dance_bpm=120.0), 3.0, 20.0)
    assert any(a.kind == "groove" for a in acts)
    assert not any(a.kind == "gesture" and a.name in ("tilt", "nod", "wiggle") for a in acts)  # no micro-reactions mid-dance
    # tracker blinks for 4 s while the dance lock holds: no search, no "where did they go"
    acts = _run(b, lambda t: Observation(dance_bpm=120.0), 20.0, 24.0)
    assert b.state == "ENGAGED" and not any(a.kind == "gesture" and a.name == "search" for a in acts)


def test_idle_look_around_turns_the_gaze_wide():
    b, _ = _brain()
    _run(b, lambda t: Observation(), 0.0, 1.0)
    wide = []
    t = 1.0
    while t < 40.0:
        b.tick(Observation(), t, 0.05)
        if b.gaze is not None:
            wide.append(b.gaze[0])
        t += 0.05
    assert wide and max(abs(y) for y in wide) >= 35.0  # not a 22-degree head flick: a proper turn
    assert any(y > 0 for y in wide) and any(y < 0 for y in wide)


def test_ear_played_with_while_petted_is_enjoyed_not_flinched():
    b, _ = _brain()
    _run(b, lambda t: Observation(), 0.0, 1.0)
    acts = b.tick(Observation(touched=True, touched_side=1, petting=True), 1.0, 0.05)
    names = [(a.kind, a.name) for a in acts]
    assert ("gesture", "lean") in names and not any(n.startswith("flinch") for k, n in names if k == "gesture")
    acts = b.tick(Observation(touched=True, touched_side=1), 10.0, 0.05)
    assert any(k == "gesture" and n.startswith("flinch") for k, n in [(a.kind, a.name) for a in acts])


def test_it_hums_a_made_up_jingle_while_pottering_about_but_not_mid_performance():
    from festival_pet.behavior import JINGLE_MAX_S

    b, _ = _brain()
    b.activity = "look_around"
    b.mood.energy = 0.8
    jingles = lambda acts: [a for a in acts if a.kind == "sound" and a.name == "jingle"]  # noqa: E731
    # a jingle is a little song, so the singing switch covers it: off means off
    assert not jingles(_run(b, lambda t: Observation(), 0.0, JINGLE_MAX_S * 3, dt=0.5))
    b, _ = _brain()
    b.activity, b.mood.energy, b.can_sing = "look_around", 0.8, True
    acts = _run(b, lambda t: Observation(), 0.0, JINGLE_MAX_S * 3, dt=0.5)
    assert jingles(acts), "nothing hummed in three windows"
    assert len(jingles(acts)) <= 5  # now and then, not chattering
    # not while it is performing, dancing, grooving, held or flat out
    b._next_jingle = 0.0
    for obs in (Observation(busy="mime"), Observation(busy="sing"), Observation(grooving=True),
                Observation(music_bpm=120.0, music_confidence=0.7), Observation(dance_bpm=120.0), Observation(held=True)):
        b._next_jingle = 0.0
        assert not jingles(b.tick(obs, 500.0, 0.5))
    # nor while it is busy with a game as its activity, or too tired
    b.activity, b._next_jingle = "mime", 0.0
    assert not jingles(b.tick(Observation(), 600.0, 0.5))
    b.activity, b.mood.energy, b._next_jingle = "watch", 0.1, 0.0
    assert not jingles(b.tick(Observation(), 700.0, 0.5))


def test_it_remembers_where_people_have_been_and_checks_back_before_giving_up():
    """The face detector drops out constantly for optical reasons. Remembering one last position and
    scanning the room five seconds later points the camera at a wall, which is how it used to lose
    somebody standing right in front of it."""
    from festival_pet.behavior import COMPANY_RECENT_S, SPOT_KEEP, SPOT_MERGE_DEG

    b, _ = _brain()
    t = 0.0
    for yaw in (10.0, 12.0, -40.0, 70.0):  # three places: the first two are the same person shifting about
        _run(b, lambda _t, y=yaw: Observation(face=FaceObs(1, y, 0.0, 0.05, None, 0.0)), t, t + 4.0)
        t += 4.0
    spots = b.seen_spots.recent(t)
    assert len(spots) == 3 <= SPOT_KEEP and abs(spots[0].yaw - 70.0) < SPOT_MERGE_DEG  # newest first
    assert min(abs(x.yaw - 11.0) for x in spots) < SPOT_MERGE_DEG  # 10 and 12 merged into one place

    # lose them: it works back through the places rather than giving up after the first
    checked, end = [], t + b.timers.face_lost_grace + b.timers.search_duration * 3 + 1.0
    while t < end:
        b.tick(Observation(), t, 0.1)
        if b.state == "SEARCHING":
            checked.append(round(b.gaze[0]))
        t += 0.1
    assert len({y for y in checked}) == 3, "it only checked one place"
    assert b.state == "IDLE"  # ...and it does eventually accept they have gone

    # right after company it checks the remembered places, it does not swing off at a wall
    b.seen_spots.note(25.0, 0.0, t)
    b.activity, b._next_glance, b._look_until = "hangout", t, 0.0
    for _ in range(20):
        t += 0.1
        b.tick(Observation(), t, 0.1)
        if b.gaze is not None:
            assert abs(b.gaze[0]) < 80.0, "wandered off to a wall with somebody just there"
    assert COMPANY_RECENT_S > 30.0
