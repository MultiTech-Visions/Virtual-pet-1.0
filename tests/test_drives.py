"""The activity layer: drives that rise and are spent, a chosen pastime, boredom that moves it on."""

import random

from festival_pet.behavior import Action, Behavior, FaceObs, Observation, Timers
from festival_pet.drives import Attention, Drives, Situation, choose
from festival_pet.memory import FaceMemory


def _brain(**drives):
    b = Behavior(FaceMemory("/nonexistent/never-written.json"), timers=Timers(), rng=random.Random(1))
    b.start(0.0, awake=True)
    for k, v in drives.items():
        setattr(b.mood, k, v)
    return b


def _run(b, obs_fn, t0, t1, dt=0.05):
    acts, t = [], t0
    while t < t1:
        acts += b.tick(obs_fn(t), t, dt)
        t += dt
    return acts


def _activities(acts):
    return [a.name for a in acts if a.kind == "activity"]


def test_boredom_rises_on_one_pastime_and_a_new_face_or_touch_relieves_it():
    d = Drives()
    for _ in range(100):
        d.tick(1.0, "ENGAGED", "watch", company=True)
    assert d.boredom > 0.6
    d.tick(1.0, "IDLE", "mime", company=False)  # a game is not a pastime that goes stale
    assert d.boredom < 0.7
    d.tick(1.0, "SLEEPING", "hangout", company=False)
    assert d.boredom == 0.0  # sleep is a fresh start


def test_chooser_scores_have_a_runner_up_and_respect_cooldowns():
    rng = random.Random(3)
    c = choose(Drives(boredom=0.9, curiosity=0.9), Situation(person=True, can_mime=True), "watch", {}, 100.0, rng)
    assert c.activity == "mime" and c.runner_up is not None and "watch" in c.scores
    c2 = choose(Drives(boredom=0.9, curiosity=0.9), Situation(person=True, can_mime=True), "watch", {"mime": 500.0}, 100.0, rng)
    assert "mime" not in c2.scores  # on cooldown: not even a candidate
    c3 = choose(Drives(energy=0.1), Situation(person=True), "watch", {}, 100.0, rng)
    assert c3.activity == "rest"


def test_bored_with_a_person_offers_the_mime_game_and_the_robot_layer_reports_it_busy():
    b = _brain()
    face = FaceObs(1, 0.0, 0.0, 0.05, None, 0.0)
    acts = _run(b, lambda t: Observation(face=face), 0.0, 3.0)
    assert b.activity == "watch" and _activities(acts) == ["watch"]  # a new face: watch them first
    b.mood.boredom, b.mood.curiosity = 0.85, 0.8  # ...a couple of minutes of nothing happening later
    b._activity_since = -20.0  # (and watching them has had its fair go)
    acts = _run(b, lambda t: Observation(face=face), 3.0, 3.5)
    assert "mime" in _activities(acts), b._scores
    assert b.activity == "mime"
    # the robot layer runs the game and says so; while it runs, the brain does not re-decide
    _run(b, lambda t: Observation(face=face, busy="mime"), 3.5, 40.0)
    assert b.activity == "mime"
    # the game ends: back to choosing, and mime is on cooldown
    acts = _run(b, lambda t: Observation(face=face), 40.0, 42.0)
    assert b.activity != "mime" and b._cool["mime"] > 42.0


def test_bored_and_alone_looks_somewhere_new_and_spends_curiosity():
    b = _brain(boredom=0.7, curiosity=0.9)
    acts = _run(b, lambda t: Observation(), 0.0, 3.0)
    assert b.activity == "look_around", b._scores
    c0 = b.mood.curiosity
    seen = set()
    t = 3.0
    while t < 30.0:
        b.tick(Observation(), t, 0.05)
        if b.gaze is not None:
            seen.add(Attention.sector(b.gaze[0]))
        t += 0.05
    assert len(seen) >= 3  # different sectors, not the same flick
    assert b.mood.curiosity < c0 - 0.2  # looking somewhere new costs curiosity


def test_singing_is_chosen_when_bored_and_enabled_and_then_cools_down():
    b = _brain(boredom=0.9, curiosity=0.1)  # bored, and it has already looked everywhere
    b.can_sing = True
    acts = _run(b, lambda t: Observation(), 0.0, 3.0)
    assert "sing" in _activities(acts), b._scores
    _run(b, lambda t: Observation(busy="sing"), 3.0, 20.0)
    assert b.activity == "sing"
    _run(b, lambda t: Observation(), 20.0, 22.0)
    assert b.activity != "sing" and b._cool["sing"] > 100.0


def test_low_social_asks_for_attention_alone_and_begs_with_company():
    b = _brain(social=0.05)
    acts = _run(b, lambda t: Observation(), 0.0, 3.0)
    assert b.activity == "ask_attention"
    assert any(a.kind == "sound" and a.name == "lonely" for a in acts)
    face = FaceObs(1, 0.0, 0.0, 0.05, None, 0.0)
    acts = _run(b, lambda t: Observation(face=face), 3.0, 8.0)
    assert any(a.kind == "sound" and a.name == "excited" for a in acts)  # "hey! over here", the moment someone shows
    _run(b, lambda t: Observation(face=face), 8.0, 60.0)
    assert b.activity == "watch" and b.mood.social > 0.35  # company fills the need; it settles into watching them


def test_low_energy_rests_and_a_close_call_hesitates():
    b = _brain(energy=0.15)
    face = FaceObs(1, 0.0, 0.0, 0.05, None, 0.0)
    acts = _run(b, lambda t: Observation(face=face), 0.0, 3.0)
    assert b.activity == "rest" and any(a.kind == "gesture" and a.name == "droop" for a in acts)
    # hesitation: force two candidates within the margin and check for the "hmm"
    b2 = _brain()
    b2._last_ask = -1e9
    b2.rng = random.Random(0)
    d = b2.mood
    d.energy, d.social, d.curiosity, d.boredom = 0.8, 0.5, 0.6, 0.0
    hesitated = False
    for seed in range(40):
        b3 = _brain(); b3.rng = random.Random(seed)
        acts = _run(b3, lambda t: Observation(), 0.0, 0.2)
        if any("hmm..." in txt for _, txt in b3.thoughts):
            hesitated = any(a.kind == "gesture" and a.name == "tilt" for a in acts)
            break
    assert hesitated  # over 40 seeds, hangout vs look_around lands inside the margin at least once


def test_mind_reports_the_activity_layer():
    b = _brain()
    _run(b, lambda t: Observation(), 0.0, 1.0)
    m = b.mind(1.0)
    assert m["activity"] in ("hangout", "look_around") and "drives" in m and set(m["drives"]) == {"energy", "social", "curiosity", "boredom"}
    assert "scores" in m and "next_decision_in_s" in m


def test_rest_is_given_a_minute_and_the_sleepy_noise_is_rationed():
    b = _brain(energy=0.15)
    face = FaceObs(1, 0.0, 0.0, 0.05, None, 0.0)
    acts = _run(b, lambda t: Observation(face=face), 0.0, 3.0)
    assert b.activity == "rest" and any(a.kind == "sound" and a.name == "sleepy" for a in acts)
    # curiosity and boredom drift and the 30 s clock come and go: it stays resting for a good while
    b.mood.curiosity = 0.95
    acts = _run(b, lambda t: Observation(face=face), 3.0, 50.0)
    assert b.activity == "rest"
    assert sum(1 for a in acts if a.kind == "sound" and a.name == "sleepy") == 0  # no more yawning for two minutes
    # ...but someone leaving is a real change and may end it at once
    _run(b, lambda t: Observation(), 50.0, 53.0)
    assert b.activity != "watch"


def test_energy_drains_slowly_and_rest_restores_it():
    d = Drives(energy=0.8)
    for _ in range(30 * 60):
        d.tick(1.0, "IDLE", "hangout", company=False)
    assert 0.3 < d.energy < 0.4  # half an hour awake: tired, not asleep
    for _ in range(60):
        d.tick(1.0, "IDLE", "rest", company=False)
    assert d.energy > 0.45  # a minute's rest buys a good chunk back
