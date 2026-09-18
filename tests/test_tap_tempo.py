import math

import numpy as np
import pytest

from festival_pet.motion import GrooveMix, MotionComposer, groove_offsets, head_pose, turn_pose
from festival_pet.tap_tempo import TapTempo


def test_tap_tempo_locks_and_free_wheels():
    t = TapTempo()
    assert not t.active
    for i in range(5):
        t.tap(10.0 + i * 0.5)  # 120 bpm
    assert abs(t.bpm - 120.0) < 0.01
    assert t.phase(12.0) == pytest.approx(0.0)
    assert t.phase(12.25) == pytest.approx(0.5)
    assert t.phase(60.0) == pytest.approx(0.0)  # keeps time long after the last tap


def test_one_sloppy_tap_does_not_move_the_tempo_and_a_pause_restarts():
    t = TapTempo()
    for i in range(6):
        t.tap(i * 0.5)
    t.tap(3.0 + 0.2)  # late by 200 ms
    assert abs(t.bpm - 120.0) < 0.01  # median holds
    t.tap(20.0)  # long pause: new count, previous tempo kept until two new taps
    assert abs(t.bpm - 120.0) < 0.01
    t.tap(21.0)  # 60 bpm
    assert abs(t.bpm - 60.0) < 0.01


def test_downbeat_counts_bars_and_phrases():
    t = TapTempo()
    t.tap(0.0, downbeat=True)
    for i in range(1, 4):
        t.tap(i * 0.5)
    assert t.downbeat_known
    assert t.beat_in_bar(0.0) == 1 and t.beat_in_bar(0.5) == 2 and t.beat_in_bar(1.6) == 4 and t.beat_in_bar(2.0) == 1
    assert t.bar_in_phrase(0.0) == 1 and t.bar_in_phrase(2.0) == 2 and t.bar_in_phrase(6.1) == 4 and t.bar_in_phrase(8.0) == 1
    # tapping along on beat 3 later keeps the "1" where it was
    t.tap(9.0)  # 18 beats after the 1 -> beat 3 of bar 5
    assert t.beat_in_bar(9.0) == 3
    assert t.phrase_phase(8.0) == pytest.approx(0.0)


def test_set_and_clear():
    t = TapTempo()
    t.set_bpm(100)
    assert t.active and t.period == pytest.approx(0.6)
    with pytest.raises(ValueError):
        t.set_bpm(500)
    t.clear()
    assert not t.active and not t.downbeat_known


def test_turn_pose_rotates_about_base_vertical():
    forward = head_pose(0.0, 10.0, 0.0, 0.0)
    turned = turn_pose(forward, 90.0)
    heading = turned[:3, 0]  # the head's forward axis now points along base +y, still pitched 10 deg down
    assert heading[0] == pytest.approx(0.0, abs=1e-9) and heading[1] == pytest.approx(math.cos(math.radians(10)))
    assert np.allclose(turn_pose(forward, 0.0), forward)
    assert np.allclose(turn_pose(forward, 90.0), head_pose(90.0, 10.0, 0.0, 0.0))  # same as authoring it turned
    # the forward shift turns with the heading: facing forward it is +x, 90 deg round it is +y
    assert head_pose(0.0, -30.0, 0.0, 0.0, 0.02)[0, 3] == pytest.approx(0.02)
    shifted = head_pose(90.0, -30.0, 0.0, 0.0, 0.02)
    assert shifted[1, 3] == pytest.approx(0.02) and shifted[0, 3] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(turn_pose(head_pose(0.0, -30.0, 0.0, 0.0, 0.02), 90.0), shifted)


def test_bow_covers_the_house_and_plays_from_neutral():
    from festival_pet.motion import BOW_S, g_bow

    centre, right, left = g_bow(0.5 / 3), g_bow(1.5 / 3), g_bow(2.5 / 3)  # mid-dip of each bow
    assert centre.pitch > 25 and abs(centre.yaw) < 1e-9 and centre.ant_r > 1.0 and centre.ant_l == 0.0
    assert right.pitch > 25 and right.yaw < -19 and right.ant_l < -1.0 and right.ant_r == 0.0
    assert left.pitch > 25 and left.yaw > 19 and left.ant_r > 1.0 and left.ant_l < -1.0
    assert g_bow(0.0).pitch == 0.0 and g_bow(0.999).pitch < 2.0  # up at the start and the end
    m = MotionComposer()
    m.set_gaze((0.0, -30.0))  # looking up at someone standing
    for i in range(100):
        m.sample(i * 0.02, 0.02)
    m.request_gesture("bow", 2.0, 4)
    for i in range(100, 160):
        head, _, _ = m.sample(i * 0.02, 0.02)
    assert m._gaze[1] > -5.0  # gaze let go of the person, so the dip reads as a bow
    assert head[0, 3] < 0.003  # and the up-look forward shift went with it
    dip = g_bow(0.5 / 3)
    assert dip.x < -0.015  # the head slides back as it dips, clear of the body's front lip
    assert BOW_S == 6.0


def test_groove_mix_scales_parts_and_body_sways_over_the_bar():
    quiet = groove_offsets(0.0, 1.0, 0.25, 0, GrooveMix(bob=0, sway=0, body=0, ears=0))
    assert quiet.pitch == 0 and quiet.body == 0 and quiet.ant_r == 0
    loud = groove_offsets(0.0, 1.0, 0.0, 0, GrooveMix(bob=2, body=2))
    assert loud.pitch != 0 and abs(loud.body) > 10
    # downbeat accent: beat 1 dips harder than beat 3 at the same beat phase
    one = groove_offsets(0.0, 1.0, 0.0, 0, None, accent_downbeat=True)
    three = groove_offsets(0.0, 1.0, 0.5, 0, None, accent_downbeat=True)
    assert abs(one.pitch) > abs(three.pitch)
    # the composer returns the body sway on top of its own body yaw, and the head yaw stays within reach of it
    m = MotionComposer()
    m.body_yaw = 30.0
    m.groove = (0.0, 0.25, 1.0)
    m.groove_mix = GrooveMix(body=2)
    m._groove_level = 1.0
    _, _, body = m.sample(1.0, 0.02)
    assert body != 30.0
