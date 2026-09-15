import math

import numpy as np

from festival_pet.motion import GESTURES, PITCH_LIMIT, YAW_LIMIT, MotionComposer
from festival_pet.senses import LoudSoundDetector, PickupDetector, TouchDetector


def _euler(pose):
    from scipy.spatial.transform import Rotation as R

    return R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)


def test_gestures_stay_in_envelope_and_finish():
    m = MotionComposer()
    for name, (dur, _) in GESTURES.items():
        assert m.request_gesture(name, 0.0, 5)
        for t in np.linspace(0.0, dur + 0.2, 60):
            head, ants = m.sample(float(t), 0.02)
            roll, pitch, yaw = _euler(head)
            assert abs(yaw) <= YAW_LIMIT + 1e-6 and abs(pitch) <= PITCH_LIMIT + 1e-6
            assert all(math.isfinite(a) for a in ants)
        assert not m.gesture_active(dur + 0.3)


def test_priority_preemption():
    m = MotionComposer()
    assert m.request_gesture("snuggle", 0.0, 4)
    assert not m.request_gesture("nod", 0.1, 1)
    assert m.request_gesture("startle", 0.2, 5)


def test_gaze_converges():
    m = MotionComposer()
    m.set_gaze((30.0, -10.0))
    for i in range(200):
        head, _ = m.sample(i * 0.02, 0.02)
    _, pitch, yaw = _euler(head)
    assert abs(yaw - 30.0) < 3.0 and abs(pitch + 10.0) < 3.0


def test_sleep_blend_lowers_head():
    m = MotionComposer()
    m.mode = "sleeping"
    for i in range(300):
        head, ants = m.sample(i * 0.02, 0.02)
    _, pitch, _ = _euler(head)
    assert pitch > 15.0 and ants[0] < -2.0 and ants[1] > 2.0


def test_pickup_detector_lifecycle():
    d = PickupDetector()
    rest = ([0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
    t = 0.0
    for _ in range(50):
        held, shaken = d.update(*rest, t)
        t += 0.02
    assert not held and not shaken
    for i in range(40):  # lift: wobbly accel + some gyro
        held, shaken = d.update([0.5, 0.3, 9.81 + 2.5 * math.sin(i)], [0.8, 0.0, 0.0], t)
        t += 0.02
    assert held and not shaken
    _, shaken = d.update([0.0, 0.0, 9.81], [6.0, 0.0, 0.0], t)
    assert shaken
    for _ in range(150):
        held, _ = d.update(*rest, t)
        t += 0.02
    assert not held


def test_touch_detector_edges():
    d = TouchDetector()
    cmd = [-0.17, 0.17]
    assert not d.update(cmd, [-0.17, 0.17])
    assert d.update(cmd, [-0.17, 0.6])  # pushed
    assert not d.update(cmd, [-0.17, 0.6])  # still held: no new edge
    assert not d.update(cmd, [-0.17, 0.2])  # released
    assert d.update(cmd, [-0.17, 0.6])


def test_loud_detector_rate_limits_and_maps_direction():
    d = LoudSoundDetector()
    assert d.update(None, 0.0) is None
    assert d.update((math.pi / 2, False), 0.0) is None
    yaw = d.update((0.0, True), 10.0)  # 0 rad = left
    assert yaw is not None and yaw > 0
    assert d.update((math.pi, True), 11.0) is None  # too soon / not quiet
    assert d.update((math.pi, True), 40.0) is not None
