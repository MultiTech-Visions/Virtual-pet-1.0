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
    for i in range(50):  # lift: strong transient
        held, shaken = d.update([0.5, 0.3, 9.81 + 3.0 * math.sin(i)], [1.5, 0.0, 0.0], t)
        t += 0.02
    assert held and not shaken
    for i in range(200):  # now just held in hands: mild jitter keeps it "held" well past settle_s
        held, _ = d.update([0.3, 0.2, 9.81 + 0.9 * math.sin(3 * i)], [0.4 * math.sin(i), 0.0, 0.0], t)
        t += 0.02
    assert held
    _, shaken = d.update([0.0, 0.0, 9.81], [6.0, 0.0, 0.0], t)
    assert shaken
    for _ in range(150):
        held, _ = d.update(*rest, t)
        t += 0.02
    assert not held


def test_touch_detector_edges():
    d = TouchDetector(persist_ticks=2)
    cmd = [-0.17, 0.17]
    assert not d.update(cmd, [-0.17, 0.17])
    assert not d.update(cmd, [-0.17, 0.6])  # pushed, but must persist
    assert d.update(cmd, [-0.17, 0.6])  # edge
    assert not d.update(cmd, [-0.17, 0.6])  # still held: no new edge
    assert not d.update(cmd, [-0.17, 0.2])  # released
    assert not d.update(cmd, [-0.17, 0.5], busy=True)  # 0.33 rad while animated is not a touch
    assert not d.update(cmd, [-0.17, 0.5], busy=True)
    d.update(cmd, [-0.17, 0.9], busy=True)
    assert d.update(cmd, [-0.17, 0.9], busy=True)


def test_loud_detector_rate_limits_and_maps_direction():
    d = LoudSoundDetector()
    assert d.update(None, 0.0) is None
    assert d.update((math.pi / 2, False), 0.0) is None
    yaw = d.update((0.0, True), 10.0)  # 0 rad = left
    assert yaw is not None and yaw > 0
    assert d.update((math.pi, True), 11.0) is None  # too soon / not quiet
    assert d.update((math.pi, True), 40.0) is not None



def test_pickup_ignores_own_head_motion():
    d = PickupDetector()
    t = 0.0
    for _ in range(30):
        d.update([0.0, 0.0, 9.81], [0.0, 0.0, 0.0], t)
        t += 0.02
    # A fast gesture: big gyro and accel while the app says it is moving the head itself.
    for i in range(60):
        held, shaken = d.update([1.0, 0.5, 9.81 + 3.0 * math.sin(i)], [3.0, 1.0, 0.0], t, self_moving=True)
        t += 0.02
    assert not held and not shaken
    # Slow idle breathing between gestures: small gyro, tiny accel -> still resting.
    for _ in range(60):
        held, _ = d.update([0.1, 0.0, 9.81 + 0.3], [0.3, 0.1, 0.0], t)
        t += 0.02
    assert not held


def test_self_motion_gate():
    from festival_pet.motion import head_pose
    from festival_pet.senses import SelfMotionGate

    g = SelfMotionGate()
    t = 0.0
    assert not g.update(head_pose(0, 0, 0, 0), t)
    for i in range(1, 50):  # slow drift: 0.2 deg per tick = 10 deg/s = 0.17 rad/s
        t += 0.02
        busy = g.update(head_pose(0.2 * i, 0, 0, 0), t)
    assert not busy
    t += 0.02
    assert g.update(head_pose(25.0, 0, 0, 0), t)  # a 15 deg snap in one tick
    t += 0.3
    assert g.update(head_pose(25.0, 0, 0, 0), t)  # still inside the hangover
    t += 0.3
    assert not g.update(head_pose(25.0, 0, 0, 0), t)


def test_touch_ignores_motor_lag_and_learns_droop():
    d = TouchDetector()
    edge = False
    for _ in range(5):
        edge |= d.update([-0.17, 0.17], [-0.17, 0.17])
    # Command snaps from 0.17 to 1.1 rad; the motor trails behind for a few ticks -> no touch.
    for k in range(12):
        cmd = [-0.17, 1.1]
        present = [-0.17, 0.17 + (1.1 - 0.17) * min(1.0, k / 8)]
        edge |= d.update(cmd, present, busy=True)
    assert not edge
    # At rest the left antenna droops 0.2 rad below command: learned as baseline, never a touch.
    for _ in range(1500):
        edge |= d.update([-0.17, 0.17], [-0.17, -0.03], busy=False, dt=0.02)
    assert not edge
    # Now a finger pushes it a further 0.5 rad.
    hits = [d.update([-0.17, 0.17], [-0.17, -0.53], busy=False) for _ in range(6)]
    assert any(hits) and d.last_side == 1
