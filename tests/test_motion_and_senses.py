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
        assert m.request_gesture(name, 0.0, 5, reps=1)
        for t in np.linspace(0.0, dur + 0.2, 60):
            head, ants, _ = m.sample(float(t), 0.02)
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
        head, _, _ = m.sample(i * 0.02, 0.02)
    _, pitch, yaw = _euler(head)
    assert abs(yaw - 30.0) < 3.0 and abs(pitch + 10.0) < 3.0


def test_sleep_blend_lowers_head():
    m = MotionComposer()
    m.mode = "sleeping"
    for i in range(300):
        head, ants, _ = m.sample(i * 0.02, 0.02)
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



def test_body_follows_far_gaze_and_head_stays_within_reach():
    from festival_pet.motion import BODY_DEADBAND, HEAD_YAW_LIMIT

    m = MotionComposer()
    m.set_gaze((90.0, 0.0))  # someone well off to the left
    body = 0.0
    for i in range(400):
        head, _, body = m.sample(i * 0.02, 0.02)
        _, _, yaw = _euler(head)
        assert abs(yaw - body) <= HEAD_YAW_LIMIT + 1e-6
    assert body > 90.0 - BODY_DEADBAND - 1.0  # body turned toward them
    assert abs(yaw - 90.0) < 3.0  # and the head is on target in world frame
    m.set_gaze((body - 5.0, 0.0))  # small correction: inside the deadband, body stays put
    b0 = body
    for i in range(400, 500):
        _, _, body = m.sample(i * 0.02, 0.02)
    assert abs(body - b0) < 1e-6


def test_beep_sway_only_while_talking():
    m = MotionComposer()
    quiet, _, _ = m.sample(0.0, 0.02)
    for i in range(1, 30):
        m.voice_level = 1.0
        head, _, _ = m.sample(i * 0.02, 0.02)
    moved = max(abs(a - b) for a, b in zip(_euler(head), _euler(quiet)))
    assert moved > 0.5
    for i in range(30, 200):
        m.voice_level = 0.0
        head, _, _ = m.sample(i * 0.02, 0.02)
    assert m._voice < 0.01


def test_repeatable_nod_varies_and_lasts_longer():
    m = MotionComposer()
    assert m.request_gesture("nod", 0.0, 5, reps=3)
    assert m.gesture_active(1.0) and not m.gesture_active(1.3)  # 3 x 0.42 s: a quick "yes", not a bow
    m2 = MotionComposer()
    m2.request_gesture("nod", 0.0, 5)
    assert 2 <= m2._gesture.reps <= 4


def test_sneeze_is_a_full_bit_and_silences_the_rest():
    from festival_pet.motion import SNEEZE_S, g_sneeze

    down = g_sneeze(0.3 / SNEEZE_S)
    assert down.pitch > 8 and down.ant_r > 0.5  # looking down, antennas drooped
    lifts = [g_sneeze(t / SNEEZE_S).pitch for t in (0.9, 1.5, 2.1)]
    assert lifts[0] > lifts[1] > lifts[2] < 0  # three lifts, each further back
    assert g_sneeze(2.2 / SNEEZE_S).ant_r < -0.6  # antennas full up just before the choo
    choo = g_sneeze(2.6 / SNEEZE_S)
    assert choo.pitch > 20 and choo.ant_r > 0.8  # hard nod down, antennas dropped
    assert 5 < g_sneeze(3.5 / SNEEZE_S).pitch < choo.pitch  # slow recovery, still down at 3.5 s
    assert abs(g_sneeze(4.7 / SNEEZE_S).yaw) > 2  # clearing shake
    assert abs(g_sneeze(0.999).pitch) < 1 and abs(g_sneeze(0.999).yaw) < 1  # ends home

    m = MotionComposer()
    m.groove, m._groove_level = (0.0, 0.0, 1.0), 1.0
    m.mimic = (30.0, 0.0, 0.0)
    assert m.request_gesture("sneeze", 0.0, 3)
    for i in range(1, 60):
        m.sample(i * 0.02, 0.02)
    assert m._groove_level < 0.5  # groove faded out under the sneeze
    assert abs(m._mimic_pose[0]) < 1.0  # mimic never crept in


def test_pose_history_pairs_frames_with_the_pose_at_exposure():
    import numpy as np

    from festival_pet.senses import PoseHistory

    live = np.eye(4) * 9
    h = PoseHistory(lambda: live, lag_s=0.12)
    assert h.at(5.0) is live  # nothing recorded yet
    for i in range(50):
        p = np.eye(4); p[2, 3] = i  # a pose we can read the time off
        h.record(10.0 + i * 0.02, p)
    assert h.at(10.5)[2, 3] == 25
    assert h.at(10.507)[2, 3] == 25 and h.at(10.513)[2, 3] == 26  # nearest, not floor
    assert h.at(0.0)[2, 3] == 0 and h.at(99.0)[2, 3] == 49  # clamps to what it has
    h.record(20.0, np.eye(4))  # everything older than keep_s is dropped
    assert len(h._hist) == 1


def test_gaze_holds_still_while_grooving():
    m = MotionComposer()
    m.set_gaze((0.0, 0.0))
    for i in range(100):
        m.sample(i * 0.02, 0.02)
    m.groove, m._groove_level = (0.0, 0.0, 1.0), 1.0
    m.set_gaze((30.0, 0.0))  # the "face" jumps: mid-groove that is more likely our own bob than them
    for i in range(100, 150):
        m.sample(i * 0.02, 0.02)
    assert abs(m._gaze[0]) < 4.0  # barely moved in a second
    m.groove, m._groove_level = None, 0.0
    for i in range(150, 250):
        m.sample(i * 0.02, 0.02)
    assert m._gaze[0] > 25.0  # follows again once the groove is over
