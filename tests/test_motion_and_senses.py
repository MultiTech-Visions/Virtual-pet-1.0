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


def test_grooving_holds_the_gaze_against_jitter_but_follows_a_real_move():
    from festival_pet.motion import BODY_DEADBAND_GROOVE, LEAN_BODY_DEG

    m = MotionComposer()
    m.set_gaze((0.0, 0.0))
    for i in range(300):  # grooving hard, settled on someone straight ahead
        m.groove = ((i * 0.02 / 0.5) % 1.0, 0.0, 0.8)
        m.sample(i * 0.02, 0.02)
    assert m._groove_level > 0.5
    m.set_gaze((6.0, 0.0))  # a small correction (the bob in the face reading): held against
    for i in range(300, 350):
        m.groove = ((i * 0.02 / 0.5) % 1.0, 0.0, 0.8)
        head, _, body = m.sample(i * 0.02, 0.02)
    assert abs(_euler(head)[2]) < 2.0 and body == 0.0
    # a nudge leans the body a few degrees that way and no further: it never turns off the person
    m.groove_lean = 1.0
    for i in range(350, 420):
        m.groove = ((i * 0.02 / 0.5) % 1.0, 0.0, 0.8)
        head, _, body = m.sample(i * 0.02, 0.02)
        m.groove_lean = 1.0  # as if the key were being held down: the lean decays otherwise
    assert 1.0 < abs(body) <= LEAN_BODY_DEG + 3.0  # (+ the groove's own body sway)
    m.groove_lean = 0.0
    # they move for real: the gaze follows all the way in, and the body comes round once it is worth it
    m.set_gaze((60.0, 0.0))
    for i in range(420, 700):
        m.groove = ((i * 0.02 / 0.5) % 1.0, 0.0, 0.8)
        head, _, body = m.sample(i * 0.02, 0.02)
    # (grooving, the body stops at the edge of its wide deadband: the head covers the rest, well inside its reach)
    assert abs(_euler(head)[2] - 60.0) < 8.0 and body >= 60.0 - BODY_DEADBAND_GROOVE - 1.0 and abs(m.yaw_short) < 1e-6


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
    from festival_pet.motion import ANT_FULL_DOWN, SNEEZE_S, g_sneeze

    down = g_sneeze(0.8 / SNEEZE_S)
    assert down.pitch > 8 and down.ant_r > ANT_FULL_DOWN - 0.05  # looking down, antennas laid right back
    lifts = [g_sneeze(t / SNEEZE_S).pitch for t in (1.8, 3.0, 4.2)]
    assert lifts[0] > lifts[1] > lifts[2] < 0  # three lifts, each further back
    ants = [g_sneeze(t / SNEEZE_S).ant_r for t in (2.3, 3.5, 4.7)]
    assert ants[0] > ants[1] > ants[2]  # antennas rise a step per lift
    wind = g_sneeze(5.35 / SNEEZE_S)
    assert wind.pitch < -20 and wind.ant_r > 0.4 and wind.ant_l < -0.4  # head back, antennas crossed on top
    choo = g_sneeze(5.7 / SNEEZE_S)
    assert choo.pitch > 20 and choo.ant_r < -1.2 and choo.ant_l > 1.2  # hard nod down, antennas out wide
    mid = g_sneeze(7.0 / SNEEZE_S)
    assert 5 < mid.pitch < choo.pitch  # slow recovery, still down at 7 s
    assert abs(g_sneeze(9.2 / SNEEZE_S).yaw) > 2  # clearing shake
    assert abs(g_sneeze(0.999).pitch) < 1 and abs(g_sneeze(0.999).yaw) < 1 and abs(g_sneeze(0.999).ant_r) < 0.1  # ends home

    m = MotionComposer()
    m.set_gaze((30.0, -30.0))
    for i in range(100):
        m.sample(i * 0.02, 0.02)
    m.groove, m._groove_level = (0.0, 0.0, 1.0), 1.0
    m.mimic = (30.0, 0.0, 0.0)
    assert m.request_gesture("sneeze", 2.0, 3)
    for i in range(100, 200):
        m.sample(i * 0.02, 0.02)
    assert m._groove_level < 0.5  # groove faded out under the sneeze
    assert abs(m._mimic_pose[0]) < 1.0  # mimic never crept in
    assert abs(m._gaze[0] - 30.0) < 1.0 and abs(m._gaze[1]) < 3.0  # it stopped looking UP at them: the bit plays level, still facing them


def test_head_slides_forward_when_looking_up_and_pose_hold_looks_there():
    m = MotionComposer()
    m.set_gaze((0.0, -30.0))
    for i in range(200):
        head, _, _ = m.sample(i * 0.02, 0.02)
    assert head[0, 3] > 0.008  # forward shift near the top of the range (base x is forward)
    m.set_gaze((0.0, 20.0))
    for i in range(200, 400):
        head, _, _ = m.sample(i * 0.02, 0.02)
    assert head[1, 3] == 0.0 and head[0, 3] == 0.0  # none when looking down
    m.forward_shift_m = 0.0
    m.set_gaze((0.0, -30.0))
    for i in range(400, 600):
        head, _, _ = m.sample(i * 0.02, 0.02)
    assert head[0, 3] == 0.0
    # a held pose overrides the gaze, with roll
    m.hold = (25.0, 0.0, 15.0, 12.0 + 3.0)
    for i in range(600, 700):
        m.sample(i * 0.02, 0.02)
    assert m._gaze[0] > 20 and m._hold_roll > 12



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


def test_held_mode_locks_the_body_and_reports_the_shortfall():
    m = MotionComposer()
    m.held = True
    m.set_gaze((90.0, 0.0))
    for i in range(200):
        _, _, body = m.sample(i * 0.02, 0.02)
    assert body == 0.0 and m.yaw_short > 30  # wanted 90, head can do 45 from a body that will not turn
    m.set_gaze((-20.0, 0.0))
    for i in range(200, 400):
        _, _, body = m.sample(i * 0.02, 0.02)
    assert abs(m.yaw_short) < 1.0
    m.held = False
    m.set_gaze((90.0, 0.0))
    for i in range(400, 700):
        _, _, body = m.sample(i * 0.02, 0.02)
    assert body > 30  # the body turns again


def test_point_gesture_points_with_that_antenna():
    from festival_pet.motion import g_point

    left = g_point(0.7, 1.0)
    assert left.ant_l < -0.8 and left.ant_r < 0 and left.yaw > 10  # left antenna forward-down, head strains left
    right = g_point(0.7, -1.0)
    assert right.ant_r > 0.8 and right.ant_l > 0 and right.yaw < -10
    assert g_point(0.1, 1.0).z > 0.005  # bounces first


def test_ear_holds_park_an_antenna_and_expire():
    from festival_pet.motion import ANTENNA_NEUTRAL

    m = MotionComposer()
    m.ears_away(0, 1.0, hold_s=2.0)
    _, ants, _ = m.sample(1.1, 0.02)
    assert ants[0] == -m.EAR_AWAY[0] and abs(ants[1] - ANTENNA_NEUTRAL[1]) < 0.3  # right parked far back, left free
    m.ears_away(0, 1.5, hold_s=2.0)
    _, ants, _ = m.sample(1.6, 0.02)
    assert ants[0] == -m.EAR_AWAY[1]  # keep-away alternates: now forward
    m.ears_tuck(1, 2.0, hold_s=5.0)
    _, ants, _ = m.sample(2.1, 0.02)
    assert ants[1] == -m.EAR_TUCK  # left antenna over the head (its sign flipped: forward)
    _, ants, _ = m.sample(8.0, 0.02)
    assert m.ear_hold == [None, None] and abs(ants[1] - ANTENNA_NEUTRAL[1]) < 0.3  # expired
    m.request_gesture("swat", 8.0, 3, side=1.0)
    _, ants, _ = m.sample(8.3, 0.02)
    assert ants[1] < ANTENNA_NEUTRAL[1] - 0.5  # the left antenna sweeps forward to bat
    # the nuzzle is a circle, not a push: the face goes up-and-forward, over, then down-and-back, twice
    from festival_pet.motion import NUZZLE_CIRCLES, NUZZLE_S

    m.request_gesture("nuzzle", 10.0, 3)
    xs, pitches = [], []
    t = 10.0
    while t < 10.0 + NUZZLE_S:
        head, _, _ = m.sample(t, 0.02)
        xs.append(head[0, 3])
        pitches.append(_euler(head)[1])
        t += 0.02
    assert max(xs) > 0.01 and min(xs) < -0.008  # it pushes forward and pulls back
    assert max(pitches) > 6.0 and min(pitches) < 0.0  # ...gaze down at the bottom, head up at the top
    # forward leads the rise by a quarter turn, so x peaks roughly halfway up: one circle, not a nod
    turns = sum(1 for a, b in zip(xs, xs[1:]) if (a > 0) != (b > 0))
    assert turns >= 2 * NUZZLE_CIRCLES - 1


def test_antennas_as_arms_read_literally_and_follow_fast():
    from festival_pet.motion import ANTENNA_NEUTRAL, ARM_BACK, ARM_FORWARD, arm_rad

    near = lambda a, b: abs(a - b) < 1e-9  # noqa: E731
    assert near(arm_rad(0.0), -ARM_BACK) and near(arm_rad(90.0), ARM_FORWARD) and arm_rad(180.0) == 0.0  # down: laid back; out: forward; up: vertical
    assert near(arm_rad(45.0), -ARM_BACK + (ARM_FORWARD + ARM_BACK) / 2) and near(arm_rad(135.0), ARM_FORWARD / 2)  # continuous in between
    assert arm_rad(-20.0) == arm_rad(0.0) and arm_rad(400.0) == arm_rad(180.0)
    m = MotionComposer()
    for i in range(50):
        m.sample(i * 0.02, 0.02)
    m.show_arms(90.0, 0.0, 1.0, 5.0)  # left out, right down
    for i in range(50, 110):
        _, ants, _ = m.sample(i * 0.02, 0.02)
    assert abs(ants[1] - (-ARM_FORWARD)) < 0.1 and abs(ants[0] - (-ARM_BACK)) < 0.1  # left antenna forward (its sign is negated), right laid back
    m.show_arms(180.0, 180.0, 2.2, 5.0)  # both up: within a quarter beat at 120 bpm it is most of the way there
    _, ants, _ = m.sample(2.34, 0.02)
    for i in range(118, 125):
        _, ants, _ = m.sample(i * 0.02, 0.02)
    assert abs(ants[0]) < 0.6 and abs(ants[1]) < 0.6
    m.ear_clock(0, 0.0, 2.5)  # the clock hand still wins on its antenna
    _, ants, _ = m.sample(2.52, 0.02)
    assert ants[0] == m.EAR_CLOCK_DOWN
    m.show_arms(None)
    for i in range(130, 230):
        _, ants, _ = m.sample(i * 0.02, 0.02)
    assert abs(ants[0] - ANTENNA_NEUTRAL[0]) < 0.3 and abs(ants[1] - ANTENNA_NEUTRAL[1]) < 0.3  # let go: back to normal


def test_imu_rub_calibration_sets_the_floor_between_rest_and_a_rub():
    from festival_pet.senses import ImuRubDetector

    d = ImuRubDetector()
    d.start_calibration(0.0)
    assert "still" in d.calibration["phase"]
    t = 0.0
    while t < 3.1:
        assert not d.calibration_step(0.04 + 0.01 * (int(t * 50) % 3), False, t); t += 0.02
    assert "rub" in d.calibration["phase"]
    done = False
    while t < 7.2 and not done:
        done = d.calibration_step(0.3 + 0.05 * (int(t * 50) % 4), False, t); t += 0.02
    assert done and d.calibration["phase"] == "done"
    assert 0.06 < d.gyro_lo < 0.3 and d.gyro_hi >= 1.0
    # a rub that does not stand out from rest fails loudly instead of setting nonsense
    d2 = ImuRubDetector(); d2.start_calibration(0.0); t = 0.0
    while t < 7.5:
        if d2.calibration_step(0.05, False, t):
            break
        t += 0.02
    assert d2.calibration["phase"] == "failed" and "stand out" in d2.calibration["reason"]


def test_petting_folds_the_antennas_into_a_steady_x_with_a_tiny_push():
    from festival_pet.motion import ANTENNA_NEUTRAL

    m = MotionComposer()
    m.petted = True
    m.groove = (0.0, 0.0, 1.0)  # even mid-groove...
    m._groove_level = 1.0
    ants = []
    for i in range(500):
        _, a, _ = m.sample(i * 0.02, 0.02)
        ants.append(a)
    late = np.array(ants[300:])
    assert abs(late[:, 0].mean() - (ANTENNA_NEUTRAL[0] + 0.9)) < 0.03 and abs(late[:, 1].mean() - (ANTENNA_NEUTRAL[1] - 0.9)) < 0.03
    assert 0.02 < late[:, 0].max() - late[:, 0].min() < 0.1  # ...they hold the X, with only the hair's-breadth push
    m.petted = False
    for i in range(500, 1200):
        _, a, _ = m.sample(i * 0.02, 0.02)
    assert abs(a[0] - ANTENNA_NEUTRAL[0]) < 0.4  # the hand gone, they come back up


def test_pose_history_survives_concurrent_reads_and_writes():
    import threading
    import numpy as np

    from festival_pet.senses import PoseHistory

    h = PoseHistory(lambda: np.eye(4), keep_s=0.05)
    stop = threading.Event()
    errors = []

    def writer():
        t = 0.0
        while not stop.is_set():
            h.record(t, np.eye(4) * t); t += 0.001

    def reader():
        while not stop.is_set():
            try:
                h.at(0.0)
            except Exception as e:  # the old code raised "deque mutated during iteration" here
                errors.append(e)
    threads = [threading.Thread(target=writer), threading.Thread(target=reader), threading.Thread(target=reader)]
    for th in threads:
        th.start()
    import time as _t
    _t.sleep(0.4)
    stop.set()
    for th in threads:
        th.join()
    assert not errors


def test_far_gaze_is_carried_by_the_body_and_the_head_stops_short_of_the_frame():
    from festival_pet.motion import HEAD_YAW_LIMIT

    m = MotionComposer()
    m.set_gaze((120.0, 0.0))
    for i in range(120):  # 2.4 s
        head, _, body = m.sample(i * 0.02, 0.02)
    yaw_world = math.degrees(math.atan2(head[1, 0], head[0, 0]))
    assert abs(yaw_world - body) <= HEAD_YAW_LIMIT + 1e-6  # never twisted into the body frame
    assert body > 60.0  # the body did the coarse work quickly
    for i in range(120, 300):
        head, _, body = m.sample(i * 0.02, 0.02)
    yaw_world = math.degrees(math.atan2(head[1, 0], head[0, 0]))
    assert abs(yaw_world - 120.0) < 3.0 and abs(body - 120.0) < 15.0  # and the head did the fine aim


def test_the_wave_is_a_wave_and_peace_is_a_routine():
    """Both used to be near enough the same shrug. A wave has to read as HEY OVER HERE from across a
    field, and peace has to be visibly different from just standing there."""
    from festival_pet.motion import ANT_FULL_DOWN, GESTURES, WAVE_ARC, WAVE_S, WAVE_SWEEPS, WAVE_TUCK

    m = MotionComposer()
    m.energy = 0.0  # no breathing on top, so the numbers are the gesture and nothing else
    m.request_gesture("wave", 0.0, 5, side=-1.0)  # the RIGHT antenna waves
    waving, other, t = [], [], 0.0
    while t < WAVE_S:
        _, ants, _ = m.sample(t, 0.02)
        waving.append(ants[0])
        other.append(ants[1])
        t += 0.02
    # it sweeps right through upright, both ways, several times over: a wave, not a twitch
    assert max(waving) > WAVE_ARC * 0.8 and min(waving) < -WAVE_ARC * 0.8
    crossings = sum(1 for a, b in zip(waving, waving[1:]) if (a > 0) != (b > 0))
    assert crossings >= 2 * WAVE_SWEEPS - 1
    assert max(other) >= WAVE_TUCK * 0.8  # ...and the other one gets out of the picture
    # ...slowly: a parade wave, about a second a sweep, not the buzz of a shake
    assert WAVE_S / WAVE_SWEEPS > 0.8

    m = MotionComposer()
    m.energy = 0.0
    m.request_gesture("peace", 10.0, 5)
    dur = GESTURES["peace"][0]
    seq, t = [], 10.0
    while t < 10.0 + dur:
        _, ants, _ = m.sample(t, 0.02)
        seq.append(ants[0])
        t += 0.02
    bottom = seq.index(min(seq))
    assert min(seq) < -ANT_FULL_DOWN * 0.9  # all the way down first...
    assert max(seq[bottom:]) > -0.05  # ...and then up to the Y
    assert bottom < len(seq) * 0.45  # in that order, with most of the gesture spent on the way up


def test_a_tilted_head_is_not_also_allowed_to_drop():
    """Rolled right over, the side of the head is already next to the body frame; dropping it as well is
    what knocks them together, which is what repeated groove nudges on the keypad used to do."""
    from festival_pet.motion import ROLL_LIMIT, TILT_Z_FROM

    m = MotionComposer()
    m.energy = 0.0
    lows = {}
    for lean in (0.0, 1.0):
        m.groove_lean = lean
        m._lean = lean  # already leaning, not easing into it
        worst = 0.0
        for k in range(400):
            t = 20.0 + k * 0.02
            m.groove = ((k * 0.02 / 0.5) % 1.0, 0.0, 1.0)
            m.request_gesture("bounce", t, 3)
            head, _, _ = m.sample(t, 0.02)
            roll = _euler(head)[0]
            if abs(roll) > TILT_Z_FROM * ROLL_LIMIT:
                worst = min(worst, head[2, 3])
        lows[lean] = worst
    assert lows[0.0] <= 0.0
    assert lows[1.0] > -1e-9  # tilted over: no drop at all, whatever the groove and the gesture ask for
