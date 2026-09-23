import random

from festival_pet.behavior import FaceObs
from festival_pet.mime import DEMO_S, GAP_S, INTRO_S, MOVES, PRAISE_S, WAIT_S, MimeGame


def face(head_yaw=0.0, head_pitch=0.0, roll=0.0):
    return FaceObs(1, 0.0, 0.0, 0.05, None, 0.0, roll, head_yaw, head_pitch)


def run(g, t0, t1, face_fn, dt=0.05):
    out, t = [], t0
    while t < t1:
        out += g.tick(face_fn(t), t)
        t += dt
    return out


def kinds(acts, kind):
    return [a[1] for a in acts if a[0] == kind]


def test_sequences_are_random_and_never_repeat_a_move():
    seqs = set()
    for i in range(20):
        g = MimeGame(random.Random(i)); g.start(0.0)
        seqs.add(tuple(g.sequence))
    assert len(seqs) > 5
    for i in range(50):
        g = MimeGame(random.Random(i)); g.start(0.0)
        assert 3 <= len(g.sequence) <= 5 and all(a != b for a, b in zip(g.sequence, g.sequence[1:]))
        assert set(g.sequence) <= set(MOVES)


def test_copying_every_move_wins_and_captures_faces():
    g = MimeGame(random.Random(3), mirror_image=True)
    acts = g.start(0.0)
    assert "mime_start" in kinds(acts, "sound")
    t, everything = 0.0, list(acts)
    for move in list(g.sequence):
        yaw, pitch, roll = MOVES[move]
        # intro/demo/gap pass with a neutral face...
        lead = INTRO_S if g.state == "intro" else PRAISE_S
        acts = run(g, t, t + lead + DEMO_S + GAP_S + 0.2, lambda _: face())
        everything += acts
        assert g.state == "wait", (move, g.state)
        t += lead + DEMO_S + GAP_S + 0.2
        # ...then they copy it, mirror-image: yaw and roll flip, pitch does not
        shown = lambda _: face(head_yaw=-yaw * 1.2, head_pitch=pitch * 1.2, roll=-roll * 1.2)
        acts = run(g, t, t + 0.6, shown)
        everything += acts
        t += 0.6
        assert "yes" in kinds(acts, "sound") and ("capture",) in acts, move
    assert g.state == "celebrate" and "tada" in kinds(acts, "sound")
    holds = [h for h in kinds(everything, "hold") if h is not None]
    for move in g.sequence:  # every move was shown, at its nominal size (no repeats were needed), from the face
        assert any((h[0], h[1], h[2]) == MOVES[move] for h in holds), move
    acts = run(g, t, t + 3.0, lambda _: face())
    assert not g.active and g.outcome == "won" and "mime_end" in kinds(acts, "sound")


def test_moves_are_shown_relative_to_the_face_and_the_gap_returns_to_it():
    g = MimeGame(random.Random(5))
    up = FaceObs(1, 30.0, -25.0, 0.05, None, 0.0)  # standing off to the left, above the camera
    acts = g.start(0.0) + run(g, 0.0, INTRO_S + 0.1, lambda _: up)
    assert g.center == (30.0, -25.0)
    demo = [h for h in kinds(acts, "hold") if h is not None][0]
    yaw, pitch, roll = MOVES[g.sequence[0]]
    assert demo[:3] == (30.0 + yaw, -25.0 + pitch, roll)  # from their face, not from neutral
    acts = run(g, INTRO_S + 0.1, INTRO_S + DEMO_S + 0.2, lambda _: None)  # face lost while the head is away: fine
    gap = [h for h in kinds(acts, "hold") if h is not None][0]
    assert gap[:3] == (30.0, -25.0, 0.0) and g.active


def test_ignoring_it_escalates_then_it_gives_up():
    g = MimeGame(random.Random(1))
    g.start(0.0)
    acts = run(g, 0.0, INTRO_S + 3 * (DEMO_S + GAP_S + WAIT_S) + 1.0, lambda _: face())
    sounds = kinds(acts, "sound")
    assert sounds.index("huff") < sounds.index("annoyed") < sounds.index("sad")
    assert "shake" in kinds(acts, "gesture") and "droop" in kinds(acts, "gesture")
    assert not g.active and g.outcome == "gave up"
    # the second and third demos are bigger than the first
    holds = [h for h in kinds(acts, "hold") if h is not None and any(abs(v) > 1 for v in h[:3])]
    assert abs(sum(holds[1][:3])) > abs(sum(holds[0][:3]))


def test_losing_the_face_ends_the_game_and_stop_works():
    g = MimeGame(random.Random(2))
    g.start(0.0)
    acts = run(g, 0.0, 8.0, lambda _: None)
    assert not g.active and g.outcome == "lost you" and "confused" in kinds(acts, "sound")
    g.start(10.0)
    assert g.active
    acts = g.stop(11.0)
    assert not g.active and g.outcome == "stopped" and "mime_end" in kinds(acts, "sound")
    assert g.stop(12.0) == []


def test_moves_are_judged_against_the_persons_own_rest_pose():
    g = MimeGame(random.Random(4), mirror_image=False)
    rest = lambda _: face(head_yaw=15.0, head_pitch=6.0, roll=-4.0)  # the estimator is offset for this person
    acts = g.start(0.0) + run(g, 0.0, INTRO_S + 0.1, rest)
    assert g.baseline == (15.0, 6.0, -4.0)
    run(g, INTRO_S + 0.1, INTRO_S + DEMO_S + GAP_S + 0.3, rest)
    assert g.state == "wait"
    yaw, pitch, roll = MOVES[g.sequence[0]]
    # holding their rest pose is NOT the move, even when its raw numbers exceed the thresholds...
    acts = run(g, 5.0, 5.7, rest)
    assert "yes" not in kinds(acts, "sound")
    # ...the move is a change from that rest pose
    moved = lambda _: face(head_yaw=15.0 + yaw * 1.2, head_pitch=6.0 + pitch * 1.2, roll=-4.0 + roll * 1.2)
    acts = run(g, 5.7, 6.4, moved)
    assert "yes" in kinds(acts, "sound")


def test_up_and_down_are_left_out_when_the_face_is_steeply_above_or_below():
    from festival_pet.mime import STEEP_PITCH

    def game_with(face_pitch, seed):
        g = MimeGame(random.Random(seed))
        g.start(0.0, length=5)
        f = FaceObs(1, 0.0, face_pitch, 0.05, None, 0.0)
        acts = run(g, 0.0, INTRO_S + 0.1, lambda _: f)
        return g, acts

    seeds = range(30)
    # looking up at a standing person: no "look up" in the set
    ups = [game_with(-STEEP_PITCH - 8.0, s) for s in seeds]
    assert all("look up" not in g.sequence for g, _ in ups) and any("look down" in g.sequence for g, _ in ups)
    assert any(a[0] == "think" and "look up" in a[1] and "won't read" in a[1] for g, acts in ups for a in acts)
    # looking down at a seated one: no "look down"
    downs = [game_with(STEEP_PITCH + 8.0, s) for s in seeds]
    assert all("look down" not in g.sequence for g, _ in downs) and any("look up" in g.sequence for g, _ in downs)
    # eye level: everything stays
    level = [game_with(0.0, s) for s in seeds]
    assert any("look up" in g.sequence for g, _ in level) and any("look down" in g.sequence for g, _ in level)
    for g, _ in ups + downs + level:
        assert len(g.sequence) == 5 and all(a != b for a, b in zip(g.sequence, g.sequence[1:]))


def arms(left, right):
    from festival_pet.pose import Arms

    deg = {"down": 10.0, "out": 90.0, "up": 170.0}
    return Arms(0.0, deg[left], deg[right], left, right, 0.9, 50.0, {})


def test_the_arm_game_builds_the_sequence_up_like_simon():
    from festival_pet.mime import ARM_LEVEL_DEG, ARM_MOVES, ARM_ROUNDS, GAP_BETWEEN_S

    g = MimeGame(random.Random(7), mirror_image=True)
    acts = g.start(0.0, kind="arms")
    assert g.kind == "arms" and len(g.sequence) == ARM_ROUNDS and set(g.sequence) <= set(ARM_MOVES) and g.round_len == 1
    assert any(a[0] == "think" and "arms" in a[1] for a in acts)
    t = 0.0
    everything = list(acts)
    face = FaceObs(1, 5.0, -10.0, 0.02, None, 0.0)  # a small far face: enough to aim at
    for rnd in range(1, ARM_ROUNDS + 1):
        # it shows the first `rnd` moves, each DEMO_S with a short drop between
        lead = INTRO_S if rnd == 1 else PRAISE_S
        span = lead + rnd * DEMO_S + (rnd - 1) * GAP_BETWEEN_S + GAP_S + 1.0  # (each state change lands a tick late)
        acts, t0 = [], t
        while t < t0 + span:
            acts += g.tick(face, t, arms("down", "down"))
            t += 0.05
        everything += acts
        shown = [a[1] for a in acts if a[0] == "arms" and a[1] is not None]
        assert len(shown) == rnd, (rnd, shown)
        for k, (l, r, _) in enumerate(shown):
            left, right = ARM_MOVES[g.sequence[k]]
            assert (l, r) == (ARM_LEVEL_DEG[left], ARM_LEVEL_DEG[right])
        assert g.state == "wait" and g.step == 0, (rnd, g.state)
        # then they copy each, in order, as in a mirror (their right arm is its left antenna)
        for k in range(rnd):
            left, right = ARM_MOVES[g.sequence[k]]
            mine = arms(right, left)
            acts = []
            for _ in range(12):
                acts += g.tick(face, t, mine)
                t += 0.05
            everything += acts
            assert "yes" in kinds(acts, "sound"), (rnd, k)
            assert ("capture",) not in acts  # no face embedding from an arm move
        if rnd < ARM_ROUNDS:
            assert g.state == "praise" and g.round_len == rnd + 1 and g.step == 0
    assert g.state == "celebrate" and "tada" in kinds(everything, "sound")
    t0 = t
    acts = []
    while t < t0 + 3.0:
        acts += g.tick(face, t, None)
        t += 0.05
    assert not g.active and g.outcome == "won" and ("arms", None) in acts


def test_the_arm_game_shows_the_whole_round_again_when_ignored_and_reads_same_direction_when_not_mirrored():
    from festival_pet.mime import ARM_MOVES

    g = MimeGame(random.Random(8), mirror_image=False)
    g.start(0.0, kind="arms")
    face = FaceObs(1, 0.0, 0.0, 0.02, None, 0.0)
    t = 0.0
    acts = []
    while t < INTRO_S + DEMO_S + GAP_S + 0.2:
        acts += g.tick(face, t, arms("down", "down")); t += 0.05
    assert g.state == "wait"
    left, right = ARM_MOVES[g.sequence[0]]
    assert g.expected_arms() == (left, right)  # same direction: their left is its left
    # the wrong arms do nothing; the wait runs out: a huff and the move again
    acts = []
    while t < INTRO_S + DEMO_S + GAP_S + WAIT_S + 0.4:
        acts += g.tick(face, t, arms(right, left) if left != right else arms("out", "out") if left != "out" else arms("up", "up")); t += 0.05
    assert "huff" in kinds(acts, "sound") and any(a[0] == "arms" and a[1] is not None for a in acts) and g.state == "demo" and g.attempt == 1
    # arms not seen for a while: lost you
    acts = []
    while t < INTRO_S + DEMO_S + GAP_S + WAIT_S + 12.0:
        acts += g.tick(face, t, None); t += 0.05
    assert not g.active and g.outcome == "lost you" and ("arms", None) in acts
    assert g.status(t)["kind"] == "arms"


def test_the_clock_ear_counts_the_wait_down_and_clears():
    from festival_pet.mime import clock_ear

    assert clock_ear("look left") == 1 and clock_ear("look right") == 0 and clock_ear("look up") == 0 and clock_ear("tilt left") == 1
    g = MimeGame(random.Random(6))
    g.start(0.0)
    acts = run(g, 0.0, INTRO_S + DEMO_S + GAP_S + 0.3, lambda _: face())
    assert g.state == "wait"
    t0 = INTRO_S + DEMO_S + GAP_S + 0.3
    early = [a for a in run(g, t0, t0 + 0.2, lambda _: face()) if a[0] == "clock"]
    late = [a for a in run(g, t0 + 3.0, t0 + 3.2, lambda _: face()) if a[0] == "clock"]
    assert early and late and early[0][1] == clock_ear(g.sequence[0]) and early[0][2] < 0.2 < 0.8 < late[0][2]
    acts = run(g, t0 + 3.2, t0 + 4.5, lambda _: face())  # time runs out
    assert ("clock", None) in acts
