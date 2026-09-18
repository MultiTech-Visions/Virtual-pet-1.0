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
