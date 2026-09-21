"""The black box: a rolling record that can be handed over as one file."""

import json

from festival_pet.trace import Recorder


def test_it_samples_at_its_own_rate_and_never_grows_past_the_ring():
    r = Recorder(capacity=50, hz=10.0)
    t = 100.0
    for _ in range(500):  # 10 s of a 50 Hz control loop
        r.tick(t, {"st": "IDLE"})
        t += 0.02
    assert len(r.rows) == 50 and r.dropped == 50  # 100 sampled rows, half rolled off the end
    stamps = [row["t"] for row in r.rows]
    assert all(0.09 < b - a < 0.11 for a, b in zip(stamps, stamps[1:]))  # ten a second, not fifty
    assert r.stats(t)["covers_s"] > 4.0


def test_events_are_recorded_the_moment_they_happen_and_marks_are_findable():
    r = Recorder(capacity=100, hz=10.0)
    r.tick(100.0, {"st": "IDLE"})
    for k in range(5):  # five events inside one sample period: none of them is dropped
        r.event(100.01 + k * 0.001, "action", f"gesture:nod{k}")
    n = r.mark(100.02, "it looked at the wall")
    r.tick(100.02, {"st": "SEARCHING"})  # too soon: not sampled
    assert n == 1 and r.stats(100.1)["marks"] == 1
    kinds = [row["k"] for row in r.rows]
    assert kinds.count("action") == 5 and kinds.count("s") == 1
    marks = [row for row in r.rows if row["k"] == "mark"]
    assert marks[0]["text"] == "it looked at the wall" and marks[0]["n"] == 1
    try:
        r.event(100.0, "vibes", "hello")
    except KeyError:
        return
    raise AssertionError("an unknown event kind should be an error, not a quiet miss")


def test_the_file_is_a_header_then_rows_in_time_order():
    r = Recorder(capacity=100, hz=10.0)
    r.header = {"build": {"version": "9.9.9"}}
    t = 100.0
    for k in range(30):
        r.tick(t, {"st": "IDLE", "n": k})
        r.event(t - 0.05, "think", f"thought {k}")  # noticed late, as the real ones are
        t += 0.1
    lines = list(r.lines(t))
    head = json.loads(lines[0])
    assert head["k"] == "header" and head["build"]["version"] == "9.9.9" and head["hz"] == 10.0
    rows = [json.loads(x) for x in lines[1:]]
    assert head["rows_in_file"] == len(rows) == 60
    assert [x["t"] for x in rows] == sorted(x["t"] for x in rows), "not in time order"
    # ...and the tail can be taken on its own, for when the interesting bit just happened
    tail = [json.loads(x) for x in r.lines(t, last_s=1.0)][1:]
    assert 0 < len(tail) < len(rows) and min(x["t"] for x in tail) >= t - 1.2


def test_recording_can_be_turned_off_and_cleared():
    r = Recorder(capacity=10, hz=10.0)
    r.tick(100.0, {"st": "IDLE"})
    r.enabled = False
    r.tick(101.0, {"st": "IDLE"})
    r.event(101.0, "action", "gesture:nod")
    assert len(r.rows) == 1 and not r.stats(101.0)["on"]
    r.enabled = True
    r.tick(102.0, {"st": "IDLE"})
    assert len(r.rows) == 2
    r.clear(103.0)
    assert not r.rows and r.stats(103.0)["since_s"] == 0.0
