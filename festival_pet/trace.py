"""The black box: a rolling, downloadable record of what the pet was actually doing.

Debugging this robot from a chat window is guesswork without it. "It lost me", "it looked at the
wall", "it didn't see my peace sign" are all symptoms of numbers that nobody can see: where the face
detector last put somebody, whether the pose model returned arms that frame, what the gaze target was
versus where the head actually went, which spot got blacklisted. The log has sentences in it; this has
the numbers, every tick, so a session can be handed over as one file and read rather than imagined.

Design:
- A fixed-size ring, so it runs for four days at a festival without ever filling the disk.
- Rows are plain dicts of numbers and short strings, written out as JSON Lines: one self-contained
  object per line, greppable, and readable a chunk at a time without parsing the whole file.
- Sampled at a modest rate (the control loop is 50 Hz; ten rows a second is plenty to see a dropout)
  but EVENTS — thoughts, actions, marks — go in the moment they happen, at full resolution.
- ``mark`` writes a labelled line, so "it did the thing, NOW" can be found in a 50,000-line file.

Pure Python, no robot imports, fake-clock friendly.
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Iterator

TRACE_HZ = 10.0  # rows per second of sampled state
TRACE_ROWS = 40_000  # about an hour at 10 Hz, plus room for events; a few tens of MB at worst
EVENT_KINDS = ("mark", "think", "action", "note")


class Recorder:
    """A ring of rows. ``tick`` is rate-limited; ``event`` is not."""

    def __init__(self, capacity: int = TRACE_ROWS, hz: float = TRACE_HZ) -> None:
        self.rows: deque[dict] = deque(maxlen=capacity)
        self.period = 1.0 / hz
        self.enabled = True  # on by default: the whole point is to already be running when it goes wrong
        self._next = 0.0
        self.started = time.time()
        self.dropped = 0  # rows pushed out of the far end of the ring
        self.header: dict = {}
        self.marks = 0

    # ------------------------------------------------------------------ writing
    def _push(self, row: dict) -> None:
        if len(self.rows) == self.rows.maxlen:
            self.dropped += 1
        self.rows.append(row)

    def due(self, now: float) -> bool:
        return self.enabled and now >= self._next

    def tick(self, now: float, row: dict) -> None:
        """One sampled row of state. Call every control tick; it writes at its own rate."""
        if not self.due(now):
            return
        # Advance from the deadline, not from now, or a 50 Hz loop sampling on a 0.1 s period lands at
        # 0.12 s and the file quietly runs at eight rows a second instead of ten.
        self._next += self.period
        if self._next <= now:
            self._next = now + self.period  # a real gap (a blocking sleep move): pick up from here
        self._push({"t": round(now, 3), "k": "s", **row})

    def event(self, now: float, kind: str, text: str, **extra) -> None:
        """Something that happened, recorded at the instant it happened."""
        if not self.enabled:
            return
        if kind not in EVENT_KINDS:
            raise KeyError(f"unknown trace event kind '{kind}'. Known: {', '.join(EVENT_KINDS)}")
        self._push({"t": round(now, 3), "k": kind, "text": text, **extra})

    def mark(self, now: float, text: str = "") -> int:
        """"It just did the thing" — a labelled line to find in the file. Returns the mark's number."""
        self.marks += 1
        self.event(now, "mark", text or f"mark {self.marks}", n=self.marks)
        return self.marks

    def clear(self, now: float) -> None:
        self.rows.clear()
        self.dropped, self.marks, self.started = 0, 0, now

    # ------------------------------------------------------------------ reading
    def stats(self, now: float) -> dict:
        span = 0.0 if not self.rows else self.rows[-1]["t"] - self.rows[0]["t"]
        return {"on": self.enabled, "rows": len(self.rows), "capacity": self.rows.maxlen,
                "covers_s": round(span, 1), "dropped": self.dropped, "marks": self.marks,
                "bytes_est": len(self.rows) * 260, "since_s": round(now - self.started, 1)}

    def lines(self, now: float, last_s: float | None = None) -> Iterator[str]:
        """The file, as JSON Lines. The first line is a header; the rest are rows, oldest first.

        ``last_s`` keeps only the tail — handy when the interesting thing happened a minute ago and the
        ring holds an hour of pottering about in front of it.
        """
        # Sorted by time on the way out: events are written when they are noticed, which can be a tick
        # or two after they happened, so the ring's own order is nearly but not quite chronological.
        ordered = sorted(self.rows, key=lambda r: r["t"])
        cutoff = -1e18 if last_s is None else (ordered[-1]["t"] if ordered else now) - last_s
        kept = [r for r in ordered if r["t"] >= cutoff]
        yield json.dumps({"k": "header", "written_at": now, **self.header, **self.stats(now),
                          "rows_in_file": len(kept), "hz": round(1.0 / self.period, 2)}, default=str)
        for row in kept:
            yield json.dumps(row, default=str)
