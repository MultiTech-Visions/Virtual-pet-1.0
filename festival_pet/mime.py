"""Simon says: "do what I do", led by a robot that cannot talk. Two kinds:

head   the close-up game: it shows a head move (turn, nod, tilt) and watches the person's head
       (yaw / pitch / roll from the face landmarks) for the same one. One move at a time, 3-5 moves.
arms   the flag-signal game: the antennas are its arms (motion.arm_rad: laid back = down, horizontal
       in front = out, vertical = up) and it watches the person's arms (pose.py). It builds the
       sequence up like the old Simon toy: one move, they copy it; the same move then a second,
       they copy both in order; and so on up to five.

Copy a move and it chirps and moves on; ignore it and it shows the move again, bigger, gets huffy
on the third try, and sulks and gives up on the fourth. Finish the set and it celebrates. Every copied
head move also asks the vision thread for a face embedding, so a finished game leaves the person
enrolled from several angles.

Every shown head move is relative to where the person's face is (``center``), not to the robot's
neutral pose: a "look up" from a head that was already looking up at a standing person is a further
look up, and the gap between moves returns to the face, so the camera keeps them in view.

While the person has their turn, one antenna is a clock hand: it drops to horizontal and rises back
to straight up over the wait, so they can see the time running out. (The antennas hinge front-to-back,
so the clock reads from the side: the hand starts pointing at them.)

Pure Python: ``tick(face, now, arms)`` returns a list of things for the pet to do.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# (name, robot yaw, pitch, roll) in degrees: + yaw = its left, + pitch = down, + roll = leans to its left
# Tilt is judged from the vision thread's rotation search (vision.refine_landmarks), not the landmarks'
# eye line, which stays level when a head rolls.
MOVES: dict[str, tuple[float, float, float]] = {
    "look left": (28.0, 0.0, 0.0),
    "look right": (-28.0, 0.0, 0.0),
    "look up": (0.0, -22.0, 0.0),
    "look down": (0.0, 20.0, 0.0),
    "tilt left": (0.0, 0.0, 18.0),
    "tilt right": (0.0, 0.0, -18.0),
}
# Arm moves: (robot LEFT antenna level, robot RIGHT antenna level). Like flag signals: eight positions.
ARM_MOVES: dict[str, tuple[str, str]] = {
    "left up": ("up", "down"),
    "right up": ("down", "up"),
    "both up": ("up", "up"),
    "left out": ("out", "down"),
    "right out": ("down", "out"),
    "both out": ("out", "out"),
    "left up, right out": ("up", "out"),
    "left out, right up": ("out", "up"),
}
ARM_LEVEL_DEG = {"down": 0.0, "out": 90.0, "up": 180.0}
YAW_MATCH, PITCH_MATCH, ROLL_MATCH = 12.0, 8.0, 10.0  # what counts as "they did it", as a change from THEIR neutral
# Up and down depend on where they stand. A robot looking up at a standing person sees them already looking
# down at it: a "look up" from there is a small change of a face seen from below, and the camera misses it.
# Looking down at a seated person, the same for "look down". Past this much gaze pitch the move is left out.
STEEP_PITCH = 10.0
MATCH_HOLD_S = 0.4  # the pose must hold this long (three or four detections)
INTRO_S = 1.8
DEMO_S = 1.6  # showing the move
GAP_S = 0.5  # back to neutral before watching
GAP_BETWEEN_S = 0.4  # between two shown moves of one arm round: arms drop, so the next one reads as new
WAIT_S = 4.0  # how long they get to copy each move
PRAISE_S = 1.0  # "yes!" before the next move: the camera settles before it moves again
CELEBRATE_S = 2.5
MAX_ATTEMPTS = 3  # demos of one move before it gives up
LOST_FACE_S = 5.0
ARM_ROUNDS = 5  # the arm game builds up to this many moves


def clock_ear(move: str) -> int:
    """Which antenna counts down for a move: the one on the move's side (0 right, 1 left); up/down use the right."""
    if move in ARM_MOVES:
        return 0
    yaw, _, roll = MOVES[move]
    side = yaw if yaw != 0 else roll
    return 1 if side > 0 else 0


@dataclass
class MimeGame:
    rng: random.Random = field(default_factory=random.Random)
    mirror_image: bool = True  # they copy as in a mirror: its left is their right
    kind: str = "head"  # "head" (one move at a time, from the face) or "arms" (flag signals, built up like Simon)
    state: str = "idle"  # idle | intro | demo | gap | wait | praise | celebrate | done
    center: tuple[float, float] = (0.0, 0.0)  # world yaw / pitch of the person's face; every move is shown from here
    baseline: tuple[float, float, float] = (0.0, 0.0, 0.0)  # their head yaw / pitch / roll at rest, measured in the intro
    _intro_poses: list = field(default_factory=list)
    outcome: str = ""  # after done: "won", "gave up", "lost you", "stopped"
    sequence: list[str] = field(default_factory=list)
    round_len: int = 0  # how many of the sequence are in play (arms: grows by one each round; head: all of them)
    step: int = 0  # the move being waited for
    attempt: int = 0
    _demo_queue: list[int] = field(default_factory=list)  # sequence indices still to show this round
    _until: float = 0.0
    _match_since: float = 0.0
    _last_face: float = 0.0
    _gain: float = 1.0  # demo amplitude grows with each repeat (head moves)

    @property
    def active(self) -> bool:
        return self.state not in ("idle", "done")

    @property
    def cumulative(self) -> bool:
        return self.kind == "arms"

    def start(self, now: float, length: int | None = None, kind: str | None = None) -> list[tuple]:
        if kind is not None:
            self.kind = kind
        if self.kind not in ("head", "arms"):
            raise ValueError(f"unknown Simon says kind '{self.kind}'")
        n = length if length is not None else (ARM_ROUNDS if self.cumulative else self.rng.randint(3, 5))
        self.sequence = self._sequence(n, list(ARM_MOVES if self.cumulative else MOVES))
        self.round_len = 1 if self.cumulative else n
        self.step, self.attempt, self._gain = 0, 0, 1.0
        self.state, self._until, self._last_face, self.outcome = "intro", now + INTRO_S, now, ""
        self._intro_poses, self.baseline, self._demo_queue = [], (0.0, 0.0, 0.0), []
        what = "arms" if self.cumulative else "head"
        return [("think", f"Simon says ({what})! {n} moves: {', '.join(self.sequence)}"), ("sound", "mime_start"), ("gesture", "perk")]

    def _sequence(self, n: int, moves: list[str]) -> list[str]:
        seq: list[str] = []
        while len(seq) < n:  # random, never the same move twice in a row
            m = self.rng.choice(moves)
            if not seq or m != seq[-1]:
                seq.append(m)
        return seq

    def allowed_moves(self) -> list[str]:
        """The moves that can be seen from where the person's face is (see STEEP_PITCH)."""
        if self.cumulative:
            return list(ARM_MOVES)
        pitch = self.center[1]  # + = the robot looks down at them
        return [m for m in MOVES if not ((m == "look up" and pitch < -STEEP_PITCH) or (m == "look down" and pitch > STEEP_PITCH))]

    def _fit_sequence(self) -> str | None:
        """After the intro, once the face's height is known: redraw the moves it cannot see from here."""
        allowed = self.allowed_moves()
        dropped = sorted(set(self.sequence) - set(allowed))
        if not dropped:
            return None
        self.sequence = self._sequence(len(self.sequence), allowed)
        return f"{' / '.join(dropped)} won't read from this angle, new set: {', '.join(self.sequence)}"

    def stop(self, now: float, outcome: str = "stopped") -> list[tuple]:
        was = self.active
        self.state, self.outcome = "done", outcome
        return [("hold", None), ("arms", None), ("clock", None), ("sound", "mime_end")] if was else []

    # ------------------------------------------------------------------ showing
    def _plan_demos(self) -> None:
        self._demo_queue = list(range(self.round_len)) if self.cumulative else [self.step]

    def _demo(self, now: float) -> list[tuple]:
        """Show the next queued move."""
        i = self._demo_queue.pop(0)
        move = self.sequence[i]
        self.state, self._until = "demo", now + DEMO_S
        if self.cumulative:
            left, right = ARM_MOVES[move]
            return [("arms", (ARM_LEVEL_DEG[left], ARM_LEVEL_DEG[right], DEMO_S)), ("sound", "mime_cue")]
        yaw, pitch, roll = MOVES[move]
        g = self._gain
        cy, cp = self.center
        return [("hold", (cy + yaw * g, cp + pitch * g, roll * g, DEMO_S)), ("sound", "mime_cue")]

    def _expected(self) -> tuple[float, float, float]:
        yaw, pitch, roll = MOVES[self.sequence[self.step]]
        s = -1.0 if self.mirror_image else 1.0
        return yaw * s, pitch, roll * s

    def expected_arms(self) -> tuple[str, str]:
        """(person's left, person's right) levels for the move being waited for."""
        left, right = ARM_MOVES[self.sequence[self.step]]  # robot left, robot right
        return (right, left) if self.mirror_image else (left, right)

    def _matches(self, face, arms) -> bool:
        if self.cumulative:
            if arms is None:
                return False
            el, er = self.expected_arms()
            return arms.left == el and arms.right == er
        if face is None:
            return False
        ey, ep, er = self._expected()
        by, bp, br = self.baseline  # the estimate is rough and offset per person: a move is a change from their rest
        hy, hp, hr = face.head_yaw_deg - by, face.head_pitch_deg - bp, face.roll_deg - br
        if ey != 0:
            return hy * ey > 0 and abs(hy) >= YAW_MATCH and abs(hp) < PITCH_MATCH
        if ep != 0:
            return hp * ep > 0 and abs(hp) >= PITCH_MATCH and abs(hy) < YAW_MATCH
        return hr * er > 0 and abs(hr) >= ROLL_MATCH

    # ------------------------------------------------------------------ the game
    def tick(self, face, now: float, arms=None) -> list[tuple]:
        """``face`` is the FaceObs in view (or None; for the arm game, a body sighting will do for aiming),
        ``arms`` the pose.Arms read (or None). Returns pet actions."""
        if not self.active:
            return []
        seen = arms if self.cumulative else face
        if face is not None and self.state in ("intro", "wait", "praise", "celebrate"):
            self.center = (face.yaw_deg, face.pitch_deg)  # only while the head is aimed at them, not mid-move
        if seen is not None:
            self._last_face = now
            if self.state == "intro" and not self.cumulative:
                self._intro_poses.append((face.head_yaw_deg, face.head_pitch_deg, face.roll_deg))
        elif now - self._last_face > LOST_FACE_S:
            self.state, self.outcome = "done", "lost you"
            return [("hold", None), ("arms", None), ("clock", None), ("think", "Simon says: where did you go?"), ("sound", "confused"), ("gesture", "search")]

        if self.state == "intro":
            if now >= self._until:
                out: list[tuple] = []
                if not self.cumulative:
                    if len(self._intro_poses) >= 3:
                        cols = list(zip(*self._intro_poses))
                        self.baseline = tuple(sorted(c)[len(c) // 2] for c in cols)  # the median: a glance away does not skew it
                    note = self._fit_sequence()
                    if note:
                        out.append(("think", note))
                self._plan_demos()
                return out + self._demo(now)
            return []
        if self.state == "demo":
            if now >= self._until:
                self._match_since = 0.0
                if self._demo_queue:  # more of the round to show: arms down for a beat, then the next
                    self.state, self._until = "gap", now + GAP_BETWEEN_S
                    return [("arms", None)]
                self.state, self._until = "gap", now + GAP_S
                if self.cumulative:
                    return [("arms", None)]
                return [("hold", (self.center[0], self.center[1], 0.0, GAP_S))]  # back to their face
            return []
        if self.state == "gap":
            if now >= self._until:
                if self._demo_queue:
                    return self._demo(now)
                self.state, self._until, self._match_since = "wait", now + WAIT_S, 0.0
                return [("hold", None)]
            return []
        if self.state == "wait":
            # the clock hand: 0 = horizontal (just started), 1 = straight up (time is up)
            tick_out: list[tuple] = [("clock", clock_ear(self.sequence[self.step]), 1.0 - max(0.0, self._until - now) / WAIT_S)]
            if self._matches(face, arms):
                if self._match_since == 0.0:
                    self._match_since = now
                if now - self._match_since >= MATCH_HOLD_S:
                    return [("clock", None)] + self._copied(now)
            else:
                self._match_since = 0.0
            if now >= self._until:
                return [("clock", None)] + self._ignored(now)
            return tick_out
        if self.state == "praise":
            if now >= self._until:
                self._plan_demos()
                return self._demo(now)
            return []
        if self.state == "celebrate":
            if now >= self._until:
                self.state, self.outcome = "done", "won"
                return [("arms", None), ("sound", "mime_end")]
            return []
        return []

    def _copied(self, now: float) -> list[tuple]:
        # praise with the antennas, not the head: a nod here moved the camera and lost their face
        move = self.sequence[self.step]
        out: list[tuple] = [("think", f"yes! they did '{move}'"), ("sound", "yes"), ("gesture", "perk")]
        if not self.cumulative:
            out.append(("capture",))
        self.step += 1
        self.attempt, self._gain, self._match_since = 0, 1.0, 0.0
        if self.cumulative and self.step < self.round_len:
            self._until = now + WAIT_S  # the rest of this round, in order
            return out
        if self.round_len >= len(self.sequence) and self.step >= self.round_len:
            self.state, self._until = "celebrate", now + CELEBRATE_S
            out += [("think", "they did the whole thing! ta-da!"), ("sound", "tada"), ("gesture", "bounce")]
            return out
        if self.cumulative:
            self.round_len += 1
            self.step = 0
            out.append(("think", f"round {self.round_len}: {', '.join(self.sequence[:self.round_len])}"))
        self.state, self._until = "praise", now + PRAISE_S
        return out

    def _ignored(self, now: float) -> list[tuple]:
        self.attempt += 1
        move = self.sequence[self.step]
        if self.attempt >= MAX_ATTEMPTS:
            self.state, self.outcome = "done", "gave up"
            return [("hold", None), ("arms", None), ("think", f"they never did '{move}'... fine. FINE."), ("sound", "sad"), ("gesture", "droop")]
        self._gain = 1.0 + 0.35 * self.attempt
        out: list[tuple] = []
        if self.attempt == 1:
            out += [("think", f"no? '{move}'. like THIS"), ("sound", "huff")]
        else:
            out += [("think", f"'{move}'!! come ON"), ("sound", "annoyed"), ("gesture", "shake")]
        if self.cumulative:
            self.step = 0  # the whole round again, from the top
        self._plan_demos()
        out += self._demo(now)
        return out

    def status(self, now: float) -> dict:
        in_play = self.active and self.step < len(self.sequence)
        return {
            "active": self.active, "kind": self.kind, "state": self.state, "outcome": self.outcome,
            "step": self.step + 1 if in_play else len(self.sequence),
            "of": self.round_len if self.cumulative else len(self.sequence), "round": self.round_len if self.cumulative else None,
            "move": self.sequence[self.step] if in_play else None,
            "expect_arms": list(self.expected_arms()) if in_play and self.cumulative else None,
            "attempt": self.attempt + 1, "time_left_s": round(max(0.0, self._until - now), 1) if self.state == "wait" else None,
            "baseline": [round(v, 1) for v in self.baseline],
        }
