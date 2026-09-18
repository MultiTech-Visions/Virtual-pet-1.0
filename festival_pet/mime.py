"""Simon says: "do what I do", led by a robot that cannot talk.

It picks a fresh random sequence of head moves each game, shows one, then watches the
person's head (yaw / pitch / roll estimated from the face landmarks) for the same move.
Copy it and it chirps and moves on; ignore it and it shows the move again, bigger, gets
huffy on the third try, and sulks and gives up on the fourth. Finish the set and it
celebrates. Every copied move also asks the vision thread for a face embedding, so a
finished game leaves the person enrolled from several angles.

Every shown move is relative to where the person's face is (``center``), not to the
robot's neutral pose: a "look up" from a head that was already looking up at a standing
person is a further look up, and the gap between moves returns to the face, so the camera
keeps them in view and the move reads as what it is.

Pure Python: ``tick(face, now)`` returns a list of things for the pet to do.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# (name, robot yaw, pitch, roll) in degrees: + yaw = its left, + pitch = down, + roll = leans to its left
MOVES: dict[str, tuple[float, float, float]] = {
    "look left": (28.0, 0.0, 0.0),
    "look right": (-28.0, 0.0, 0.0),
    "look up": (0.0, -22.0, 0.0),
    "look down": (0.0, 20.0, 0.0),
    "tilt left": (0.0, 0.0, 18.0),
    "tilt right": (0.0, 0.0, -18.0),
}
YAW_MATCH, PITCH_MATCH, ROLL_MATCH = 14.0, 10.0, 10.0  # what counts as "they did it"
MATCH_HOLD_S = 0.4  # the pose must hold this long (three or four detections)
INTRO_S = 1.8
DEMO_S = 1.6  # showing the move
GAP_S = 0.5  # back to neutral before watching
WAIT_S = 4.0  # how long they get to copy it
PRAISE_S = 1.0  # "yes!" before the next move: the camera settles before it moves again
CELEBRATE_S = 2.5
MAX_ATTEMPTS = 3  # demos of one move before it gives up
LOST_FACE_S = 5.0


@dataclass
class MimeGame:
    rng: random.Random = field(default_factory=random.Random)
    mirror_image: bool = True  # they copy as in a mirror: its left is their right
    state: str = "idle"  # idle | intro | demo | gap | wait | praise | celebrate | done
    center: tuple[float, float] = (0.0, 0.0)  # world yaw / pitch of the person's face; every move is shown from here
    outcome: str = ""  # after done: "won", "gave up", "lost you", "stopped"
    sequence: list[str] = field(default_factory=list)
    step: int = 0
    attempt: int = 0
    _until: float = 0.0
    _match_since: float = 0.0
    _last_face: float = 0.0
    _gain: float = 1.0  # demo amplitude grows with each repeat

    @property
    def active(self) -> bool:
        return self.state not in ("idle", "done")

    def start(self, now: float, length: int | None = None) -> list[tuple]:
        n = length if length is not None else self.rng.randint(3, 5)
        moves = list(MOVES)
        seq: list[str] = []
        while len(seq) < n:  # random, never the same move twice in a row
            m = self.rng.choice(moves)
            if not seq or m != seq[-1]:
                seq.append(m)
        self.sequence, self.step, self.attempt, self._gain = seq, 0, 0, 1.0
        self.state, self._until, self._last_face, self.outcome = "intro", now + INTRO_S, now, ""
        return [("think", f"Simon says! {n} moves: {', '.join(seq)}"), ("sound", "mime_start"), ("gesture", "perk")]

    def stop(self, now: float, outcome: str = "stopped") -> list[tuple]:
        was = self.active
        self.state, self.outcome = "done", outcome
        return [("hold", None), ("sound", "mime_end")] if was else []

    def _demo(self, now: float) -> list[tuple]:
        yaw, pitch, roll = MOVES[self.sequence[self.step]]
        g = self._gain
        cy, cp = self.center
        self.state, self._until = "demo", now + DEMO_S
        return [("hold", (cy + yaw * g, cp + pitch * g, roll * g, DEMO_S)), ("sound", "mime_cue")]

    def _expected(self) -> tuple[float, float, float]:
        yaw, pitch, roll = MOVES[self.sequence[self.step]]
        s = -1.0 if self.mirror_image else 1.0
        return yaw * s, pitch, roll * s

    def _matches(self, face) -> bool:
        ey, ep, er = self._expected()
        hy, hp, hr = face.head_yaw_deg, face.head_pitch_deg, face.roll_deg
        if ey != 0:
            return hy * ey > 0 and abs(hy) >= YAW_MATCH and abs(hp) < PITCH_MATCH
        if ep != 0:
            return hp * ep > 0 and abs(hp) >= PITCH_MATCH and abs(hy) < YAW_MATCH
        return hr * er > 0 and abs(hr) >= ROLL_MATCH

    def tick(self, face, now: float) -> list[tuple]:
        """``face`` is the FaceObs in view (or None). Returns pet actions."""
        if not self.active:
            return []
        if face is not None:
            self._last_face = now
            if self.state in ("intro", "wait", "praise", "celebrate"):
                self.center = (face.yaw_deg, face.pitch_deg)  # only while the head is aimed at them, not mid-move
        elif now - self._last_face > LOST_FACE_S:
            self.state, self.outcome = "done", "lost you"
            return [("hold", None), ("think", "Simon says: where did you go?"), ("sound", "confused"), ("gesture", "search")]

        if self.state == "intro":
            if now >= self._until:
                return self._demo(now)
            return []
        if self.state == "demo":
            if now >= self._until:
                self.state, self._until, self._match_since = "gap", now + GAP_S, 0.0
                return [("hold", (self.center[0], self.center[1], 0.0, GAP_S))]  # back to their face
            return []
        if self.state == "gap":
            if now >= self._until:
                self.state, self._until, self._match_since = "wait", now + WAIT_S, 0.0
                return [("hold", None)]
            return []
        if self.state == "wait":
            if face is not None and self._matches(face):
                if self._match_since == 0.0:
                    self._match_since = now
                if now - self._match_since >= MATCH_HOLD_S:
                    return self._copied(now)
            else:
                self._match_since = 0.0
            if now >= self._until:
                return self._ignored(now)
            return []
        if self.state == "praise":
            if now >= self._until:
                return self._demo(now)
            return []
        if self.state == "celebrate":
            if now >= self._until:
                self.state, self.outcome = "done", "won"
                return [("sound", "mime_end")]
            return []
        return []

    def _copied(self, now: float) -> list[tuple]:
        # praise with the antennas, not the head: a nod here moved the camera and lost their face
        out: list[tuple] = [("think", f"yes! they did '{self.sequence[self.step]}'"), ("sound", "yes"), ("gesture", "perk"), ("capture",)]
        self.step += 1
        self.attempt, self._gain = 0, 1.0
        if self.step >= len(self.sequence):
            self.state, self._until = "celebrate", now + CELEBRATE_S
            out += [("think", "they did the whole thing! ta-da!"), ("sound", "tada"), ("gesture", "bounce")]
            return out
        self.state, self._until = "praise", now + PRAISE_S
        return out

    def _ignored(self, now: float) -> list[tuple]:
        self.attempt += 1
        move = self.sequence[self.step]
        if self.attempt >= MAX_ATTEMPTS:
            self.state, self.outcome = "done", "gave up"
            return [("hold", None), ("think", f"they never did '{move}'... fine. FINE."), ("sound", "sad"), ("gesture", "droop")]
        self._gain = 1.0 + 0.35 * self.attempt
        out: list[tuple] = []
        if self.attempt == 1:
            out += [("think", f"no? '{move}'. like THIS"), ("sound", "huff")]
        else:
            out += [("think", f"'{move}'!! come ON"), ("sound", "annoyed"), ("gesture", "shake")]
        out += self._demo(now)
        return out

    def status(self, now: float) -> dict:
        return {
            "active": self.active, "state": self.state, "outcome": self.outcome,
            "step": self.step + 1 if self.active and self.step < len(self.sequence) else len(self.sequence),
            "of": len(self.sequence), "move": self.sequence[self.step] if self.active and self.step < len(self.sequence) else None,
            "attempt": self.attempt + 1, "time_left_s": round(max(0.0, self._until - now), 1) if self.state == "wait" else None,
        }
