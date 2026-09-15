"""The pet's brain: a small state machine plus a mood model.

Pure Python, no robot imports, driven by ``Behavior.tick(obs, now)`` so it can
be unit-tested with a fake clock. Each tick returns a list of ``Action``
requests (sounds, gestures, library moves) that the robot layer executes.

States
    SLEEPING  head down, antennas down. Wakes on touch, pickup, a persistent face, or a loud sound.
    IDLE      breathing, occasional look-around, gets lonely, eventually dozes off.
    ENGAGED   locked onto a person; greets by relationship tier, reacts over time.
    SEARCHING lost the face; looks where it was, then gives up.
    HELD      picked up; snuggles / purrs; goes dizzy if shaken.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from festival_pet.memory import FaceMemory, Person

State = Literal["SLEEPING", "WAKING", "IDLE", "ENGAGED", "SEARCHING", "HELD"]


@dataclass
class FaceObs:
    """One tracked face for this tick (from the vision thread)."""

    track_id: int
    yaw_deg: float  # absolute world yaw the head should take to look at it
    pitch_deg: float
    area_frac: float  # bbox area / frame area, a proxy for closeness
    person: Person | None  # None while unknown / not yet embedded
    similarity: float


@dataclass
class Observation:
    """Everything the senses report for one tick."""

    face: FaceObs | None = None
    held: bool = False
    shaken: bool = False
    touched: bool = False  # antenna pushed this tick (edge, not level)
    loud_yaw_deg: float | None = None  # direction of a sudden loud sound, if any


@dataclass
class Action:
    """A request for the robot layer."""

    kind: Literal["sound", "gesture", "move", "wake", "sleep"]
    name: str
    priority: int = 1  # higher preempts lower for gestures/moves


@dataclass
class Mood:
    energy: float = 0.8  # 0 exhausted .. 1 bouncy
    social: float = 0.5  # 0 lonely .. 1 fulfilled

    def clamp(self) -> None:
        self.energy = min(1.0, max(0.0, self.energy))
        self.social = min(1.0, max(0.0, self.social))


@dataclass
class Timers:
    """Tunable timing, in seconds."""

    face_lost_grace: float = 2.5
    search_duration: float = 5.0
    lonely_after: float = 90.0
    lonely_repeat: float = 45.0
    sleep_after: float = 420.0
    wake_face_hold: float = 1.5
    react_min: float = 4.0
    react_max: float = 11.0
    idle_glance_min: float = 3.0
    idle_glance_max: float = 8.0
    held_settle: float = 0.6
    held_purr_min: float = 5.0
    held_purr_max: float = 12.0
    touch_cooldown: float = 1.2
    engaged_sad_if_over: float = 20.0


@dataclass
class Behavior:
    memory: FaceMemory
    timers: Timers = field(default_factory=Timers)
    rng: random.Random = field(default_factory=random.Random)
    state: State = "SLEEPING"
    mood: Mood = field(default_factory=Mood)

    # bookkeeping
    _state_since: float = 0.0
    _last_interaction: float = 0.0
    _last_face_time: float = 0.0
    _face_first_seen: float = 0.0
    _last_lonely: float = -1e9
    _next_react: float = 0.0
    _next_glance: float = 0.0
    _next_purr: float = 0.0
    _last_touch: float = -1e9
    _engaged_track: int | None = None
    _engaged_person: Person | None = None
    _engaged_since: float = 0.0
    _last_seen_yaw: float = 0.0
    _last_seen_pitch: float = 0.0
    _greeted_track: int | None = None
    _prev_held: bool = False
    _dizzy_until: float = 0.0

    # gaze the motion layer should aim for (None = free to idle-drift)
    gaze: tuple[float, float] | None = None

    def start(self, now: float, awake: bool) -> None:
        """Call once before ticking. ``awake`` reflects whether the daemon already woke the robot."""
        self.state = "IDLE" if awake else "SLEEPING"
        self._state_since = now
        self._last_interaction = now
        self._last_face_time = now
        self._next_glance = now + self.rng.uniform(self.timers.idle_glance_min, self.timers.idle_glance_max)

    # ------------------------------------------------------------------ helpers
    def _enter(self, state: State, now: float) -> None:
        self.state = state
        self._state_since = now

    def _person_greeting(self, person: Person | None) -> tuple[str, str, str]:
        """(sound, gesture, library move) for meeting someone."""
        if person is None:
            return "hello_new", "perk", "curious1"
        tier = person.tier()
        if tier == "bestie":
            return "hello_bestie", "bounce", "loving1"
        if tier == "friend":
            return "hello_friend", "wiggle", "welcoming1"
        return "hello_new", "perk", "curious1"

    # ------------------------------------------------------------------ main step
    def tick(self, obs: Observation, now: float, dt: float) -> list[Action]:
        actions: list[Action] = []
        t = self.timers

        # -- mood drift ----------------------------------------------------------------
        if self.state == "SLEEPING":
            self.mood.energy += dt * 0.01
        else:
            self.mood.energy -= dt * 0.0006
        if self.state in ("ENGAGED", "HELD"):
            self.mood.social += dt * 0.01
        else:
            self.mood.social -= dt * 0.0015
        self.mood.clamp()

        # -- global interrupts: pickup and touch work in every awake state ---------------
        pickup_edge = obs.held and not self._prev_held
        setdown_edge = self._prev_held and not obs.held
        self._prev_held = obs.held

        if obs.touched and now - self._last_touch > t.touch_cooldown:
            self._last_touch = now
            self._last_interaction = now
            if self.state == "SLEEPING":
                actions.append(Action("wake", "touch", 5))
                self._enter("WAKING", now)
            else:
                actions.append(Action("sound", self.rng.choice(["giggle", "happy", "purr"]), 2))
                actions.append(Action("gesture", "wiggle", 2))
                self.mood.social += 0.05
                if self._engaged_person is not None:
                    self.memory.add_pet(self._engaged_person)

        if pickup_edge:
            self._last_interaction = now
            if self.state == "SLEEPING":
                actions.append(Action("wake", "pickup", 5))
            actions.append(Action("sound", "surprised", 4))
            actions.append(Action("gesture", "startle", 4))
            self._next_purr = now + t.held_settle + self.rng.uniform(1.0, 2.5)
            if self._engaged_person is not None:
                self.memory.add_hold(self._engaged_person)
            self._enter("HELD", now)

        # -- per-state ------------------------------------------------------------------
        if self.state == "SLEEPING":
            self.gaze = None
            face_persistent = obs.face is not None and obs.face.area_frac > 0.01
            if face_persistent:
                if self._face_first_seen == 0.0:
                    self._face_first_seen = now
                elif now - self._face_first_seen > t.wake_face_hold:
                    actions.append(Action("wake", "face", 5))
                    self._enter("WAKING", now)
            else:
                self._face_first_seen = 0.0
            if obs.loud_yaw_deg is not None and self.mood.energy > 0.3:
                actions.append(Action("wake", "sound", 5))
                self._enter("WAKING", now)

        elif self.state == "WAKING":
            # The robot layer runs the wake move; we just wait it out.
            if now - self._state_since > 2.5:
                self._last_interaction = now
                self._enter("IDLE", now)
                self._next_glance = now + self.rng.uniform(1.0, 3.0)

        elif self.state == "HELD":
            self.gaze = None
            if obs.shaken and now > self._dizzy_until:
                self._dizzy_until = now + 3.0
                actions.append(Action("sound", "dizzy", 4))
                actions.append(Action("gesture", "dizzy", 4))
                self._next_purr = now + 4.0
            elif now > self._dizzy_until and now >= self._next_purr:
                actions.append(Action("sound", self.rng.choice(["purr", "content", "content"]), 2))
                actions.append(Action("gesture", "snuggle", 2))
                self._next_purr = now + self.rng.uniform(t.held_purr_min, t.held_purr_max)
            if setdown_edge:
                actions.append(Action("sound", "happy", 3))
                actions.append(Action("gesture", "shake_off", 3))
                self._last_interaction = now
                self._enter("IDLE", now)

        elif self.state in ("IDLE", "SEARCHING", "ENGAGED"):
            face = obs.face
            if face is not None:
                self._last_face_time = now
                self._last_seen_yaw, self._last_seen_pitch = face.yaw_deg, face.pitch_deg
                self.gaze = (face.yaw_deg, face.pitch_deg)

                if self.state != "ENGAGED" or face.track_id != self._engaged_track:
                    # New engagement (or switched to a different person).
                    self._engaged_track = face.track_id
                    self._engaged_person = face.person
                    self._engaged_since = now
                    self._greeted_track = None
                    self._enter("ENGAGED", now)
                    self._next_react = now + self.rng.uniform(t.react_min, t.react_max)

                # Greet once we know who it is (or have given up identifying them).
                if self._greeted_track != face.track_id and (face.person is not None or now - self._engaged_since > 1.5):
                    self._greeted_track = face.track_id
                    self._engaged_person = face.person
                    if face.person is not None:
                        self.memory.sighted(face.person, now)
                    sound, gesture, move = self._person_greeting(face.person)
                    actions.append(Action("sound", sound, 3))
                    if face.person is not None and face.person.tier() != "acquaintance" and self.mood.energy > 0.35:
                        actions.append(Action("move", move, 3))
                    else:
                        actions.append(Action("gesture", gesture, 3))
                    self.mood.social += 0.1
                elif face.person is not None and self._engaged_person is None:
                    # Identified mid-engagement after a generic greeting: a small "oh it's you" for friends.
                    self._engaged_person = face.person
                    self.memory.sighted(face.person, now)
                    if face.person.tier() != "acquaintance":
                        actions.append(Action("sound", "hello_friend", 2))
                        actions.append(Action("gesture", "wiggle", 2))

                if self._engaged_person is not None:
                    self.memory.add_attention(self._engaged_person, dt)
                self._last_interaction = now

                # Periodic micro-reactions while someone is around.
                if now >= self._next_react:
                    self._next_react = now + self.rng.uniform(t.react_min, t.react_max)
                    close = face.area_frac > 0.06
                    if close:
                        choice = self.rng.choice(["giggle", "happy", "curious", "excited"])
                        gesture = {"giggle": "wiggle", "happy": "nod", "curious": "tilt", "excited": "bounce"}[choice]
                    else:
                        choice = self.rng.choice(["curious", "curious", "happy", "confused"])
                        gesture = {"curious": "tilt", "happy": "nod", "confused": "tilt"}[choice]
                    if self.mood.energy < 0.25:
                        choice, gesture = "sleepy", "droop"
                    actions.append(Action("sound", choice, 1))
                    actions.append(Action("gesture", gesture, 1))

            else:  # no face this tick
                if self.state == "ENGAGED":
                    if now - self._last_face_time > t.face_lost_grace:
                        engaged_for = now - self._engaged_since
                        self._enter("SEARCHING", now)
                        self.gaze = (self._last_seen_yaw, self._last_seen_pitch)
                        actions.append(Action("gesture", "search", 2))
                        if engaged_for > t.engaged_sad_if_over:
                            actions.append(Action("sound", "sad", 2))
                        else:
                            actions.append(Action("sound", "confused", 2))
                elif self.state == "SEARCHING":
                    self.gaze = (self._last_seen_yaw, self._last_seen_pitch)
                    if now - self._state_since > t.search_duration:
                        self._engaged_track = None
                        self._engaged_person = None
                        self._enter("IDLE", now)
                        self.gaze = None
                        self._next_glance = now + self.rng.uniform(t.idle_glance_min, t.idle_glance_max)
                else:  # IDLE
                    self.gaze = None
                    if obs.loud_yaw_deg is not None:
                        self.gaze = (obs.loud_yaw_deg, 0.0)
                        actions.append(Action("gesture", "perk", 2))
                        actions.append(Action("sound", "curious", 1))
                        self._next_glance = now + 2.0
                    elif now >= self._next_glance:
                        self._next_glance = now + self.rng.uniform(t.idle_glance_min, t.idle_glance_max)
                        actions.append(Action("gesture", "glance", 0))
                    alone_for = now - self._last_interaction
                    if alone_for > t.sleep_after or self.mood.energy < 0.08:
                        actions.append(Action("sound", "yawn", 3))
                        actions.append(Action("sleep", "tired", 5))
                        self._enter("SLEEPING", now)
                        self._face_first_seen = 0.0
                    elif alone_for > t.lonely_after and now - self._last_lonely > t.lonely_repeat:
                        self._last_lonely = now
                        actions.append(Action("sound", "lonely", 1))
                        actions.append(Action("gesture", "droop", 1))

        return actions

    def status(self) -> dict:
        return {
            "state": self.state,
            "energy": round(self.mood.energy, 2),
            "social": round(self.mood.social, 2),
            "engaged_person": None if self._engaged_person is None else self._engaged_person.person_id,
            "gaze": self.gaze,
        }
