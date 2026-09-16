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
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from festival_pet.memory import FaceMemory, Person

State = Literal["SLEEPING", "WAKING", "IDLE", "ENGAGED", "SEARCHING", "HELD"]


def awake_now(state: str) -> bool:
    return state not in ("SLEEPING", "WAKING")


@dataclass
class FaceObs:
    """One tracked face for this tick (from the vision thread)."""

    track_id: int
    yaw_deg: float  # absolute world yaw the head should take to look at it
    pitch_deg: float
    area_frac: float  # bbox area / frame area, a proxy for closeness
    person: Person | None  # None while unknown / not yet embedded
    similarity: float
    roll_deg: float = 0.0  # the person's head tilt


@dataclass
class Observation:
    """Everything the senses report for one tick."""

    face: FaceObs | None = None
    held: bool = False
    shaken: bool = False
    touched: bool = False  # an antenna ("ear") was pushed this tick (edge, not level)
    touched_side: int = 0  # 0 right, 1 left
    petted: bool = False  # hand rubbing the head this tick (edge)
    petting: bool = False  # rub still going on (level)
    loud_yaw_deg: float | None = None  # direction of a sudden loud sound, if any
    scratched: bool = False  # fingernails on the shell this tick (edge)
    name_heard: bool = False  # someone said "Reachy" (edge)
    command: str | None = None  # "dance", "hello", "hi", "good", "sleep" (edge)
    voice_yaw_deg: float | None = None  # where the latest speech came from
    music_bpm: float = 0.0  # 0 when no confident beat
    music_confidence: float = 0.0


@dataclass
class Action:
    """A request for the robot layer."""

    kind: Literal["sound", "gesture", "move", "wake", "sleep", "groove", "mirror"]
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
    name_attentive: float = 8.0  # how long a name call keeps it attentive / listening for a trick
    trick_window: float = 15.0  # a second "dance" within this window upgrades to the lively dance
    peekaboo_min_gap: float = 0.6  # face hidden at least this long...
    peekaboo_max_gap: float = 3.5  # ...and back within this = peekaboo
    shy_stare: float = 14.0  # someone very close for this long -> shy
    sneeze_min: float = 240.0
    sneeze_max: float = 900.0
    hiccup_chance_per_s: float = 0.002


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
    _attentive_until: float = -1e9
    _last_dance_cmd: float = -1e9
    _dance_tier: int = 0
    _face_lost_at: float = 0.0
    _close_since: float = 0.0
    _shy_done_track: int | None = None
    _next_sneeze: float = 0.0
    _next_sing: float = 0.0
    _nodding_off: bool = False
    _music_since: float = 0.0
    _music_greeted: bool = False
    _little_dance_until: float = -1e9
    _ear_tickles: int = 0
    _last_ear_tickle: float = -1e9
    _last_pet_purr: float = -1e9

    # gaze the motion layer should aim for (None = free to idle-drift)
    gaze: tuple[float, float] | None = None
    thoughts: deque = field(default_factory=lambda: deque(maxlen=60))  # (time, text) inner monologue

    def _think(self, now: float, text: str) -> None:
        self.thoughts.append((now, text))

    def start(self, now: float, awake: bool) -> None:
        """Call once before ticking. ``awake`` reflects whether the daemon already woke the robot."""
        self.state = "IDLE" if awake else "SLEEPING"
        self._state_since = now
        self._last_interaction = now
        self._last_face_time = now
        self._next_glance = now + self.rng.uniform(self.timers.idle_glance_min, self.timers.idle_glance_max)
        self._next_sneeze = now + self.rng.uniform(self.timers.sneeze_min, self.timers.sneeze_max)

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
                # Ears are ticklish: pull the touched antenna away like a dog flicking its ear, and giggle.
                self._ear_tickles = self._ear_tickles + 1 if now - self._last_ear_tickle < 6.0 else 1
                self._last_ear_tickle = now
                side = "left" if obs.touched_side == 1 else "right"
                if self._ear_tickles >= 4:
                    self._think(now, f"my {side} ear again?! okay that's enough")
                    actions.append(Action("sound", "annoyed", 3))
                    actions.append(Action("gesture", "shake_off", 3))
                    self._ear_tickles = 0
                else:
                    self._think(now, f"eek, my {side} ear! ticklish")
                    actions.append(Action("sound", self.rng.choice(["giggle", "ticklish"]), 2))
                    actions.append(Action("gesture", f"flinch:{'+' if obs.touched_side == 1 else '-'}", 3))
                self.mood.social += 0.03

        if obs.petted:
            self._last_interaction = now
            if self.state == "SLEEPING":
                actions.append(Action("wake", "pet", 5))
                self._enter("WAKING", now)
            else:
                self._think(now, "ahh, head pets... leaning in")
                actions.append(Action("sound", self.rng.choice(["purr", "content"]), 3))
                actions.append(Action("gesture", "lean", 3))
                self.mood.social += 0.08
                if self._engaged_person is not None:
                    self.memory.add_pet(self._engaged_person)
        elif obs.petting and awake_now(self.state) and now - self._last_pet_purr > 3.0:
            self._last_pet_purr = now
            actions.append(Action("sound", "purr", 1))
            actions.append(Action("gesture", "lean", 1))

        awake = self.state not in ("SLEEPING", "WAKING")

        if obs.scratched:
            self._last_interaction = now
            if self.state == "SLEEPING":
                actions.append(Action("wake", "scratch", 5))
                self._enter("WAKING", now)
            else:
                actions.append(Action("sound", "ticklish", 3))
                self._think(now, "hehe that tickles! (shell scratched)")
                actions.append(Action("gesture", self.rng.choice(["wiggle", "bounce"]), 3))
                self.mood.social += 0.08
                self.mood.energy += 0.03
                if self._engaged_person is not None:
                    self.memory.add_pet(self._engaged_person)

        if obs.name_heard:
            self._last_interaction = now
            self._attentive_until = now + t.name_attentive
            self._think(now, f"someone said my name{' from the ' + ('left' if (obs.voice_yaw_deg or 0) > 0 else 'right') if obs.voice_yaw_deg is not None else ''}! listening for a trick for {t.name_attentive:.0f} s")
            if self.state == "SLEEPING":
                actions.append(Action("wake", "name", 5))
                self._enter("WAKING", now)
            else:
                actions.append(Action("sound", "name", 3))
                actions.append(Action("gesture", "perk", 3))
                if obs.face is None and obs.voice_yaw_deg is not None and self.state in ("IDLE", "SEARCHING"):
                    self.gaze = (obs.voice_yaw_deg, 0.0)
                    self._last_seen_yaw, self._last_seen_pitch = obs.voice_yaw_deg, 0.0
                    self._enter("SEARCHING", now)

        if obs.command is not None and awake and (now <= self._attentive_until or obs.face is not None):
            self._last_interaction = now
            self._think(now, f"heard '{obs.command}' and I was paying attention")
            actions += self._on_command(obs.command, now)

        # Music: groove when there is a confident beat and nothing bigger is happening.
        if obs.music_bpm > 0 and obs.music_confidence > 0:
            if self._music_since == 0.0:
                self._music_since = now
            settled = now - self._music_since > 4.0
            if settled and awake and self.state != "HELD" and now > self._dizzy_until:
                intensity = 0.25 + 0.35 * self.mood.energy
                if now < self._little_dance_until:
                    intensity = 0.9  # "little dance" trick: visibly grooving for a few seconds
                actions.append(Action("groove", f"{intensity:.2f}", 0))
                if not self._music_greeted:
                    self._music_greeted = True
                    self._think(now, f"ooh, music! {obs.music_bpm:.0f} bpm, I'll bob along")
                    actions.append(Action("sound", "happy", 1))
                    actions.append(Action("gesture", "bounce", 1))
                if now >= self._next_sing and self.state in ("IDLE", "ENGAGED") and self.mood.energy > 0.3:
                    self._next_sing = now + self.rng.uniform(6.0, 20.0)
                    actions.append(Action("sound", "sing", 0))
        else:
            self._music_since = 0.0
            self._music_greeted = False
            if now < self._little_dance_until and awake and self.state != "HELD":
                actions.append(Action("groove", "0.90", 1))  # no music: dance to its own inner tempo

        if obs.face is not None:
            actions.append(Action("mirror", f"{obs.face.roll_deg:.1f}", 0))

        if pickup_edge:
            self._last_interaction = now
            if self.state == "SLEEPING":
                actions.append(Action("wake", "pickup", 5))
            actions.append(Action("sound", "surprised", 4))
            actions.append(Action("gesture", "startle", 4))
            self._think(now, "whoa, I'm being picked up!")
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
                self._think(now, "awake and ready")
                self._enter("IDLE", now)
                self._next_glance = now + self.rng.uniform(1.0, 3.0)

        elif self.state == "HELD":
            self.gaze = None
            if obs.shaken and now > self._dizzy_until:
                self._dizzy_until = now + 3.0
                actions.append(Action("sound", "dizzy", 4))
                self._think(now, "too much shaking... dizzy")
                actions.append(Action("gesture", "dizzy", 4))
                self._next_purr = now + 4.0
            elif now > self._dizzy_until and now >= self._next_purr:
                actions.append(Action("sound", self.rng.choice(["purr", "content", "content"]), 2))
                actions.append(Action("gesture", "snuggle", 2))
                self._next_purr = now + self.rng.uniform(t.held_purr_min, t.held_purr_max)
            if setdown_edge:
                actions.append(Action("sound", "happy", 3))
                actions.append(Action("gesture", "shake_off", 3))
                self._think(now, "back on the ground, shaking it off")
                self._last_interaction = now
                self._enter("IDLE", now)

        elif self.state in ("IDLE", "SEARCHING", "ENGAGED"):
            face = obs.face
            if face is not None:
                self._last_seen_yaw, self._last_seen_pitch = face.yaw_deg, face.pitch_deg
                self.gaze = (face.yaw_deg, face.pitch_deg)

                gap = now - self._last_face_time
                if face.track_id == self._engaged_track and t.peekaboo_min_gap < gap < t.peekaboo_max_gap:
                    # Peekaboo! They hid and came right back.
                    actions.append(Action("sound", "giggle", 3))
                    actions.append(Action("gesture", "bounce", 3))
                    self._think(now, "peekaboo! they came back")
                    self.mood.social += 0.05
                if face.area_frac > 0.12:
                    if self._close_since == 0.0:
                        self._close_since = now
                    elif now - self._close_since > t.shy_stare and self._shy_done_track != face.track_id:
                        self._shy_done_track = face.track_id
                        actions.append(Action("sound", "shy", 2))
                        self._think(now, "they've been staring so close for so long... getting shy")
                        actions.append(Action("gesture", "shy", 2))
                else:
                    self._close_since = 0.0
                self._last_face_time = now

                if self.state != "ENGAGED" or face.track_id != self._engaged_track:
                    # New engagement (or switched to a different person).
                    self._engaged_track = face.track_id
                    self._engaged_person = face.person
                    self._engaged_since = now
                    self._greeted_track = None
                    self._enter("ENGAGED", now)
                    self._next_react = now + self.rng.uniform(t.react_min, t.react_max)
                    self._think(now, f"a face! track #{face.track_id}, {'close' if face.area_frac > 0.06 else 'a bit away'}, figuring out who")

                # Greet once we know who it is (or have given up identifying them).
                if self._greeted_track != face.track_id and (face.person is not None or now - self._engaged_since > 1.5):
                    self._greeted_track = face.track_id
                    self._engaged_person = face.person
                    if face.person is not None:
                        self.memory.sighted(face.person, now)
                    sound, gesture, move = self._person_greeting(face.person)
                    self._think(now, "a new face, saying hi" if face.person is None else f"it's person #{face.person.person_id} ({face.person.tier()}, visit {face.person.encounters}), greeting them")
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
                self._close_since = 0.0
                if self.state == "ENGAGED":
                    if now - self._last_face_time > t.face_lost_grace:
                        engaged_for = now - self._engaged_since
                        self._face_lost_at = self._last_face_time
                        self._enter("SEARCHING", now)
                        self.gaze = (self._last_seen_yaw, self._last_seen_pitch)
                        self._think(now, f"where did they go? looking where I last saw them ({'miss them' if engaged_for > t.engaged_sad_if_over else 'hm?'})")
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
                        self._think(now, "gave up looking, back to idling")
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
                    if self.mood.energy < 0.3 and not self._nodding_off and alone_for > 30.0 and self.rng.random() < dt * 0.02:
                        self._nodding_off = True
                        actions.append(Action("sound", "sleepy", 1))
                        actions.append(Action("gesture", "nod_off", 1))
                    if alone_for > t.sleep_after or self.mood.energy < 0.08:
                        actions.append(Action("sound", "yawn", 3))
                        actions.append(Action("sleep", "tired", 5))
                        self._think(now, "so tired... going to sleep" if self.mood.energy < 0.08 else f"nobody around for {alone_for / 60:.0f} min, dozing off")
                        self._enter("SLEEPING", now)
                        self._face_first_seen = 0.0
                    elif alone_for > t.lonely_after and now - self._last_lonely > t.lonely_repeat:
                        self._last_lonely = now
                        actions.append(Action("sound", "lonely", 1))
                        self._think(now, f"alone for {alone_for:.0f} s... lonely")
                        actions.append(Action("gesture", "droop", 1))

        if awake and self.state in ("IDLE", "ENGAGED"):
            if now >= self._next_sneeze:
                self._next_sneeze = now + self.rng.uniform(t.sneeze_min, t.sneeze_max)
                actions.append(Action("sound", "sneeze", 2))
                self._think(now, "ah... ah... choo!")
                actions.append(Action("gesture", "sneeze", 2))
            elif self.rng.random() < dt * t.hiccup_chance_per_s:
                actions.append(Action("sound", "hiccup", 1))
                actions.append(Action("gesture", "hiccup", 1))
        if self.state == "SLEEPING":
            self._nodding_off = False

        return actions

    def _on_command(self, command: str, now: float) -> list[Action]:
        t = self.timers
        if command == "dance":
            if now - self._last_dance_cmd < t.trick_window:
                self._dance_tier = 2
                self._last_dance_cmd = now
                self._little_dance_until = -1e9
                move = self.rng.choice(["dance1", "dance2", "dance3"])
                return [Action("sound", "excited", 3), Action("move", move, 3)]
            self._dance_tier = 1
            self._last_dance_cmd = now
            self._little_dance_until = now + 6.0
            # A little dance: a few seconds of visible grooving (to the music, or its own inner tempo).
            return [Action("sound", "happy", 2), Action("gesture", "bounce", 2)]
        if command in ("hello", "hi"):
            return [Action("sound", "hello_friend", 2), Action("gesture", "nod", 2)]
        if command == "good":
            self.mood.social += 0.1
            if self._engaged_person is not None:
                self.memory.add_pet(self._engaged_person)
            return [Action("sound", "content", 2), Action("gesture", "wiggle", 2)]
        if command == "sleep":
            self._enter("SLEEPING", now)
            self._face_first_seen = 0.0
            return [Action("sound", "yawn", 3), Action("sleep", "asked", 5)]
        raise KeyError(f"Unknown command '{command}'")

    def status(self) -> dict:
        return {
            "state": self.state,
            "energy": round(self.mood.energy, 2),
            "social": round(self.mood.social, 2),
            "engaged_person": None if self._engaged_person is None else self._engaged_person.person_id,
            "engaged_tier": None if self._engaged_person is None else self._engaged_person.tier(),
            "gaze": self.gaze,
        }

    def mind(self, now: float) -> dict:
        """Everything driving the next decision, for the 'inside the mind' page."""
        t = self.timers
        alone = now - self._last_interaction
        return {
            **self.status(),
            "state_for_s": round(now - self._state_since, 1),
            "alone_for_s": round(alone, 1),
            "lonely_in_s": round(max(0.0, t.lonely_after - alone), 1),
            "sleep_in_s": round(max(0.0, t.sleep_after - alone), 1),
            "next_reaction_in_s": round(max(0.0, self._next_react - now), 1) if self.state == "ENGAGED" else None,
            "next_glance_in_s": round(max(0.0, self._next_glance - now), 1) if self.state == "IDLE" else None,
            "listening_for_trick_s": round(max(0.0, self._attentive_until - now), 1),
            "little_dance_s": round(max(0.0, self._little_dance_until - now), 1),
            "next_sneeze_in_s": round(max(0.0, self._next_sneeze - now)),
            "dizzy": now < self._dizzy_until,
            "thoughts": [{"t": round(now - ts, 1), "text": txt} for ts, txt in reversed(self.thoughts)],
        }
