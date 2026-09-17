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

from festival_pet.hearing import intents_in
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
    head_yaw_deg: float = 0.0  # where their head is turned (rough)
    head_pitch_deg: float = 0.0


@dataclass
class Observation:
    """Everything the senses report for one tick."""

    face: FaceObs | None = None
    body: FaceObs | None = None  # a torso with no face: where the head should be
    heard_text: str | None = None  # a finished sentence from the free-vocabulary recogniser
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
    voice_started: bool = False  # someone began talking after a pause (edge)
    music_bpm: float = 0.0  # 0 when no confident beat
    music_confidence: float = 0.0
    dance_bpm: float = 0.0  # someone visibly bobbing at this tempo (0 = nobody dancing)


@dataclass
class Action:
    """A request for the robot layer."""

    kind: Literal["sound", "gesture", "move", "wake", "sleep", "groove", "mirror", "heard", "mimic"]
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
    peekaboo_min_gap: float = 1.5  # face hidden at least this long (detector dropouts are shorter)...
    peekaboo_max_gap: float = 4.0  # ...and back within this = peekaboo
    peekaboo_cooldown: float = 20.0
    regreet_person_after: float = 120.0  # a known person is not greeted again within this
    regreet_stranger_after: float = 30.0  # an unknown track that re-locks quickly gets a small "oh, hi again"
    mimic_cooldown: float = 6.0
    shy_stare: float = 14.0  # someone very close for this long -> shy
    sneeze_min: float = 240.0
    sneeze_max: float = 900.0
    hiccup_chance_per_s: float = 0.002
    voice_glance_cooldown: float = 4.0
    mimic_min_area: float = 0.08  # face must fill this much of the frame to start the mirror game...
    mimic_hold: float = 2.0  # ...and stay for this long
    mimic_max_s: float = 60.0
    mimic_exit_area: float = 0.04  # they backed away


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
    _last_voice_glance: float = -1e9
    _voice_lock_until: float = -1e9
    _last_heard_react: float = -1e9
    _body_since: float = 0.0
    _last_peekaboo: float = -1e9
    _last_greet_time: float = -1e9
    _last_greet_person: int | None = None
    _face_hist: deque = field(default_factory=lambda: deque(maxlen=40))  # (t, yaw, pitch) for nod/shake mimicry
    _last_mimic: float = -1e9
    _mimic_candidate_since: float = 0.0
    _mimic_since: float = 0.0
    mimicking: bool = False
    _dance_since: float = 0.0
    _dance_greeted: bool = False

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
                if obs.voice_yaw_deg is not None and self.state != "HELD":
                    # They called me: turn to the voice even if I was watching someone else.
                    self.gaze = (obs.voice_yaw_deg, 0.0)
                    self._last_seen_yaw, self._last_seen_pitch = obs.voice_yaw_deg, 0.0
                    self._engaged_track = None
                    self._voice_lock_until = now + 1.5  # hold the turn even if a face is still in view
                    self._enter("SEARCHING", now)

        if obs.voice_started and obs.voice_yaw_deg is not None and awake and self.state in ("IDLE", "SEARCHING") and now - self._last_voice_glance > t.voice_glance_cooldown:
            self._last_voice_glance = now
            self._think(now, f"a voice from the {'left' if obs.voice_yaw_deg > 0 else 'right'}... who's that?")
            self.gaze = (obs.voice_yaw_deg, 0.0)
            self._last_seen_yaw, self._last_seen_pitch = obs.voice_yaw_deg, 0.0
            self._enter("SEARCHING", now)
            actions.append(Action("gesture", "perk", 1))
            if self.rng.random() < 0.4:
                actions.append(Action("sound", "curious", 1))

        if obs.heard_text is not None and awake:
            found = intents_in(obs.heard_text)
            actions.append(Action("heard", "|".join([obs.heard_text, *found]), 0))
            actions += self._on_heard(obs.heard_text, found, now)

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
        elif obs.dance_bpm > 0 and awake and self.state != "HELD" and now > self._dizzy_until:
            # No music to hear, but someone is visibly dancing: dance along to what we see.
            self._music_since = 0.0
            if self._dance_since == 0.0:
                self._dance_since = now
            if not self._dance_greeted and now - self._dance_since > 1.0:
                self._dance_greeted = True
                self._think(now, f"they're dancing! (~{obs.dance_bpm:.0f} bpm) I'll dance too")
                actions.append(Action("sound", "excited", 2))
                actions.append(Action("gesture", "bounce", 2))
            actions.append(Action("groove", f"{0.5 + 0.4 * self.mood.energy:.2f}|visual", 0))
        else:
            self._music_since = 0.0
            self._music_greeted = False
            self._dance_since = 0.0
            self._dance_greeted = False
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
            # Camera is off while asleep: only the ears (name, a loud voice, head pets, ear tickles) wake it.
            self.gaze = None
            if obs.loud_yaw_deg is not None and self.mood.energy > 0.3:
                self._think(now, "huh?! what was that noise")
                actions.append(Action("sound", "surprised", 5))
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
                if now >= self._voice_lock_until:
                    self.gaze = (face.yaw_deg, face.pitch_deg)

                gap = now - self._last_face_time
                if face.track_id == self._engaged_track and t.peekaboo_min_gap < gap < t.peekaboo_max_gap and now - self._last_peekaboo > t.peekaboo_cooldown:
                    # Peekaboo! They hid and came right back.
                    self._last_peekaboo = now
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
                    same_person_recently = face.person is not None and self._last_greet_person == face.person.person_id and now - self._last_greet_time < t.regreet_person_after
                    stranger_recently = face.person is None and now - self._last_greet_time < t.regreet_stranger_after
                    if same_person_recently or stranger_recently:
                        # We just said hello; a full greeting again would look like a broken record.
                        self._think(now, "oh, you again! (no big hello twice)")
                        actions.append(Action("sound", "curious", 1))
                        actions.append(Action("gesture", "tilt", 1))
                    else:
                        self._last_greet_time = now
                        self._last_greet_person = None if face.person is None else face.person.person_id
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

                # The mirror game: a close, steady face for a couple of seconds and it goes quiet and copies you.
                if not self.mimicking:
                    if face.area_frac >= t.mimic_min_area:
                        if self._mimic_candidate_since == 0.0:
                            self._mimic_candidate_since = now
                        elif now - self._mimic_candidate_since >= t.mimic_hold and obs.dance_bpm == 0:
                            self.mimicking, self._mimic_since = True, now
                            self._think(now, "you're right up close... let's play mirror. I'll copy you")
                            actions.append(Action("sound", "curious", 1))
                    else:
                        self._mimic_candidate_since = 0.0
                elif face.area_frac < t.mimic_exit_area or now - self._mimic_since > t.mimic_max_s or obs.dance_bpm > 0:
                    self.mimicking = False
                    self._mimic_candidate_since = 0.0
                    self._think(now, "mirror game over")
                    actions.append(Action("gesture", "wiggle", 1))
                if self.mimicking:
                    actions.append(Action("mimic", f"{face.head_yaw_deg:.1f},{face.head_pitch_deg:.1f},{face.roll_deg:.1f}", 0))
                    self._next_react = now + 5.0  # no micro-reactions while mirroring

                # Mirror them: nod back at a nod, shake back at a shake (tilt is mirrored continuously by the body).
                self._face_hist.append((now, face.yaw_deg, face.pitch_deg))
                # Not while they (or the music) are moving to a beat: a bob that keeps going is dancing, not a nod.
                if not self.mimicking and now - self._last_mimic > t.mimic_cooldown and obs.dance_bpm == 0 and obs.music_bpm == 0:
                    mimic = self._detect_nod_or_shake(now)
                    if mimic is not None:
                        self._last_mimic = now
                        self._face_hist.clear()
                        self._think(now, f"they {'nodded' if mimic == 'nod' else 'shook their head'}... me too!")
                        actions.append(Action("sound", "happy" if mimic == "nod" else "curious", 2))
                        actions.append(Action("gesture", mimic, 2))

                # Periodic micro-reactions while someone is around (not while we are dancing with them).
                if obs.dance_bpm > 0:
                    self._next_react = max(self._next_react, now + 3.0)
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

            elif obs.body is not None and self.state != "ENGAGED" and now - self._last_face_time > 2.5:
                # Someone's torso is in view but not their face: look up to where the head should be.
                self._close_since = 0.0
                if self._body_since == 0.0:
                    self._body_since = now
                    self._think(now, "a body! looking up for the face")
                self.gaze = (obs.body.yaw_deg, obs.body.pitch_deg)
                self._last_seen_yaw, self._last_seen_pitch = obs.body.yaw_deg, obs.body.pitch_deg
                self._last_interaction = now
                if self.state == "IDLE":
                    self._enter("SEARCHING", now)
                    actions.append(Action("gesture", "perk", 1))
                elif self.state == "SEARCHING":
                    self._state_since = now  # keep searching while there is a body to look at
            else:  # no face this tick
                self._close_since = 0.0
                self._body_since = 0.0
                self._mimic_candidate_since = 0.0
                if self.mimicking and now - self._last_face_time > 1.0:
                    self.mimicking = False
                    self._think(now, "mirror game over (lost you)")
                if self.state == "ENGAGED" and obs.dance_bpm > 0:
                    pass  # mid-dance the tracker blinks a lot (we are moving too): keep dancing, keep the gaze
                elif self.state == "ENGAGED":
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
                actions.append(Action("sound", "sneeze", 3))
                self._think(now, "ah... ah... choo!")
                actions.append(Action("gesture", "sneeze", 3))
            elif self.rng.random() < dt * t.hiccup_chance_per_s:
                actions.append(Action("sound", "hiccup", 1))
                actions.append(Action("gesture", "hiccup", 1))
        if self.state == "SLEEPING":
            self._nodding_off = False

        return actions

    def _detect_nod_or_shake(self, now: float) -> str | None:
        """Two or more up/down (nod) or left/right (shake) reversals of 3+ degrees within the last 2 s,
        and the head has come to rest in the last half second: a nod is a burst that ends, whereas a bob
        that keeps going is dancing and belongs to the dance detector."""
        pts = [(t, y, p) for t, y, p in self._face_hist if now - t <= 2.0]
        if len(pts) < 8:
            return None
        for axis, name in ((2, "nod"), (1, "shake")):
            vals = [p[axis] for p in pts]
            base = sum(vals) / len(vals)
            dev = [v - base for v in vals]
            if max(dev) - min(dev) < 3.0:
                continue
            recent = [d for (t_, _, _), d in zip(pts, dev) if now - t_ <= 0.5]
            if len(recent) >= 2 and max(recent) - min(recent) > 1.5:
                continue  # still moving
            # count sign reversals of the deviation, ignoring the small stuff
            signs = [1 if d > 1.0 else -1 if d < -1.0 else 0 for d in dev]
            signs = [x for x in signs if x != 0]
            reversals = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
            if reversals >= 3:
                return name
        return None

    def _on_heard(self, text: str, found: list[str], now: float) -> list[Action]:
        """React to the gist of what someone said. Rough transcripts, so small reactions only."""
        if not found:
            self._think(now, f"heard: \"{text}\"")
            return []
        self._think(now, f"heard: \"{text}\" ... sounds like {', '.join(found)}")
        if now - self._last_heard_react < 3.0:
            return []
        self._last_heard_react = now
        self._last_interaction = now
        intent = found[0]
        if intent == "praise":
            self.mood.social += 0.08
            if self._engaged_person is not None:
                self.memory.add_pet(self._engaged_person)
            return [Action("sound", "happy", 2), Action("gesture", "bounce", 2)]
        if intent == "cute":
            return [Action("sound", "shy", 2), Action("gesture", "shy", 2)]
        if intent == "greeting":
            return [Action("sound", "hello_friend", 2), Action("gesture", "nod", 2)]
        if intent == "farewell":
            return [Action("sound", "sad", 2), Action("gesture", "droop", 2)]
        if intent == "question":
            return [Action("sound", "curious", 1), Action("gesture", "tilt", 1)]
        if intent == "laugh":
            return [Action("sound", "giggle", 2), Action("gesture", "wiggle", 2)]
        if intent == "scold":
            self.mood.social -= 0.05
            return [Action("sound", "sad", 2), Action("gesture", "droop", 2)]
        if intent == "sad":
            return [Action("sound", "content", 1), Action("gesture", "lean", 1)]
        if intent == "photo":
            return [Action("sound", "tada", 2), Action("gesture", "tada", 2)]
        if intent == "come":
            return [Action("sound", "curious", 1), Action("gesture", "perk", 1)]
        raise KeyError(intent)

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
            "mimicking": self.mimicking,
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
