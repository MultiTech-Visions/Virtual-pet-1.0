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

from festival_pet.drives import (
    ASK_CEILING_S,
    ASK_MIN_INTERVAL_S,
    BORED_NEW_ACTIVITY,
    BORED_NEW_FACE,
    BORED_TOUCH,
    CURIOSITY_LOUD,
    CURIOSITY_NEW_FACE,
    CURIOSITY_NEW_SECTOR,
    HESITATE_MARGIN,
    MIN_DWELL_S,
    Attention,
    Drives,
    Situation,
    choose,
    signature,
)
from festival_pet.hearing import intents_in
from festival_pet.memory import FaceMemory, Person

State = Literal["SLEEPING", "WAKING", "IDLE", "ENGAGED", "SEARCHING", "HELD"]
EAR_WINDOW_S = 12.0  # tickles this close together count as one bout
EAR_TUCK_AFTER = 4  # the tickle that ends the game
EAR_TUCK_S = 26.0  # how long the sulk lasts, left alone (matches MotionComposer.EAR_TUCK_S)
EAR_SWAT_S = 1.3  # one swat (GESTURES["swat"]): pokes during it are covered by its taps
EAR_MAKEUP_S = 3.0  # after the last poke, the nuzzle: keep poking and it keeps batting
WAVE_COOLDOWN_S = 6.0  # one wave back per wave, not one per swing
HUG_COOLDOWN_S = 20.0
BODY_CONFIRM_S = 1.0  # a torso must be seen this long before it is worth looking up at
BODY_GIVE_UP_S = 6.0  # looking up at a torso this long without finding a face: not a person
BODY_IGNORE_S = 120.0  # ...and that spot is ignored for this long
BODY_IGNORE_DEG = 25.0


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
    smile: float = 0.0  # mouth width / eye distance; 0.8+ is a smile


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
    grooving: bool = False  # manual groove with a tempo tapped in: we are dancing, nothing else starts
    busy: str | None = None  # what the robot layer is running: "mime", "sing", "gesture" (a solo one) or "turn" (keypad groove turn)
    arms: object | None = None  # pose.Arms: the person's arms are readable this tick (the arm games need this)
    waved: str | None = None  # they waved a hand this tick (edge): the PERSON's "left" or "right"
    hugged: bool = False  # arms held out wide at it for a while (edge): a hug


@dataclass
class Action:
    """A request for the robot layer."""

    kind: Literal["sound", "gesture", "move", "wake", "sleep", "groove", "mirror", "heard", "mimic", "activity", "ears"]
    name: str
    priority: int = 1  # higher preempts lower for gestures/moves


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
    mood: Drives = field(default_factory=Drives)

    # the activity layer (drives.py): what it has decided to be doing, and how it got there
    activity: str = "hangout"
    runner_up: str | None = None
    can_sing: bool = False  # the robot layer says whether songs are switched on
    can_mime: bool = True
    attention: Attention = field(default_factory=Attention)
    _activity_since: float = 0.0
    _activity_prev: str = ""
    _cool: dict = field(default_factory=dict)  # activity -> earliest time it may be chosen again
    _last_sig: tuple = ()
    _last_ask: float = -1e9
    _margin: float = 1.0
    _scores: dict = field(default_factory=dict)
    _close_hold_since: float = 0.0
    _next_nag: float = 0.0  # ask_attention repeats
    _nag_company: bool = False
    _ear_tucked: int | None = None  # which antenna is parked over the head (not in the mood)
    _ear_seq_at: float = 0.0  # when the swatting is over and the make-up nuzzle starts (pushed back by every poke)
    _last_swat: float = -1e9
    _last_wave: float = -1e9
    _last_hug: float = -1e9
    _body_first: float = 0.0  # a torso has been in view since (0 = none)
    _body_last: float = -1e9  # a torso was last in view at
    _last_sleepy: float = -1e9  # the sleepy noise is rationed
    _body_ignore: list = field(default_factory=list)  # (yaw, until): spots that turned out not to be people

    # bookkeeping
    _state_since: float = 0.0
    _last_interaction: float = 0.0
    _last_face_time: float = 0.0
    _face_first_seen: float = 0.0
    _last_lonely: float = -1e9
    _next_react: float = 0.0
    _next_glance: float = 0.0
    _look_until: float = 0.0  # idle look-around: gaze held wide (body turns) until then
    _look_at: tuple[float, float] = (0.0, 0.0)
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
    _sneeze_show_until: float = 0.0
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
        self._activity_since = now

    # ------------------------------------------------------------------ the activity layer
    def _situation(self, obs: Observation, now: float) -> Situation:
        face = obs.face
        if face is not None and face.area_frac >= self.timers.mimic_min_area:
            if self._close_hold_since == 0.0:
                self._close_hold_since = now
        else:
            self._close_hold_since = 0.0
        close = self._close_hold_since != 0.0 and now - self._close_hold_since >= self.timers.mimic_hold
        return Situation(person=face is not None, close=close, beat=obs.music_bpm > 0 or obs.dance_bpm > 0 or obs.grooving, held=obs.held,
                         busy=obs.busy, can_sing=self.can_sing, can_mime=self.can_mime and (face is not None or obs.arms is not None))

    def _choose(self, obs: Observation, now: float) -> list[Action]:
        """Re-decide what to be doing when the situation changes, when a game or song ends, or every so often."""
        sit = self._situation(obs, now)
        actions: list[Action] = []
        if self.activity in ("mime", "sing") and sit.busy != self.activity and now - self._activity_since > 1.0:
            # the robot layer finished (or refused) it: back to choosing, and not that again for a while
            self._end_activity(now, cooldown=self.rng.uniform(90.0, 240.0))
        if sit.busy in ("mime", "sing") and self.activity != sit.busy:
            self._switch(sit.busy, now)  # started from the page or a keypad: that is what we are doing now
            return actions
        if sit.busy is not None:
            return actions  # a game or a song runs to its end
        sig = signature(self.mood, sit)
        changed = sig != self._last_sig and now - self._last_ask >= ASK_MIN_INTERVAL_S
        if not (changed or now - self._last_ask >= ASK_CEILING_S or self._last_ask == -1e9):
            return actions
        # Give what it is doing a fair go: only the situation itself (the first five signature fields), not a
        # drive creeping over a band edge or the 30 s clock, may cut an activity short of its dwell time.
        big_change = sig[:5] != self._last_sig[:5] if self._last_sig else True
        if now - self._activity_since < MIN_DWELL_S.get(self.activity, 0.0) and not big_change:
            self._last_sig = sig
            return actions
        self._last_sig, self._last_ask = sig, now
        choice = choose(self.mood, sit, self.activity, self._cool, now, self.rng)
        self.runner_up, self._margin, self._scores = choice.runner_up, choice.margin, choice.scores
        if choice.activity == self.activity:
            return actions
        if choice.margin < HESITATE_MARGIN and choice.runner_up is not None:
            self._think(now, f"hmm... {choice.activity.replace('_', ' ')}? or {choice.runner_up.replace('_', ' ')}?")
            actions.append(Action("gesture", "tilt", 1))
        actions += self._switch(choice.activity, now)
        return actions

    def _end_activity(self, now: float, cooldown: float) -> None:
        self._cool[self.activity] = now + cooldown
        self._activity_prev, self.activity, self._activity_since = self.activity, "hangout", now
        self._last_sig, self._last_ask = (), -1e9  # re-decide on the next tick

    def _switch(self, new: str, now: float) -> list[Action]:
        old = self.activity
        if new != self._activity_prev:  # a genuinely new pastime; bouncing back to the one just left is no relief
            self.mood.boredom -= BORED_NEW_ACTIVITY
            self.mood.clamp()
        if old == "mirror" and self.mimicking:
            self.mimicking = False
            self._cool["mirror"] = now + 60.0
        self._activity_prev, self.activity, self._activity_since = old, new, now
        self._last_sig = ()
        out: list[Action] = [Action("activity", new, 0)]
        why = f"(bored {self.mood.boredom:.1f}, curious {self.mood.curiosity:.1f}, social {self.mood.social:.1f}, energy {self.mood.energy:.1f})"
        if new == "mirror":
            self.mimicking, self._mimic_since = True, now
            self._think(now, f"you're right up close... let's play mirror. I'll copy you {why}")
            out.append(Action("sound", "mirror_start", 2))
        elif new == "mime":
            self._think(now, f"I want to play... Simon says! {why}")
        elif new == "sing":
            self._think(now, f"I feel a song coming on {why}")
        elif new == "look_around":
            self._think(now, f"what else is around here? {why}")
            self._next_glance = now
        elif new == "rest":
            self._think(now, f"so tired... resting {why}")
            out.append(Action("gesture", "droop", 1))
            if now - self._last_sleepy > 120.0:  # the yawn-and-droop noise, at most once in a couple of minutes
                self._last_sleepy = now
                out.append(Action("sound", "sleepy", 1))
        elif new == "ask_attention":
            self._next_nag = now
            self._think(now, f"I want some attention {why}")
        elif new == "watch":
            self._think(now, f"someone's here, watching them {why}")
        elif new == "hangout":
            self._think(now, f"just hanging out {why}")
        return out

    def _ear_touched(self, i: int, now: float) -> list[Action]:
        """An antenna ("ear") was pushed, and it is not being petted.

        A little game of keep-away: the touched antenna moves somewhere else and stays there, with a
        giggle. Keep it up and it is not in the mood: the antenna goes forward over the head and stays.
        Disturb it there and the other antenna bats at the hand with a nuh-uh-uh; then both come back
        round, the gaze drops and the head pushes forward into the hand, asking for a pet instead.
        """
        side = "left" if i == 1 else "right"
        self._ear_tickles = self._ear_tickles + 1 if now - self._last_ear_tickle < EAR_WINDOW_S else 1
        self._last_ear_tickle = now
        self.mood.social += 0.03
        if self._ear_tucked is not None:
            # Every poke while it sulks gets batted: the make-up nuzzle waits until they have stopped.
            other = 1 - self._ear_tucked
            self._ear_seq_at = now + EAR_MAKEUP_S
            if now - self._last_swat < EAR_SWAT_S:
                return []  # the taps of the last swat are still landing
            self._last_swat = now
            self._think(now, f"nuh-uh-uh! I said leave it (batting with my {'left' if other == 1 else 'right'} ear)")
            return [Action("sound", "no_no", 3), Action("gesture", f"swat:{'+' if other == 1 else '-'}", 3)]
        if self._ear_tickles >= EAR_TUCK_AFTER:
            self._ear_tucked = i
            self._think(now, f"my {side} ear AGAIN. not in the mood. it's going over my head, leave it")
            return [Action("sound", "annoyed", 3), Action("ears", f"tuck:{i}", 3)]
        self._think(now, f"eek, my {side} ear! can't catch it")
        return [Action("sound", self.rng.choice(["giggle", "ticklish"]), 2), Action("ears", f"away:{i}", 3), Action("gesture", f"flinch:{'+' if i == 1 else '-'}", 2)]

    def _body_ignored(self, yaw: float, now: float) -> bool:
        self._body_ignore = [(y, until) for y, until in self._body_ignore if until > now]
        return any(abs(yaw - y) < BODY_IGNORE_DEG for y, _ in self._body_ignore)

    def _nag(self, obs: Observation, now: float) -> list[Action]:
        """ask_attention: complain alone, beg when someone is near, every 20 s or so."""
        company = obs.face is not None
        if now < self._next_nag and not (company and not self._nag_company):
            return []  # (someone turning up is worth begging at once)
        self._next_nag, self._nag_company = now + self.rng.uniform(15.0, 25.0), company
        if company:
            self._think(now, "hey! over here! play with me?")
            return [Action("sound", "excited", 2), Action("gesture", "bounce", 2)]
        self._think(now, "anyone...? so lonely")
        self._last_lonely = now
        return [Action("sound", "lonely", 1), Action("gesture", "droop", 1)]

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

        # -- drives drift ----------------------------------------------------------------
        self.mood.tick(dt, self.state, self.activity, obs.face is not None)
        if obs.touched or obs.petted or obs.scratched:
            self.mood.boredom -= BORED_TOUCH
        if obs.loud_yaw_deg is not None:
            self.mood.curiosity -= CURIOSITY_LOUD
        if obs.face is not None and obs.face.track_id != self._engaged_track:
            self.mood.boredom -= BORED_NEW_FACE  # a new face is the most interesting thing that can happen
            self.mood.curiosity -= CURIOSITY_NEW_FACE
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
            elif obs.petting:
                # An ear played with while being petted is an ear massage: lean in, no flinch.
                side = "left" if obs.touched_side == 1 else "right"
                self._think(now, f"mm, my {side} ear... keep going")
                actions.append(Action("sound", self.rng.choice(["purr", "content"]), 3))
                actions.append(Action("gesture", "lean", 3))
                self.mood.social += 0.05
                if self._engaged_person is not None:
                    self.memory.add_pet(self._engaged_person)
            else:
                actions += self._ear_touched(obs.touched_side, now)

        if self._ear_seq_at and now >= self._ear_seq_at:
            # after the swatting, once they have stopped: the top antenna comes back round, the touched one comes down,
            # and it asks for a pet instead
            self._ear_seq_at = 0.0
            self._ear_tucked = None
            self._ear_tickles = 0
            self._think(now, "...okay, okay. pet me instead?")
            actions += [Action("ears", "clear", 3), Action("gesture", "nuzzle", 3), Action("sound", "curious", 2)]
        elif self._ear_tucked is not None and now - self._last_ear_tickle > EAR_TUCK_S:
            self._ear_tucked = None  # left alone long enough: the antenna has come back down on its own (motion.ears_tuck)
            self._ear_tickles = 0

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

        if obs.waved is not None and awake and self.state != "HELD" and now - self._last_wave > WAVE_COOLDOWN_S:
            self._last_wave = now
            self._last_interaction = now
            # wave back with the mirrored antenna: their right hand is on our left
            self._think(now, f"they're waving their {obs.waved} hand at me! *waves back*")
            actions.append(Action("sound", "hello_friend" if self._engaged_person is not None else "happy", 3))
            actions.append(Action("gesture", f"wave:{'+' if obs.waved == 'right' else '-'}", 3))
            self.mood.social += 0.05
            self.mood.clamp()
            if self._engaged_person is not None:
                self.memory.add_attention(self._engaged_person, 2.0)

        if obs.hugged and awake and self.state != "HELD" and now - self._last_hug > HUG_COOLDOWN_S:
            self._last_hug = now
            self._last_interaction = now
            self._think(now, "arms out... a hug! *nuzzles in*")
            actions.append(Action("sound", "coo", 3))
            actions.append(Action("gesture", "hug", 3))
            self.mood.social += 0.15
            self.mood.clamp()
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
            if self._sneeze_show_until and now >= self._sneeze_show_until:
                self._sneeze_show_until = 0.0
                if obs.face.smile >= 0.8:
                    self._think(now, "they're smiling at my sneeze... hehe")
                    actions.append(Action("sound", "giggle", 2))
                    actions.append(Action("gesture", "wiggle", 2))

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

        # -- what to be doing ----------------------------------------------------------------
        if self.state in ("IDLE", "SEARCHING", "ENGAGED"):
            actions += self._choose(obs, now)
            if self.activity == "ask_attention":
                actions += self._nag(obs, now)
        elif self.activity != "hangout":
            if self.activity == "mirror" and self.mimicking:
                self.mimicking = False
            self._activity_prev, self.activity, self._activity_since = self.activity, "hangout", now

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
                    self.attention.looked(face.yaw_deg, now)

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

                # The mirror game is chosen by the activity layer (a close face for a couple of seconds makes it
                # available); it ends when they back away, after a minute, or when a dance starts.
                if self.mimicking and (face.area_frac < t.mimic_exit_area or now - self._mimic_since > t.mimic_max_s or obs.dance_bpm > 0 or obs.grooving):
                    self.mimicking = False
                    self._think(now, "mirror game over")
                    actions.append(Action("sound", "mirror_end", 2))
                    actions.append(Action("gesture", "wiggle", 1))
                    self._end_activity(now, cooldown=60.0)
                if self.mimicking:
                    actions.append(Action("mimic", f"{face.head_yaw_deg:.1f},{face.head_pitch_deg:.1f},{face.roll_deg:.1f}", 0))
                    self._next_react = now + 5.0  # no micro-reactions while mirroring

                # Mirror them: nod back at a nod, shake back at a shake (tilt is mirrored continuously by the body).
                self._face_hist.append((now, face.yaw_deg, face.pitch_deg))
                # Not while they (or the music) are moving to a beat: a bob that keeps going is dancing, not a nod.
                if not self.mimicking and now - self._last_mimic > t.mimic_cooldown and obs.dance_bpm == 0 and obs.music_bpm == 0 and not obs.grooving:
                    mimic = self._detect_nod_or_shake(now)
                    if mimic is not None:
                        self._last_mimic = now
                        self._face_hist.clear()
                        self._think(now, f"they {'nodded' if mimic == 'nod' else 'shook their head'}... me too!")
                        actions.append(Action("sound", "happy" if mimic == "nod" else "curious", 2))
                        actions.append(Action("gesture", mimic, 2))

                # Periodic micro-reactions while someone is around (not while we are dancing with them).
                if obs.dance_bpm > 0 or obs.grooving:
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
                    if self.mood.energy < 0.25 or self.activity == "rest":
                        if now - self._last_sleepy < 120.0:
                            continue_quietly = True  # tired: a droop without the noise
                        else:
                            continue_quietly = False
                            self._last_sleepy = now
                        actions.append(Action("gesture", "droop", 1))
                        if not continue_quietly:
                            actions.append(Action("sound", "sleepy", 1))
                    else:
                        actions.append(Action("sound", choice, 1))
                        actions.append(Action("gesture", gesture, 1))

            elif (obs.body is not None and self.state != "ENGAGED" and now - self._last_face_time > 2.5 and obs.busy != "mime"
                  and not self._body_ignored(obs.body.yaw_deg, now)):
                # Someone's torso is in view but not their face: look up to where the head should be.
                # A torso has to persist before it is believed (the detector flickers on furniture), and if a
                # minute of staring finds no face, that spot is not a person and is ignored for a while.
                self._close_since = 0.0
                if self._body_first == 0.0:
                    self._body_first = now
                self._body_last = now
                if now - self._body_first < BODY_CONFIRM_S:
                    # not believed yet, but worth pausing for: stop the sweep on it so it does not roll by
                    self.gaze = (obs.body.yaw_deg, obs.body.pitch_deg)
                    self._look_until = max(self._look_until, now + 1.5)
                    self._next_glance = max(self._next_glance, now + 1.5)
                elif now - self._body_since > BODY_GIVE_UP_S and self._body_since != 0.0:
                    self._body_ignore.append((obs.body.yaw_deg, now + BODY_IGNORE_S))
                    self._think(now, f"no face up there after {BODY_GIVE_UP_S:.0f} s... that's not a person. ignoring it")
                    self._body_since, self._body_first = 0.0, 0.0
                    self.gaze = None
                    if self.state == "SEARCHING":
                        self._enter("IDLE", now)
                        self._next_glance = now + 0.5
                else:
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
                if obs.body is None and now - self._body_last > 1.5:
                    self._body_first = 0.0  # a torso gone for a while (not a blink of the detector): start over
                    self._body_since = 0.0
                self._mimic_candidate_since = 0.0
                if self.mimicking and now - self._last_face_time > 1.0:
                    self.mimicking = False
                    self._think(now, "mirror game over (lost you)")
                    actions.append(Action("sound", "mirror_end", 2))
                    self._end_activity(now, cooldown=60.0)
                if self.state == "ENGAGED" and (obs.dance_bpm > 0 or obs.grooving or obs.busy in ("mime", "gesture", "turn")):
                    pass  # mid-dance, a Simon says move, a solo gesture (bow, sneeze) or a keypad turn: the camera is moving, keep the gaze
                    self._last_face_time = max(self._last_face_time, now - 0.5)  # and the face-lost clock waits too
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
                        self._look_until = 0.0
                        actions.append(Action("gesture", "perk", 2))
                        actions.append(Action("sound", "curious", 1))
                        self._next_glance = now + 2.0
                    elif now >= self._next_glance and self.activity != "rest":
                        # Look around properly: a wide gaze target, so the body turns too and it can see
                        # someone standing right beside it, off camera. (A head-only glance never turned the body.)
                        if self.activity == "look_around":
                            # curious: go somewhere it has not looked lately, and LINGER there: the body has to
                            # get round (it carries the coarse turn) and the camera needs a few seconds on a
                            # scene to find a face in it. A quick flick saw nothing.
                            yaw = self.attention.stalest(now, self.rng)
                            self._next_glance = now + self.rng.uniform(5.0, 8.0)
                            self._look_until = self._next_glance
                        else:
                            yaw = self.rng.choice((-1.0, 1.0)) * self.rng.uniform(35.0, 100.0)
                            self._next_glance = now + self.rng.uniform(t.idle_glance_min, t.idle_glance_max)
                            self._look_until = now + self.rng.uniform(2.5, 4.0)
                        if self.attention.looked(yaw, now):
                            self.mood.curiosity -= CURIOSITY_NEW_SECTOR  # a real change of scene spends curiosity
                        self._look_at = (yaw, self.rng.uniform(-6.0, 10.0))
                        actions.append(Action("gesture", "glance:" + ("+" if yaw > 0 else "-"), 0))
                    if now < self._look_until:
                        self.gaze = self._look_at
                    alone_for = now - self._last_interaction
                    if self.mood.energy < 0.3 and not self._nodding_off and alone_for > 30.0 and self.rng.random() < dt * 0.02:
                        self._nodding_off = True
                        actions.append(Action("gesture", "nod_off", 1))
                        if now - self._last_sleepy > 120.0:
                            self._last_sleepy = now
                            actions.append(Action("sound", "sleepy", 1))
                    if alone_for > t.sleep_after or self.mood.energy < 0.08:
                        actions.append(Action("sound", "yawn", 3))
                        actions.append(Action("sleep", "tired", 5))
                        self._think(now, "so tired... going to sleep" if self.mood.energy < 0.08 else f"nobody around for {alone_for / 60:.0f} min, dozing off")
                        self._enter("SLEEPING", now)
                        self._face_first_seen = 0.0
                    elif alone_for > t.lonely_after and now - self._last_lonely > t.lonely_repeat and self.activity != "ask_attention":
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
                self._sneeze_show_until = now + 10.6  # motion.SNEEZE_S; after it, a smile in the audience gets a giggle
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
            "activity": self.activity,
            "runner_up": self.runner_up,
            "drives": self.mood.as_dict(),
        }

    def mind(self, now: float) -> dict:
        """Everything driving the next decision, for the 'inside the mind' page."""
        t = self.timers
        alone = now - self._last_interaction
        return {
            **self.status(),
            "state_for_s": round(now - self._state_since, 1),
            "activity_for_s": round(now - self._activity_since, 1),
            "margin": round(self._margin, 2),
            "scores": self._scores,
            "cooldowns": {k: round(v - now) for k, v in self._cool.items() if v > now},
            "next_decision_in_s": round(max(0.0, self._last_ask + ASK_CEILING_S - now)),
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
