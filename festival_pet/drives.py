"""Drives and the activity chooser: what the pet *wants*, and what it decides to be doing.

Before this the brain was purely reactive: every behaviour was a trigger, a cooldown and a dice
roll, and idle time was filled by timers. This adds the layer pocket-tank has (drives that rise
and are spent, a chosen pastime, boredom that pushes it to change) with a rule stub as the brain
instead of a language model. The rules are scores, not an if-chain, so there is always a runner-up
and a margin: a close call shows as a visible hesitation before it commits.

Four drives, all 0..1:
    energy     drains awake, recovers asleep; games cost more, rest gives a little back
    social     rises with company and touch, decays alone
    curiosity  recovers slowly; spent by looking somewhere new, a new face, a loud sound, a game
    boredom    rises while the same pastime goes on; relieved by a genuinely new one, never by
               bouncing back to the one just left; a new face or a touch helps; sleep resets it

Activities (one at a time, the thing it is committed to):
    hangout        nothing much: breathing, the odd glance
    watch          someone is here: watch them, react, greet
    look_around    alone and curious: turn to sectors it has not looked at lately
    mirror         the mirror game (copy the person's head)
    mime           the mime game (do what I do)
    sing           make up a song
    rest           low energy: droop, react less
    ask_attention  low social: complain when alone, beg when someone is near

Pure Python, fake-clock friendly, no robot imports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ACTIVITIES = ("hangout", "watch", "look_around", "mirror", "mime", "sing", "rest", "ask_attention")
LEISURE = frozenset({"hangout", "watch", "look_around"})  # the pastimes that go stale; games and songs end by themselves

BORED_PER_S = 0.007  # 0 -> 1 in about two and a half minutes on one pastime
BORED_NEW_ACTIVITY = 0.4  # relief for a pastime that is not the one just left
BORED_NEW_FACE = 0.3
BORED_TOUCH = 0.15
CURIOSITY_PER_S = 0.004  # 0 -> 1 in about four minutes
CURIOSITY_NEW_SECTOR = 0.12
CURIOSITY_NEW_FACE = 0.25
CURIOSITY_LOUD = 0.08
CURIOSITY_GAME_PER_S = 0.003
ASK_MIN_INTERVAL_S = 1.5  # between re-decisions when the situation changed
ASK_CEILING_S = 30.0  # re-decide anyway after this long
HESITATE_MARGIN = 0.06  # top two closer than this: a visible "hmm"
SECTORS = 6  # the world in front of it, -150..150 deg of yaw, for "somewhere it has not looked"
SECTOR_STALE_S = 40.0


@dataclass
class Drives:
    energy: float = 0.8  # 0 exhausted .. 1 bouncy
    social: float = 0.5  # 0 lonely .. 1 fulfilled
    curiosity: float = 0.6
    boredom: float = 0.0

    def clamp(self) -> None:
        for k in ("energy", "social", "curiosity", "boredom"):
            setattr(self, k, min(1.0, max(0.0, getattr(self, k))))

    def tick(self, dt: float, state: str, activity: str, company: bool) -> None:
        """Slow drift, once per tick. Events (touch, a new face, a game) bump the drives elsewhere."""
        if state == "SLEEPING":
            self.energy += dt * 0.01
            self.boredom = 0.0
        else:
            self.energy -= dt * (0.0006 + (0.0015 if activity in ("mime", "mirror", "sing") else 0.0))
            if activity == "rest":
                self.energy += dt * 0.0012
            self.curiosity += dt * CURIOSITY_PER_S
            if activity in ("mime", "mirror"):
                self.curiosity -= dt * CURIOSITY_GAME_PER_S
            if activity in LEISURE:
                self.boredom += dt * BORED_PER_S
            else:
                self.boredom -= dt * 0.02
        if state in ("ENGAGED", "HELD") or company:
            self.social += dt * 0.01
        else:
            self.social -= dt * 0.0015
        self.clamp()

    def as_dict(self) -> dict:
        return {k: round(getattr(self, k), 2) for k in ("energy", "social", "curiosity", "boredom")}


@dataclass
class Situation:
    """What the chooser looks at: coarse, so the signature only changes when something meaningful does."""

    person: bool = False
    close: bool = False
    beat: bool = False  # music or a dancing person: dancing is reactive, but games do not start mid-dance
    held: bool = False
    busy: str | None = None  # a game or a song the robot layer is running right now ("mime", "sing") or None
    can_sing: bool = False
    can_mime: bool = False


@dataclass
class Choice:
    activity: str
    runner_up: str | None
    margin: float
    scores: dict[str, float]


def _band(v: float) -> int:
    return 0 if v < 0.33 else 1 if v < 0.66 else 2


def signature(d: Drives, s: Situation) -> tuple:
    return (s.person, s.close, s.beat, s.held, s.busy, s.can_sing, s.can_mime, _band(d.energy), _band(d.social), _band(d.curiosity), _band(d.boredom))


def score(d: Drives, s: Situation, current: str, cool: dict[str, float], now: float, rng: random.Random) -> dict[str, float]:
    """Every activity that is possible right now, scored. Higher wins; a little dice keeps it from being a lookup table."""
    out: dict[str, float] = {}
    dice = lambda: rng.uniform(-0.06, 0.06)  # noqa: E731
    ok = lambda name: cool.get(name, -1e9) <= now  # noqa: E731
    out["rest"] = 1.6 * (1.0 - d.energy) - 0.6 + dice()
    if s.person:
        out["watch"] = 0.5 + 0.25 * (1.0 - d.social) - 0.45 * d.boredom + dice()
        if s.close and not s.beat and ok("mirror"):
            out["mirror"] = 0.64 + 0.35 * d.boredom + 0.25 * d.curiosity + dice()
        if s.can_mime and not s.beat and ok("mime"):
            out["mime"] = 0.15 + 0.6 * d.boredom + 0.3 * d.curiosity + dice()
    else:
        out["hangout"] = 0.4 - 0.3 * d.boredom + dice()
        out["look_around"] = 0.35 + 0.4 * d.curiosity + 0.3 * d.boredom + dice()
    if s.can_sing and not s.beat and ok("sing"):
        out["sing"] = 0.2 + 0.45 * d.boredom + 0.25 * d.energy + dice()
    if d.social < 0.35:
        out["ask_attention"] = 0.3 + 1.2 * (0.35 - d.social) + (0.2 if s.person else 0.0) + dice()
    # a little hysteresis: what it is doing is worth a touch more than switching for nothing
    if current in out:
        out[current] += 0.04
    return out


def choose(d: Drives, s: Situation, current: str, cool: dict[str, float], now: float, rng: random.Random) -> Choice:
    scores = score(d, s, current, cool, now, rng)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    return Choice(top[0], None if second is None else second[0], 1.0 if second is None else top[1] - second[1], {k: round(v, 2) for k, v in scores.items()})


@dataclass
class Attention:
    """Where it has looked lately, by yaw sector, so "look around" goes somewhere new."""

    seen: list[float] = field(default_factory=lambda: [-1e9] * SECTORS)

    @staticmethod
    def sector(yaw_deg: float) -> int:
        return min(SECTORS - 1, max(0, int((yaw_deg + 150.0) / 300.0 * SECTORS)))

    def looked(self, yaw_deg: float, now: float) -> bool:
        """Record a look; True if that sector was stale (a real change of scene)."""
        i = self.sector(yaw_deg)
        stale = now - self.seen[i] > SECTOR_STALE_S
        self.seen[i] = now
        return stale

    def stalest(self, now: float, rng: random.Random) -> float:
        """A yaw inside one of the two sectors it has not looked at for longest."""
        order = sorted(range(SECTORS), key=lambda i: self.seen[i])
        i = rng.choice(order[:2])
        lo = -150.0 + 300.0 * i / SECTORS
        return rng.uniform(lo + 8.0, lo + 300.0 / SECTORS - 8.0)
