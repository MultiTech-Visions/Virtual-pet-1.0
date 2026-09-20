"""Persistent face memory: who the pet has met and how much history it has with them.

Storage is a single JSON file (one read at startup, debounced writes). Each
person holds up to ``MAX_EMBEDDINGS`` face embeddings so recognition improves
as lighting / angle vary across the day. No names are ever stored; people are
identified only by an opaque id and their relationship stats.

Matching is cosine similarity against every stored embedding; the best match
above ``match_threshold`` wins. SFace's published cosine threshold is 0.363;
we default a bit stricter because a festival crowd is a lot of strangers.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclasses_fields
from pathlib import Path

import numpy as np

MAX_EMBEDDINGS = 8
SCHEMA_VERSION = 1


@dataclass
class Person:
    """Everything the pet remembers about one human."""

    person_id: int
    embeddings: list[list[float]]
    first_seen: float
    last_seen: float
    encounters: int = 0  # distinct visits (separated by ENCOUNTER_GAP)
    attention_seconds: float = 0.0  # cumulative time the pet spent engaged with them
    pets: int = 0  # antenna touches / cuddles while this person was engaged
    holds: int = 0  # times picked up while this person was engaged
    kandi: int = 0  # PLUR handshakes traded with this person
    affection: float = 0.0  # 0..1, grows with positive interaction, decays slowly
    nickname_seed: int = field(default=0)  # stable seed so the pet's "song" for them is consistent

    def tier(self) -> str:
        """Relationship tier used by the behavior layer to pick greetings."""
        if self.encounters >= 4 or self.affection >= 0.6:
            return "bestie"
        if self.encounters >= 2 or self.affection >= 0.25:
            return "friend"
        return "acquaintance"


ENCOUNTER_GAP = 120.0  # seconds away before a re-sighting counts as a new encounter


class FaceMemory:
    """JSON-backed store of known people and their relationship stats."""

    def __init__(self, path: Path, match_threshold: float = 0.42, save_interval: float = 5.0) -> None:
        self.path = Path(path)
        self.match_threshold = match_threshold
        self.save_interval = save_interval
        self.people: dict[int, Person] = {}
        self._next_id = 1
        self._dirty = False
        self._last_save = 0.0
        self._lock = threading.Lock()
        if self.path.exists():
            self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        with self.path.open() as f:
            data = json.load(f)
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Memory file schema {data['schema_version']} != {SCHEMA_VERSION}: {self.path}")
        fields = {f.name for f in dataclasses_fields(Person)}
        self.people = {int(k): Person(**{a: b for a, b in v.items() if a in fields}) for k, v in data["people"].items()}
        self._next_id = data["next_id"]

    def save(self, force: bool = False) -> None:
        """Write to disk if dirty and the debounce interval has elapsed (or force). Thread-safe."""
        with self._lock:
            now = time.time()
            if not self._dirty:
                return
            if not force and now - self._last_save < self.save_interval:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "next_id": self._next_id,
                "people": {str(k): asdict(v) for k, v in self.people.items()},
            }
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w") as f:
                json.dump(payload, f)
            tmp.replace(self.path)
            self._dirty = False
            self._last_save = now

    # ------------------------------------------------------------------ matching
    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            raise ValueError("Zero-length embedding cannot be normalised")
        return v / norm

    def match(self, embedding: np.ndarray) -> tuple[Person | None, float]:
        """Return (best person or None, best cosine similarity)."""
        q = self._normalize(embedding)
        best: Person | None = None
        best_sim = -1.0
        for person in self.people.values():
            mat = np.asarray(person.embeddings, dtype=np.float32)
            sims = mat @ q
            s = float(np.max(sims))
            if s > best_sim:
                best_sim = s
                best = person
        if best is None or best_sim < self.match_threshold:
            return None, best_sim
        return best, best_sim

    # ------------------------------------------------------------------ thumbnails
    def faces_dir(self) -> Path:
        return self.path.parent / "faces"

    def thumbnail_path(self, person_id: int) -> Path:
        return self.faces_dir() / f"{person_id}.jpg"

    def set_thumbnail(self, person: Person, jpeg: bytes) -> None:
        """Store a small face crop for the People page (kept only on the robot)."""
        self.faces_dir().mkdir(parents=True, exist_ok=True)
        self.thumbnail_path(person.person_id).write_bytes(jpeg)

    def delete(self, person_id: int) -> None:
        """Forget one person. Raises KeyError if unknown."""
        del self.people[person_id]
        self._dirty = True
        thumb = self.thumbnail_path(person_id)
        if thumb.exists():
            thumb.unlink()

    def merge(self, keep_id: int, other_id: int) -> Person:
        """Fold ``other`` into ``keep``: embeddings and stats combine, ``other`` is deleted."""
        if keep_id == other_id:
            raise ValueError("cannot merge a person into themselves")
        keep, other = self.people[keep_id], self.people[other_id]
        # Keep the enrolment view of each, then the freshest of the rest, within the cap.
        merged = [keep.embeddings[0], other.embeddings[0]] + keep.embeddings[1:] + other.embeddings[1:]
        keep.embeddings = merged[:MAX_EMBEDDINGS]
        keep.first_seen = min(keep.first_seen, other.first_seen)
        keep.last_seen = max(keep.last_seen, other.last_seen)
        keep.encounters += other.encounters
        keep.attention_seconds += other.attention_seconds
        keep.pets += other.pets
        keep.holds += other.holds
        keep.affection = min(1.0, max(keep.affection, other.affection) + 0.5 * min(keep.affection, other.affection))
        self.delete(other_id)
        return keep

    def enroll(self, embedding: np.ndarray, now: float) -> Person:
        """Create a new person from one embedding."""
        q = self._normalize(embedding)
        person = Person(
            person_id=self._next_id,
            embeddings=[q.tolist()],
            first_seen=now,
            last_seen=now,
            encounters=1,
            nickname_seed=self._next_id * 7919,
        )
        self.people[person.person_id] = person
        self._next_id += 1
        self._dirty = True
        return person

    def reinforce(self, person: Person, embedding: np.ndarray, similarity: float) -> None:
        """Add a fresh embedding for a known person when it adds diversity."""
        # Only keep views that differ from what we already have; a 0.9+ match adds nothing.
        if similarity >= 0.9:
            return
        q = self._normalize(embedding)
        person.embeddings.append(q.tolist())
        if len(person.embeddings) > MAX_EMBEDDINGS:
            # Drop the oldest, keeping the original enrolment view at index 0.
            del person.embeddings[1]
        self._dirty = True

    # ------------------------------------------------------------------ relationship stats
    def sighted(self, person: Person, now: float) -> bool:
        """Record a sighting; returns True when this starts a new encounter."""
        new_encounter = (now - person.last_seen) > ENCOUNTER_GAP
        if new_encounter:
            person.encounters += 1
        person.last_seen = now
        self._dirty = True
        return new_encounter

    def add_attention(self, person: Person, seconds: float) -> None:
        person.attention_seconds += seconds
        person.affection = min(1.0, person.affection + seconds * 0.002)
        self._dirty = True

    def add_pet(self, person: Person) -> None:
        person.pets += 1
        person.affection = min(1.0, person.affection + 0.05)
        self._dirty = True

    def add_hold(self, person: Person) -> None:
        person.holds += 1
        person.affection = min(1.0, person.affection + 0.08)
        self._dirty = True

    def add_kandi(self, person: Person) -> None:
        """A whole PLUR handshake, which is about the biggest thing anyone does with it: worth a lot."""
        person.kandi += 1
        person.affection = min(1.0, person.affection + 0.25)
        self._dirty = True

    def summary(self) -> dict:
        """Small dict for the status page."""
        return {
            "people": len(self.people),
            "tiers": {t: sum(1 for p in self.people.values() if p.tier() == t) for t in ("acquaintance", "friend", "bestie")},
            "top": sorted(
                (
                    {
                        "id": p.person_id,
                        "tier": p.tier(),
                        "encounters": p.encounters,
                        "attention_s": round(p.attention_seconds, 1),
                        "pets": p.pets,
                        "holds": p.holds,
                        "kandi": p.kandi,
                        "affection": round(p.affection, 2),
                        "first_seen": p.first_seen,
                        "last_seen": p.last_seen,
                        "has_face": self.thumbnail_path(p.person_id).exists(),
                        "views": len(p.embeddings),
                    }
                    for p in self.people.values()
                ),
                key=lambda d: d["affection"],
                reverse=True,
            ),
        }
