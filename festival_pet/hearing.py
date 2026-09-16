"""Offline name / command spotting with Vosk, gated by the mic array's speech flag.

Grammar-constrained recognition on the small English model is cheap and
robust for a handful of words. Everything else decodes to ``[unk]`` and is
ignored. The recogniser only runs while ``speech_detected`` is true (the
ReSpeaker gives us that for free), so at a loud campsite it idles most of
the time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NAME = "reachy"
# "reachy" is not in the model's lexicon, so we spot in-vocabulary sound-alikes.
# Verified against synthesized speech in several voices: "ricci" fires on
# "reachy" every time, "reach it" catches a British reading.
NAME_ALIASES = ("ricci", "richie", "ritchie", "reach it", "reach eat")
# Decoys absorb near-misses so they do not get forced into a name alias.
DECOYS = ("peachy", "reach", "beach", "teach")
COMMANDS = ("dance", "hello", "hi", "good", "sleep")
GRAMMAR = [*NAME_ALIASES, *DECOYS, *COMMANDS, "[unk]"]
_WORDS = set(COMMANDS)


# What people say to a robot pet, and how it should feel about it. Matched on the
# free-vocabulary transcript (which is rough, so we key on robust little words).
INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("praise", ("good boy", "good girl", "good robot", "good job", "well done", "clever", "smart", "nice one", "love you", "i love")),
    ("cute", ("cute", "adorable", "sweet", "so little", "precious", "aww", "awesome", "amazing", "cool")),
    ("greeting", ("hello", "hi there", "hey there", "good morning", "good evening", "how are you", "what's up", "whats up")),
    ("farewell", ("bye", "goodbye", "see you", "later", "good night", "goodnight")),
    ("question", ("what", "who", "why", "how", "where", "can you", "do you", "are you", "is it", "does it")),
    ("laugh", ("haha", "ha ha", "lol", "funny", "hilarious")),
    ("scold", ("bad robot", "stop it", "no no", "stop that", "shut up", "go away", "naughty")),
    ("sad", ("sad", "sorry", "poor thing", "miss you")),
    ("photo", ("picture", "photo", "selfie", "camera", "smile", "cheese")),
    ("come", ("come here", "over here", "look at me", "look here", "this way")),
)


def intents_in(text: str) -> list[str]:
    """Which intents a transcript contains, in INTENTS order (no duplicates)."""
    padded = f" {text.lower()} "
    return [name for name, cues in INTENTS if any(f" {c} " in padded for c in cues)]


class NameSpotter:
    """Feed float32 mono at 16 kHz; ``poll()`` returns recognised name/command words.

    With ``transcribe=True`` a second, free-vocabulary recogniser runs on the same
    audio so the pet can "understand" the gist of what people say (see ``INTENTS``);
    finished sentences come back from ``poll_transcript()``.
    """

    def __init__(self, model_dir: Path, sample_rate: int = 16000, transcribe: bool = True) -> None:
        import vosk  # imported here so the rest of the package has no hard dependency

        vosk.SetLogLevel(-1)
        if not Path(model_dir).is_dir():
            raise FileNotFoundError(f"Vosk model directory not found: {model_dir}")
        self._model = vosk.Model(str(model_dir))
        self._rec = vosk.KaldiRecognizer(self._model, sample_rate, json.dumps(GRAMMAR))
        self._rec.SetWords(False)
        self._free = vosk.KaldiRecognizer(self._model, sample_rate) if transcribe else None
        self._pending: list[str] = []
        self._sentences: list[str] = []
        self._name_emitted = False  # once per utterance, so partial results do not re-fire it

    def push(self, mono: np.ndarray) -> None:
        pcm = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        if self._free is not None and self._free.AcceptWaveform(pcm):
            self._sentence(json.loads(self._free.Result())["text"])
        if self._rec.AcceptWaveform(pcm):
            self._collect(json.loads(self._rec.Result())["text"])
            self._name_emitted = False
        else:
            partial = json.loads(self._rec.PartialResult())["partial"]
            # Partial results let us react before the utterance ends.
            if not self._name_emitted and self._contains_name(partial):
                self._name_emitted = True
                self._pending.append(NAME)

    def flush(self) -> None:
        """Call when speech stops so the last words are emitted."""
        self._collect(json.loads(self._rec.FinalResult())["text"])
        self._name_emitted = False
        if self._free is not None:
            self._sentence(json.loads(self._free.FinalResult())["text"])

    def _sentence(self, text: str) -> None:
        text = text.strip()
        if text:
            self._sentences.append(text)

    def poll_transcript(self) -> list[str]:
        out, self._sentences = self._sentences, []
        return out

    @staticmethod
    def _contains_name(text: str) -> bool:
        padded = f" {text} "
        return any(f" {alias} " in padded for alias in NAME_ALIASES)

    def _collect(self, text: str) -> None:
        if self._contains_name(text) and not self._name_emitted:
            self._pending.append(NAME)
        for w in text.split():
            if w in _WORDS and (not self._pending or self._pending[-1] != w):
                self._pending.append(w)

    def poll(self) -> list[str]:
        out, self._pending = self._pending, []
        return out
