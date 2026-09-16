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


class NameSpotter:
    """Feed float32 mono at 16 kHz; ``poll()`` returns recognised words."""

    def __init__(self, model_dir: Path, sample_rate: int = 16000) -> None:
        import vosk  # imported here so the rest of the package has no hard dependency

        vosk.SetLogLevel(-1)
        if not Path(model_dir).is_dir():
            raise FileNotFoundError(f"Vosk model directory not found: {model_dir}")
        self._model = vosk.Model(str(model_dir))
        self._rec = vosk.KaldiRecognizer(self._model, sample_rate, json.dumps(GRAMMAR))
        self._rec.SetWords(False)
        self._pending: list[str] = []
        self._name_emitted = False  # once per utterance, so partial results do not re-fire it

    def push(self, mono: np.ndarray) -> None:
        pcm = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16).tobytes()
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
