from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class KokoroBackend:
    model_dir: Path
    _kokoro: object = None  # kokoro_onnx.Kokoro, loaded lazily

    def _get_kokoro(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(
                str(self.model_dir / "kokoro-v1.0.onnx"),
                str(self.model_dir / "voices-v1.0.bin"),
            )
        return self._kokoro

    def synthesize_audio(self, text: str, voice: str = "af_heart") -> tuple[np.ndarray, int]:
        kokoro = self._get_kokoro()
        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        return samples, sample_rate

    def speak(self, text: str, output_path: Path | None = None, voice: str = "af_heart") -> None:
        audio, sample_rate = self.synthesize_audio(text, voice=voice)
        if output_path:
            sf.write(str(output_path), audio, sample_rate)


def load(model_dir: Path) -> KokoroBackend:
    onnx_path = model_dir / "kokoro-v1.0.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"kokoro model not found at {onnx_path}. Run: onda pull kokoro/en-v1")
    return KokoroBackend(model_dir=model_dir)
