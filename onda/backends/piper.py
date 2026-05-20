from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from onda.utils import play_wav


@dataclass
class PiperBackend:
    model_dir: Path
    _voice: object = None  # piper.PiperVoice, loaded lazily

    def _get_voice(self):
        if self._voice is None:
            from piper import PiperVoice
            model_file = next(self.model_dir.glob("*.onnx"))
            self._voice = PiperVoice.load(str(model_file))
        return self._voice

    def speak(self, text: str, output_path: Path | None = None) -> None:
        voice = self._get_voice()
        chunks = list(voice.synthesize(text))
        if not chunks:
            return

        # Concatenate all audio chunks
        audio = np.concatenate([c.audio_float_array for c in chunks])
        sample_rate = chunks[0].sample_rate

        if output_path:
            sf.write(str(output_path), audio, sample_rate)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                sf.write(str(tmp_path), audio, sample_rate)
                play_wav(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)


def load(model_dir: Path) -> PiperBackend:
    onnx_files = list(model_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file found in {model_dir}")
    return PiperBackend(model_dir=model_dir)
