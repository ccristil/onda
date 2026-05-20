from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from onda.downloader import get_piper_binary
from onda.utils import play_wav


@dataclass
class PiperBackend:
    model_dir: Path

    def speak(self, text: str, output_path: Path | None = None) -> None:
        piper_bin = get_piper_binary()
        model_file = next(self.model_dir.glob("*.onnx"))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_wav = Path(tmpdir) / "output.wav"
            subprocess.run(
                [
                    str(piper_bin),
                    "--model", str(model_file),
                    "--output_file", str(tmp_wav),
                ],
                input=text.encode(),
                check=True,
                capture_output=True,
            )
            if output_path:
                shutil.copy2(tmp_wav, output_path)
            else:
                play_wav(tmp_wav)


def load(model_dir: Path) -> PiperBackend:
    onnx_files = list(model_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file found in {model_dir}")
    return PiperBackend(model_dir=model_dir)
