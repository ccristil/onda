from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import typer

from onda.registry import ModelEntry, get_model, is_downloaded, model_dir
from onda.utils import chunk_text


def _load_backend(model: ModelEntry):
    d = model_dir(model.name)
    if model.backend == "piper":
        from onda.backends.piper import load
        return load(d)
    if model.backend == "kokoro":
        from onda.backends.kokoro import load
        return load(d)
    raise ValueError(f"Unknown backend: {model.backend}")


def run(model_name: str, text: str, output_path: Path | None = None) -> None:
    """Synthesize text with the named model, playing or saving the result."""
    model = get_model(model_name)
    if model is None:
        typer.echo(
            f"Error: Model '{model_name}' not found in registry. Run: onda list",
            err=True,
        )
        raise SystemExit(1)

    if not is_downloaded(model_name):
        typer.echo(
            f"Error: Model '{model_name}' is not downloaded. Run: onda pull {model_name}",
            err=True,
        )
        raise SystemExit(1)

    backend = _load_backend(model)
    chunks = chunk_text(text)

    if output_path:
        arrays: list[np.ndarray] = []
        samplerate: int | None = None
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, chunk in enumerate(chunks):
                chunk_path = Path(tmpdir) / f"chunk_{i}.wav"
                backend.speak(chunk, output_path=chunk_path)
                data, sr = sf.read(str(chunk_path))
                arrays.append(data)
                samplerate = sr
        combined = np.concatenate(arrays)
        sf.write(str(output_path), combined, samplerate)
    else:
        import queue
        import threading
        import sounddevice as sd

        audio_queue: queue.Queue = queue.Queue(maxsize=2)
        exc_holder: list[BaseException] = []

        def _producer() -> None:
            try:
                for chunk in chunks:
                    audio, sr = backend.synthesize_audio(chunk)
                    audio_queue.put((audio, sr))
            except Exception as exc:
                exc_holder.append(exc)
            finally:
                audio_queue.put(None)

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        while True:
            item = audio_queue.get()
            if item is None:
                break
            audio, sr = item
            if audio.size > 0:
                sd.play(audio, sr)
                sd.wait()

        producer_thread.join()

        if exc_holder:
            raise exc_holder[0]
