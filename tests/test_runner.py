import json
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock, call

from onda.runner import run


@pytest.fixture
def registry_with_amy(tmp_path, monkeypatch):
    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps([{
        "name": "en_US-amy-low",
        "backend": "piper",
        "onnx_url": "https://example.com/amy.onnx",
        "config_url": "https://example.com/amy.onnx.json",
        "size_mb": 5,
        "description": "Fast.",
    }]))
    monkeypatch.setattr("onda.registry.MODELS_PATH", models_file)
    monkeypatch.setattr("onda.registry.ONDA_DIR", tmp_path)

    # Create model dir with expected files
    model_path = tmp_path / "models" / "en_US-amy-low"
    model_path.mkdir(parents=True)
    (model_path / "en_US-amy-low.onnx").touch()
    (model_path / "en_US-amy-low.onnx.json").touch()
    return tmp_path


def test_run_streams_audio_for_each_chunk(registry_with_amy, mocker):
    mock_backend = MagicMock()
    mock_backend.synthesize_audio.return_value = (np.zeros(100, dtype="float32"), 22050)
    mocker.patch("onda.runner._load_backend", return_value=mock_backend)
    mocker.patch("onda.runner.chunk_text", return_value=["chunk one", "chunk two"])
    mocker.patch("sounddevice.OutputStream")  # MagicMock auto-supports __enter__/__exit__

    run("en_US-amy-low", "chunk one chunk two")

    assert mock_backend.synthesize_audio.call_count == 2


def test_run_raises_on_unknown_model(registry_with_amy):
    with pytest.raises(SystemExit):
        run("nonexistent-model", "hello")


def test_run_raises_when_not_downloaded(tmp_path, monkeypatch):
    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps([{
        "name": "en_US-amy-low",
        "backend": "piper",
        "onnx_url": "https://example.com/amy.onnx",
        "config_url": "https://example.com/amy.onnx.json",
        "size_mb": 5,
        "description": "Fast.",
    }]))
    monkeypatch.setattr("onda.registry.MODELS_PATH", models_file)
    monkeypatch.setattr("onda.registry.ONDA_DIR", tmp_path)
    # Don't create model files — is_downloaded should return False

    with pytest.raises(SystemExit):
        run("en_US-amy-low", "hello")


def test_run_with_output_path_concatenates_wavs(registry_with_amy, tmp_path, mocker):
    # Chunks are written to separate WAVs; runner concatenates them
    chunk_wavs = []

    def fake_speak(text, output_path=None):
        if output_path:
            sf.write(str(output_path), np.zeros(100, dtype="float32"), 22050)
            chunk_wavs.append(output_path)

    mock_backend = MagicMock()
    mock_backend.speak.side_effect = fake_speak
    mocker.patch("onda.runner._load_backend", return_value=mock_backend)
    mocker.patch("onda.runner.chunk_text", return_value=["chunk one", "chunk two"])

    out = tmp_path / "final.wav"
    run("en_US-amy-low", "chunk one chunk two", output_path=out)

    assert out.exists()
    data, _ = sf.read(str(out))
    assert len(data) == 200  # 100 samples per chunk × 2 chunks
